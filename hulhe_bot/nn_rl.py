"""Neural-network residual fine-tuner (Deep-CFR–inspired).

Drop-in replacement for the tabular ResidualFineTuner.  Uses a tiny
numpy-only MLP with experience replay so it runs on CPU in minutes.
Activated by setting  fine_tune_mode = "nn"  in the config JSON.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import numpy as np

from .baselines import LoosePassiveAgent, PolicyAgent, TightAggressiveAgent
from .config import AbstractHULHEConfig
from .evaluator import Evaluator
from .models import Action, InfoSetKey, Observation, PolicyArtifact, Street
from .policy import PolicyRuntime, normalize_distribution, softmax_preferences

# ---------------------------------------------------------------------------
# Action ↔ index mapping (used by the network's output layer)
# ---------------------------------------------------------------------------
ACTION_INDEX: dict[Action, int] = {
    Action.FOLD: 0,
    Action.CHECK: 1,
    Action.CALL: 2,
    Action.RAISE: 3,
}
NUM_ACTIONS = 4

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
STREET_INDEX: dict[str, int] = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
FEATURE_DIM = 10


def extract_features(obs: Observation, config: AbstractHULHEConfig) -> np.ndarray:
    """Convert an Observation into a fixed-size feature vector (dim=10)."""
    feats = np.zeros(FEATURE_DIM, dtype=np.float64)
    # Street one-hot (4 dims)
    feats[STREET_INDEX.get(obs.street.value, 0)] = 1.0
    # Position (1 dim)
    feats[4] = float(obs.acting_player)
    # Bucket percentile / equity (1 dim, [0,1])
    feats[5] = obs.bucket_percentile
    # Pot normalised by starting stack (1 dim)
    feats[6] = obs.pot / max(1, config.starting_stack)
    # To-call normalised by big bet (1 dim)
    feats[7] = obs.to_call / max(1, config.big_bet)
    # History length (1 dim, normalised)
    feats[8] = len(obs.history_id) / 20.0
    # Raise count in history (1 dim, normalised)
    feats[9] = obs.history_id.count("r") / 8.0
    return feats


# ---------------------------------------------------------------------------
# Pure-numpy MLP
# ---------------------------------------------------------------------------
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(pre: np.ndarray) -> np.ndarray:
    return (pre > 0).astype(np.float64)


class SimpleMLP:
    """Tiny MLP in pure numpy — no GPU required."""

    def __init__(self, layer_sizes: list[int], seed: int = 42):
        rng = np.random.RandomState(seed)
        self.layer_sizes = list(layer_sizes)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(rng.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    # ---- forward / predict ------------------------------------------------
    def forward(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        pre_activations: list[np.ndarray] = []
        activations: list[np.ndarray] = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ w + b
            pre_activations.append(z)
            x = _relu(z) if i < len(self.weights) - 1 else z
            activations.append(x)
        return x, activations, pre_activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _, _ = self.forward(x)
        return out

    # ---- single-sample SGD with Huber loss --------------------------------
    def update(
        self, x: np.ndarray, target_idx: int, target_value: float, lr: float = 0.001
    ):
        out, activations, pre_activations = self.forward(x.reshape(1, -1))
        delta = out[0, target_idx] - target_value
        grad = delta if abs(delta) <= 1.0 else math.copysign(1.0, delta)

        d_out = np.zeros((1, self.layer_sizes[-1]))
        d_out[0, target_idx] = grad

        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                d_out = d_out * _relu_grad(pre_activations[i])
            dw = activations[i].T @ d_out
            db = d_out.sum(axis=0)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            if i > 0:
                d_out = d_out @ self.weights[i].T

    # ---- serialisation (JSON-safe) ----------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "SimpleMLP":
        mlp = cls(data["layer_sizes"])
        mlp.weights = [np.array(w) for w in data["weights"]]
        mlp.biases = [np.array(b) for b in data["biases"]]
        return mlp


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------
class ExperienceBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, int, float]] = []
        self.position = 0

    def add(self, features: np.ndarray, action_idx: int, reward: float):
        entry = (features, action_idx, reward)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, rng: np.random.RandomState
    ) -> list[tuple[np.ndarray, int, float]]:
        n = min(batch_size, len(self.buffer))
        indices = rng.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# NN fine-tuner  (same public interface as ResidualFineTuner)
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [64, 32]


class NNResidualFineTuner:
    """Fine-tunes a blueprint policy with a small MLP instead of a Q-table."""

    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()
        self.evaluator = Evaluator(self.config)

    # ---- main entry point -------------------------------------------------
    def train(
        self,
        blueprint_policy: PolicyArtifact,
        episodes: int | None = None,
    ) -> PolicyArtifact:
        total_episodes = episodes or self.config.fine_tune_hands
        print(f"  [NN-FINE-TUNE] Starting NN fine-tuning ({total_episodes} episodes)...")

        rng = random.Random(self.config.seed + 99)
        np_rng = np.random.RandomState(self.config.seed + 99)

        blueprint_runtime = PolicyRuntime(
            blueprint_policy, self.config, self.evaluator.bucketer
        )
        blueprint_agent = PolicyAgent(blueprint_runtime, name="blueprint_mirror")

        layer_sizes = [FEATURE_DIM] + HIDDEN_SIZES + [NUM_ACTIONS]
        net = SimpleMLP(layer_sizes, seed=self.config.seed)
        buffer = ExperienceBuffer(capacity=50_000)
        lr = getattr(self.config, "nn_learning_rate", 0.001)
        batch_size = getattr(self.config, "nn_batch_size", 32)

        baseline_score = self._validate(blueprint_policy)
        print(f"    [EVAL] Baseline score: {baseline_score:.4f}")
        best_score = baseline_score
        best_weights = net.state_dict()

        for episode in range(1, total_episodes + 1):
            opponent = self._sample_opponent(blueprint_agent, rng)
            state = self.evaluator.game.new_hand(
                seed=self.config.seed * 10_007 + episode, button=episode % 2
            )
            visited: list[tuple[np.ndarray, int]] = []

            while not state.terminal:
                observation = self.evaluator.make_observation(state)
                if state.current_player == 0:
                    features = extract_features(observation, self.config)
                    action = self._select_training_action(
                        observation, features, net, blueprint_runtime, rng
                    )
                    visited.append((features, ACTION_INDEX[action]))
                else:
                    action = opponent.select_action(observation, rng)
                state = self.evaluator.game.apply_action(state, action)

            reward = state.payoffs[0]
            for features, action_idx in visited:
                buffer.add(features, action_idx, reward)

            # Batch training from replay buffer
            if len(buffer) >= batch_size and episode % 10 == 0:
                batch = buffer.sample(batch_size, np_rng)
                for feat, act_idx, rew in batch:
                    net.update(feat, act_idx, rew, lr=lr)

            # Periodic validation
            if episode % self.config.fine_tune_eval_interval == 0:
                candidate = PolicyArtifact(
                    algorithm=f"{blueprint_policy.algorithm}+nn_residual",
                    backend=blueprint_policy.backend,
                    policy_table=blueprint_policy.policy_table,
                    metadata={
                        **blueprint_policy.metadata,
                        "fine_tune_mode": "nn",
                        "fine_tune_episodes": episode,
                    },
                    nn_weights=net.state_dict(),
                )
                score = self._validate(candidate)
                print(
                    f"    [EVAL] Episode {episode}/{total_episodes}, "
                    f"Score: {score:.4f}, Best: {best_score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_weights = deepcopy(net.state_dict())

        tuned = PolicyArtifact(
            algorithm=f"{blueprint_policy.algorithm}+nn_residual",
            backend=blueprint_policy.backend,
            policy_table=blueprint_policy.policy_table,
            metadata={
                **blueprint_policy.metadata,
                "fine_tune_mode": "nn",
                "baseline_score": baseline_score,
                "best_score": best_score,
            },
            nn_weights=best_weights,
        )
        return tuned

    # ---- helpers ----------------------------------------------------------
    def _select_training_action(
        self,
        observation: Observation,
        features: np.ndarray,
        net: SimpleMLP,
        blueprint_runtime: PolicyRuntime,
        rng: random.Random,
    ) -> Action:
        legal_actions = list(observation.legal_actions)
        if rng.random() < self.config.fine_tune_epsilon:
            return rng.choice(legal_actions)

        blueprint_dist = blueprint_runtime.distribution(observation)
        q_values = net.predict(features.reshape(1, -1))[0]
        nn_pref = {a: float(q_values[ACTION_INDEX[a]]) for a in legal_actions}
        nn_dist = softmax_preferences(nn_pref, self.config.residual_temperature)

        weight = self.config.residual_mix_weight
        blended = normalize_distribution(
            {
                a: (1.0 - weight) * blueprint_dist.get(a, 0.0)
                + weight * nn_dist.get(a, 0.0)
                for a in legal_actions
            }
        )

        threshold = rng.random()
        cumulative = 0.0
        for action, probability in blended.items():
            cumulative += probability
            if threshold <= cumulative:
                return action
        return legal_actions[-1]

    def _sample_opponent(self, blueprint_agent: PolicyAgent, rng: random.Random):
        draw = rng.random()
        w_mirror = self.config.opponent_mix_mirror
        w_lp = self.config.opponent_mix_lp
        if draw < w_mirror:
            return blueprint_agent
        if draw < w_mirror + w_lp:
            return LoosePassiveAgent()
        return TightAggressiveAgent()

    def _validate(self, candidate: PolicyArtifact) -> float:
        runtime = PolicyRuntime(candidate, self.config, self.evaluator.bucketer)
        agent = PolicyAgent(runtime, name="candidate")
        no_residual = PolicyArtifact(
            algorithm=candidate.algorithm,
            backend=candidate.backend,
            policy_table=candidate.policy_table,
            metadata=candidate.metadata,
        )
        blueprint_agent = PolicyAgent(
            PolicyRuntime(no_residual, self.config, self.evaluator.bucketer),
            name="mirror_baseline",
        )
        mix = [
            (self.config.opponent_mix_mirror, blueprint_agent),
            (self.config.opponent_mix_lp, LoosePassiveAgent()),
            (self.config.opponent_mix_tag, TightAggressiveAgent()),
        ]
        score = 0.0
        base_seed = self.config.seed + 50_000
        for index, (weight, opponent) in enumerate(mix):
            result = self.evaluator.match(
                agent,
                opponent,
                hands=self.config.fine_tune_validation_hands,
                seed=base_seed + index,
            )
            score += weight * result.sb_per_100
        return score
