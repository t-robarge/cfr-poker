from __future__ import annotations

from copy import deepcopy
import random

from .baselines import LoosePassiveAgent, PolicyAgent, TightAggressiveAgent
from .config import AbstractHULHEConfig
from .evaluator import Evaluator
from .models import Action, InfoSetKey, PolicyArtifact
from .policy import PolicyRuntime, normalize_distribution, softmax_preferences


class ResidualFineTuner:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()
        self.evaluator = Evaluator(self.config)

    def train(
        self,
        blueprint_policy: PolicyArtifact,
        episodes: int | None = None,
    ) -> PolicyArtifact:
        total_episodes = episodes or self.config.fine_tune_hands
        print(f"  [FINE-TUNE] Starting RL fine-tuning ({total_episodes} episodes)...")
        rng = random.Random(self.config.seed + 99)
        blueprint_runtime = PolicyRuntime(blueprint_policy, self.config, self.evaluator.bucketer)
        blueprint_agent = PolicyAgent(blueprint_runtime, name="blueprint_mirror")
        q_values: dict[str, dict[str, float]] = {}

        baseline_score = self._validate(blueprint_policy)
        print(f"    [EVAL] Baseline score: {baseline_score:.4f}")
        best_score = baseline_score
        best_residual: dict[str, dict[str, float]] = {}

        for episode in range(1, total_episodes + 1):
            opponent = self._sample_opponent(blueprint_agent, rng)
            state = self.evaluator.game.new_hand(seed=self.config.seed * 10_007 + episode, button=episode % 2)
            visited: list[tuple[str, str]] = []

            while not state.terminal:
                observation = self.evaluator.make_observation(state)
                if state.current_player == 0:
                    info_key = InfoSetKey(
                        street=observation.street.value,
                        position=observation.acting_player,
                        history_id=observation.history_id,
                        bucket_id=observation.bucket_id,
                    ).encode()
                    action = self._select_training_action(
                        observation,
                        info_key,
                        blueprint_runtime,
                        q_values,
                        rng,
                    )
                    visited.append((info_key, action.value))
                else:
                    action = opponent.select_action(observation, rng)
                state = self.evaluator.game.apply_action(state, action)

            reward = state.payoffs[0]
            for info_key, action_name in visited:
                action_values = q_values.setdefault(info_key, {})
                prior = action_values.get(action_name, 0.0)
                action_values[action_name] = prior + self.config.fine_tune_alpha * (reward - prior)

            if episode % self.config.fine_tune_eval_interval == 0:
                candidate = PolicyArtifact(
                    algorithm=f"{blueprint_policy.algorithm}+residual_rl",
                    backend=blueprint_policy.backend,
                    policy_table=blueprint_policy.policy_table,
                    metadata={
                        **blueprint_policy.metadata,
                        "fine_tune_episodes": episode,
                    },
                    residual_table=deepcopy(q_values),
                )
                score = self._validate(candidate)
                print(f"    [EVAL] Episode {episode}/{total_episodes}, Score: {score:.4f}, Best: {best_score:.4f}")
                if score > best_score:
                    best_score = score
                    best_residual = deepcopy(q_values)

        if best_residual:
            tuned = PolicyArtifact(
                algorithm=f"{blueprint_policy.algorithm}+residual_rl",
                backend=blueprint_policy.backend,
                policy_table=blueprint_policy.policy_table,
                metadata={
                    **blueprint_policy.metadata,
                    "baseline_score": baseline_score,
                    "best_score": best_score,
                },
                residual_table=best_residual,
            )
            return tuned

        unchanged = deepcopy(blueprint_policy)
        unchanged.metadata = {
            **blueprint_policy.metadata,
            "baseline_score": baseline_score,
            "best_score": baseline_score,
            "fine_tune_skipped": "validation did not improve over the blueprint policy",
        }
        return unchanged

    def _select_training_action(
        self,
        observation,
        info_key: str,
        blueprint_runtime: PolicyRuntime,
        q_values: dict[str, dict[str, float]],
        rng: random.Random,
    ) -> Action:
        legal_actions = list(observation.legal_actions)
        if rng.random() < self.config.fine_tune_epsilon:
            return rng.choice(legal_actions)

        blueprint_dist = blueprint_runtime.distribution(observation)
        residual_pref = {
            action: q_values.get(info_key, {}).get(action.value, 0.0)
            for action in legal_actions
        }
        residual_dist = softmax_preferences(residual_pref, self.config.residual_temperature)
        blended = normalize_distribution(
            {
                action: (1.0 - self.config.residual_mix_weight) * blueprint_dist.get(action, 0.0)
                + self.config.residual_mix_weight * residual_dist.get(action, 0.0)
                for action in legal_actions
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
        if draw < 0.50:
            return blueprint_agent
        if draw < 0.75:
            return LoosePassiveAgent()
        return TightAggressiveAgent()

    def _validate(self, candidate: PolicyArtifact) -> float:
        runtime = PolicyRuntime(candidate, self.config, self.evaluator.bucketer)
        agent = PolicyAgent(runtime, name="candidate")
        blueprint_agent = PolicyAgent(
            PolicyRuntime(candidate if not candidate.residual_table else PolicyArtifact(
                algorithm=candidate.algorithm,
                backend=candidate.backend,
                policy_table=candidate.policy_table,
                metadata=candidate.metadata,
            ), self.config, self.evaluator.bucketer),
            name="mirror_baseline",
        )
        mix = [
            (0.50, blueprint_agent),
            (0.25, LoosePassiveAgent()),
            (0.25, TightAggressiveAgent()),
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
