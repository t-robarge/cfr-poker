from __future__ import annotations

import math
import random

from .bucketing import Bucketer
from .config import AbstractHULHEConfig
from .models import Action, InfoSetKey, Observation, PolicyArtifact


def normalize_distribution(raw: dict[Action, float]) -> dict[Action, float]:
    positive = {action: max(0.0, value) for action, value in raw.items()}
    total = sum(positive.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(positive))
        return {action: uniform for action in positive}
    return {action: value / total for action, value in positive.items()}


def softmax_preferences(preferences: dict[Action, float], temperature: float) -> dict[Action, float]:
    safe_temp = max(temperature, 1e-6)
    max_pref = max(preferences.values(), default=0.0)
    exps = {
        action: math.exp((value - max_pref) / safe_temp)
        for action, value in preferences.items()
    }
    return normalize_distribution(exps)


class PolicyRuntime:
    def __init__(
        self,
        artifact: PolicyArtifact,
        config: AbstractHULHEConfig | None = None,
        bucketer: Bucketer | None = None,
    ):
        self.artifact = artifact
        self.config = config or AbstractHULHEConfig()
        self.bucketer = bucketer or Bucketer(self.config)
        self._nn_model = None          # lazy-loaded SimpleMLP cache
        self._subgame_solver = None    # lazy-loaded RiverSubgameSolver

    def infoset_key(self, observation: Observation) -> InfoSetKey:
        return InfoSetKey(
            street=observation.street.value,
            position=observation.acting_player,
            history_id=observation.history_id,
            bucket_id=observation.bucket_id,
        )

    def distribution(self, observation: Observation) -> dict[Action, float]:
        key = self.infoset_key(observation).encode()
        legal = list(observation.legal_actions)
        base_raw = self.artifact.policy_table.get(key, {})
        base = normalize_distribution(
            {action: float(base_raw.get(action.value, 0.0)) for action in legal}
            or {action: 1.0 for action in legal}
        )

        # --- residual blending (NN or tabular) ---
        if self.artifact.nn_weights:
            dist = self._blend_nn_residual(observation, base, legal)
        else:
            residual_raw = self.artifact.residual_table.get(key)
            if residual_raw:
                residual = softmax_preferences(
                    {action: float(residual_raw.get(action.value, 0.0)) for action in legal},
                    temperature=self.config.residual_temperature,
                )
                weight = self.config.residual_mix_weight
                dist = normalize_distribution(
                    {
                        action: (1.0 - weight) * base.get(action, 0.0)
                        + weight * residual.get(action, 0.0)
                        for action in legal
                    }
                )
            else:
                dist = base

        # --- river subgame refinement (opt-in) ---
        if getattr(self.config, "use_subgame_solving", False):
            dist = self._apply_subgame(observation, dist)

        return dist

    # -- NN residual helper -------------------------------------------------
    def _blend_nn_residual(
        self,
        observation: Observation,
        base: dict[Action, float],
        legal: list[Action],
    ) -> dict[Action, float]:
        from .nn_rl import SimpleMLP, extract_features, ACTION_INDEX, FEATURE_DIM, HIDDEN_SIZES, NUM_ACTIONS
        import numpy as np

        if self._nn_model is None:
            self._nn_model = SimpleMLP.from_state_dict(self.artifact.nn_weights)
        features = extract_features(observation, self.config)
        q_values = self._nn_model.predict(features.reshape(1, -1))[0]
        nn_pref = {a: float(q_values[ACTION_INDEX[a]]) for a in legal}
        nn_dist = softmax_preferences(nn_pref, self.config.residual_temperature)
        weight = self.config.residual_mix_weight
        return normalize_distribution(
            {
                a: (1.0 - weight) * base.get(a, 0.0) + weight * nn_dist.get(a, 0.0)
                for a in legal
            }
        )

    # -- subgame refinement helper ------------------------------------------
    def _apply_subgame(
        self, observation: Observation, dist: dict[Action, float]
    ) -> dict[Action, float]:
        from .subgame import RiverSubgameSolver
        from .models import Street

        if observation.street is not Street.RIVER:
            return dist
        if self._subgame_solver is None:
            self._subgame_solver = RiverSubgameSolver(self.config)
        return self._subgame_solver.refine(observation, dist)

    def act(self, observation: Observation, rng: random.Random | None = None) -> Action:
        rng = rng or random.Random(self.config.seed)
        distribution = self.distribution(observation)
        threshold = rng.random()
        cumulative = 0.0
        for action, probability in distribution.items():
            cumulative += probability
            if threshold <= cumulative:
                return action
        return next(iter(distribution))

