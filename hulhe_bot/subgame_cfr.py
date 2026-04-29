from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .abstract_game import AbstractGameSpec
from .config import AbstractHULHEConfig
from .exact_river_cfr import ExactRiverCFRSolver
from .exact_posterior import ExactRiverRangeTranslator
from .models import Action, InfoSetKey, Observation, Street
from .policy import normalize_distribution
from .trainer import LocalExternalSamplingTrainer


class CFRSubgameSolver:
    """Traditional local CFR resolver over the existing abstract game.

    This resolver reuses the project's bucketed abstract game rather than
    inventing a separate late-street framework. At a turn or river decision it:

    1. loads the pre-built abstract game,
    2. swaps the root node to the current public state,
    3. either:
       - conditions a bucket-pair root belief and runs the legacy abstract
         resolve, or
       - on the river, translates both players into exact combo ranges and runs
         an exact-card resolve,
    4. blends the resolved root strategy with the incoming blueprint/residual
       distribution.

    This is intentionally conservative: it stays close to the current codebase
    and uses the same abstraction, transition tables, and external-sampling CFR
    implementation already used for the blueprint.
    """

    def __init__(
        self,
        config: AbstractHULHEConfig,
        base_policy_lookup: Callable[[Observation], dict[Action, float]] | None = None,
    ):
        self.config = config
        self.base_policy_lookup = base_policy_lookup
        self._spec: AbstractGameSpec | None = None
        self._street_distribution_cache: dict[str, dict[str, float]] = {}
        self._resolved_root_cache: dict[tuple[object, ...], dict[Action, float]] = {}
        self._exact_river_translator: ExactRiverRangeTranslator | None = None
        self._exact_river_solver: ExactRiverCFRSolver | None = None

    def refine(
        self,
        observation: Observation,
        blueprint_dist: dict[Action, float],
    ) -> dict[Action, float]:
        if not self._should_resolve(observation.street):
            return blueprint_dist

        if observation.history_id not in self._load_spec().nodes:
            return blueprint_dist

        cache_key = self._cache_key(observation)
        resolved_dist = self._resolved_root_cache.get(cache_key)
        if resolved_dist is None:
            resolved_dist = self._resolve_root_distribution(observation)
            if not resolved_dist:
                return blueprint_dist
            self._resolved_root_cache[cache_key] = resolved_dist

        blend_weight = self._blend_weight(observation.street)
        blended = {
            action: (1.0 - blend_weight) * blueprint_dist.get(action, 0.0)
            + blend_weight * resolved_dist.get(action, 0.0)
            for action in observation.legal_actions
        }
        return normalize_distribution(blended)

    @staticmethod
    def _cache_key(observation: Observation) -> tuple[object, ...]:
        if observation.street is Street.RIVER:
            return (
                observation.history_id,
                observation.relative_position,
                tuple(sorted(observation.hole_cards)),
                observation.board,
            )
        return (
            observation.history_id,
            observation.relative_position,
            observation.bucket_id,
        )

    def _resolve_root_distribution(self, observation: Observation) -> dict[Action, float]:
        exact = self._exact_river_distribution(observation)
        if exact:
            return exact

        root_distribution = self._root_distribution(observation)
        if not root_distribution:
            return {}

        local_spec = replace(
            self._load_spec(),
            root_node=observation.history_id,
            initial_bucket_distribution=root_distribution,
        )
        trainer = LocalExternalSamplingTrainer(
            local_spec,
            self.config,
            algorithm=getattr(self.config, "subgame_cfr_algorithm", "cfr_plus"),
        )
        artifact = trainer.train(
            iterations=getattr(self.config, "subgame_cfr_iterations", 200),
            checkpoint_every=0,
        )
        return self._extract_root_distribution(observation, artifact.policy_table)

    def _load_spec(self) -> AbstractGameSpec:
        if self._spec is None:
            self._spec = AbstractGameSpec.load(self.config.abstraction_file)
        return self._spec

    def _should_resolve(self, street: Street) -> bool:
        if street is Street.TURN:
            return getattr(self.config, "subgame_resolve_turn", True)
        if street is Street.RIVER:
            return getattr(self.config, "subgame_resolve_river", True)
        return False

    def _blend_weight(self, street: Street) -> float:
        if street is Street.TURN:
            return getattr(
                self.config,
                "turn_subgame_blend_weight",
                getattr(self.config, "subgame_blend_weight", 0.5),
            )
        return getattr(self.config, "subgame_blend_weight", 0.5)

    def _extract_root_distribution(
        self,
        observation: Observation,
        policy_table: dict[str, dict[str, float]],
    ) -> dict[Action, float]:
        info_key = InfoSetKey(
            street=observation.street.value,
            position=observation.relative_position,
            history_id=observation.history_id,
            bucket_id=observation.bucket_id,
        ).encode()
        raw = policy_table.get(info_key)
        if not raw:
            return {}
        return normalize_distribution(
            {
                action: float(raw.get(action.value, 0.0))
                for action in observation.legal_actions
            }
        )

    def _condition_root_distribution(self, observation: Observation) -> dict[str, float]:
        street_distribution = self._street_distribution(observation.street)
        conditioned = {
            pair: probability
            for pair, probability in street_distribution.items()
            if self._hero_bucket(pair, observation.relative_position) == observation.bucket_id
        }
        if conditioned:
            total = sum(conditioned.values())
            return {pair: value / total for pair, value in conditioned.items()}

        # Fallback: if the exact bucket is unseen in the sampled street tables,
        # preserve execution by returning a small uniform mass over compatible
        # pairs discovered anywhere in that street's support.
        compatible = [
            pair for pair in street_distribution
            if self._hero_bucket(pair, observation.relative_position) == observation.bucket_id
        ]
        if compatible:
            uniform = 1.0 / len(compatible)
            return {pair: uniform for pair in compatible}
        return {}

    def _root_distribution(self, observation: Observation) -> dict[str, float]:
        if (
            observation.street is Street.RIVER
            and getattr(self.config, "subgame_exact_posterior_river", False)
            and self.base_policy_lookup is not None
        ):
            if self._exact_river_translator is None:
                self._exact_river_translator = ExactRiverRangeTranslator(
                    self.config,
                    self._load_spec(),
                    self.base_policy_lookup,
                )
            translated = self._exact_river_translator.root_bucket_distribution(observation)
            if translated:
                return translated
        return self._condition_root_distribution(observation)

    def _exact_river_distribution(self, observation: Observation) -> dict[Action, float]:
        if (
            observation.street is not Street.RIVER
            or not getattr(self.config, "subgame_exact_resolve_river", False)
            or self.base_policy_lookup is None
        ):
            return {}
        if self._exact_river_solver is None:
            self._exact_river_solver = ExactRiverCFRSolver(
                self.config,
                self._load_spec(),
                self.base_policy_lookup,
            )
        return self._exact_river_solver.refine(observation)

    def _street_distribution(self, street: Street) -> dict[str, float]:
        cached = self._street_distribution_cache.get(street.value)
        if cached is not None:
            return cached

        spec = self._load_spec()
        distribution = dict(spec.initial_bucket_distribution)
        if street is Street.PREFLOP:
            self._street_distribution_cache[street.value] = distribution
            return distribution

        for chance_stage in (Street.PREFLOP.value, Street.FLOP.value, Street.TURN.value):
            distribution = self._advance_distribution(distribution, chance_stage)
            next_street = {
                Street.PREFLOP.value: Street.FLOP,
                Street.FLOP.value: Street.TURN,
                Street.TURN.value: Street.RIVER,
            }[chance_stage]
            if next_street is street:
                self._street_distribution_cache[street.value] = distribution
                return distribution

        self._street_distribution_cache[street.value] = distribution
        return distribution

    def _advance_distribution(
        self,
        current: dict[str, float],
        chance_stage: str,
    ) -> dict[str, float]:
        spec = self._load_spec()
        transitions = spec.street_transitions.get(chance_stage, {})
        next_distribution: dict[str, float] = {}
        for source, source_prob in current.items():
            destinations = transitions.get(source)
            if not destinations:
                continue
            for destination, probability in destinations.items():
                next_distribution[destination] = next_distribution.get(destination, 0.0) + source_prob * probability
        total = sum(next_distribution.values())
        if total <= 0:
            return {}
        return {pair: value / total for pair, value in next_distribution.items()}

    @staticmethod
    def _hero_bucket(pair: str, relative_position: int) -> int:
        left, right = pair.split("-")
        return int(left if relative_position == 0 else right)
