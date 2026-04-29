from __future__ import annotations

from collections import defaultdict
import itertools
from typing import Callable

from .abstract_game import AbstractGameSpec
from .bucketing import Bucketer
from .cards import DECK
from .config import AbstractHULHEConfig
from .models import Action, Observation, Street

Combo = tuple[str, str]
PolicyLookup = Callable[[Observation], dict[Action, float]]

_STREET_CODE = {
    "p": Street.PREFLOP,
    "f": Street.FLOP,
    "t": Street.TURN,
    "r": Street.RIVER,
}
_ACTION_CODE = {
    "f": Action.FOLD,
    "k": Action.CHECK,
    "c": Action.CALL,
    "r": Action.RAISE,
}


def _normalize(values: dict[object, float]) -> dict[object, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0.0:
        return {}
    return {key: max(0.0, value) / total for key, value in values.items()}


def _street_board(board: tuple[str, ...], street: Street) -> tuple[str, ...]:
    if street is Street.PREFLOP:
        return ()
    if street is Street.FLOP:
        return board[:3]
    if street is Street.TURN:
        return board[:4]
    return board[:5]


def _parse_history(history_id: str) -> dict[Street, str]:
    sequences = {street: "" for street in Street}
    if history_id == "root":
        return sequences
    for section in history_id.split("/"):
        code, sequence = section.split(":", 1)
        sequences[_STREET_CODE[code]] = sequence
    return sequences


def enumerate_legal_hole_combos(
    board: tuple[str, ...],
    dead_cards: tuple[str, ...] = (),
) -> tuple[Combo, ...]:
    blocked = set(board) | set(dead_cards)
    remaining = [card for card in DECK if card not in blocked]
    return tuple(tuple(sorted(combo)) for combo in itertools.combinations(remaining, 2))


class ExactRiverRangeTranslator:
    """Translate bucketed blueprint beliefs into exact river combo weights.

    The abstraction only knows bucket-level policies, so this translator builds
    approximate exact-card posteriors by:
    - splitting each player's preflop bucket mass uniformly over legal combos,
    - replaying the public action line,
    - reweighting combos whenever that player acted, using the deployed
      non-subgame policy for the combo's current bucket.

    For exact river resolving we use both players' translated marginal ranges,
    then apply blocker consistency inside the solver via combo compatibility.
    For the legacy bucketed river resolve we still expose a villain posterior
    filtered by the hero's exact blockers.
    """

    def __init__(
        self,
        config: AbstractHULHEConfig,
        spec: AbstractGameSpec,
        policy_lookup: PolicyLookup,
    ):
        self.config = config
        self.spec = spec
        self.policy_lookup = policy_lookup
        self.bucketer = Bucketer(config)
        self._action_prob_cache: dict[tuple[str, str, str, str], float] = {}
        self._range_cache: dict[tuple[str, tuple[str, ...], int], dict[Combo, float]] = {}
        self._root_bucket_cache: dict[tuple[str, Combo, tuple[str, ...], int], dict[str, float]] = {}

    def translate_range(self, observation: Observation, target_role: int) -> dict[Combo, float]:
        cache_key = (observation.history_id, observation.board, target_role)
        cached = self._range_cache.get(cache_key)
        if cached is not None:
            return cached

        weights = self._initial_range_weights(observation.board, target_role)
        if not weights:
            return {}

        current_node_id = self.spec.root_node
        line = _parse_history(observation.history_id)
        for street in Street:
            if street.value == observation.street.value:
                street_line = line[street]
            elif list(Street).index(street) < list(Street).index(observation.street):
                street_line = line[street]
            else:
                break

            board_prefix = _street_board(observation.board, street)
            for action_code in street_line:
                node = self.spec.nodes[current_node_id]
                observed_action = _ACTION_CODE[action_code]
                actor_role = int(node["current_player"])
                if actor_role == target_role:
                    updated = {
                        combo: weight
                        * self._action_probability(
                            node=node,
                            board=board_prefix,
                            combo=combo,
                            action=observed_action,
                        )
                        for combo, weight in weights.items()
                    }
                    weights = self._normalize_or_fallback(updated, weights)
                current_node_id = node["transitions"][observed_action.value]["next_node"]

        self._range_cache[cache_key] = weights
        return weights

    def translate_root_ranges(self, observation: Observation) -> tuple[dict[Combo, float], dict[Combo, float]]:
        return (
            self.translate_range(observation, 0),
            self.translate_range(observation, 1),
        )

    def translate_villain_weights(self, observation: Observation) -> dict[Combo, float]:
        villain_role = 1 - observation.relative_position
        original = self.translate_range(observation, villain_role)
        hero_cards = set(observation.hole_cards)
        filtered = {
            combo: weight
            for combo, weight in original.items()
            if hero_cards.isdisjoint(combo)
        }
        return self._normalize_or_fallback(filtered, original)

    def root_bucket_distribution(self, observation: Observation) -> dict[str, float]:
        hero_combo = tuple(sorted(observation.hole_cards))
        cache_key = (
            observation.history_id,
            hero_combo,
            observation.board,
            observation.relative_position,
        )
        cached = self._root_bucket_cache.get(cache_key)
        if cached is not None:
            return cached

        villain_weights = self.translate_villain_weights(observation)
        if not villain_weights:
            self._root_bucket_cache[cache_key] = {}
            return {}

        hero_bucket = observation.bucket_id
        distribution: dict[str, float] = {}
        for combo, weight in villain_weights.items():
            villain_bucket = self.bucketer.bucket(observation.street, combo, observation.board)
            if observation.relative_position == 0:
                pair = f"{hero_bucket}-{villain_bucket}"
            else:
                pair = f"{villain_bucket}-{hero_bucket}"
            distribution[pair] = distribution.get(pair, 0.0) + weight
        normalized = {key: value for key, value in _normalize(distribution).items()}
        self._root_bucket_cache[cache_key] = normalized
        return normalized

    def _initial_range_weights(
        self,
        board: tuple[str, ...],
        target_role: int,
    ) -> dict[Combo, float]:
        bucket_mass = defaultdict(float)
        for pair, probability in self.spec.initial_bucket_distribution.items():
            left_raw, right_raw = pair.split("-")
            left = int(left_raw)
            right = int(right_raw)
            bucket_id = left if target_role == 0 else right
            bucket_mass[bucket_id] += probability

        legal_combos = enumerate_legal_hole_combos(board)
        by_bucket: dict[int, list[Combo]] = defaultdict(list)
        for combo in legal_combos:
            by_bucket[self.bucketer.bucket(Street.PREFLOP, combo, ())].append(combo)

        weights: dict[Combo, float] = {}
        for bucket_id, combos in by_bucket.items():
            mass = bucket_mass.get(bucket_id, 0.0)
            if mass <= 0.0:
                continue
            share = mass / len(combos)
            for combo in combos:
                weights[combo] = share

        if weights:
            normalized = _normalize(weights)
            return {key: float(value) for key, value in normalized.items()}

        if not legal_combos:
            return {}
        uniform = 1.0 / len(legal_combos)
        return {combo: uniform for combo in legal_combos}

    def _action_probability(
        self,
        node: dict[str, object],
        board: tuple[str, ...],
        combo: Combo,
        action: Action,
    ) -> float:
        cache_key = (str(node["history_id"]), ",".join(combo), ",".join(board), action.value)
        cached = self._action_prob_cache.get(cache_key)
        if cached is not None:
            return cached

        street = Street(str(node["street"]))
        legal_actions = tuple(Action(name) for name in node["legal_actions"])
        details = self.bucketer.bucket_details(street, combo, board)
        observation = Observation(
            acting_player=int(node["current_player"]),
            # The abstract game is built with button-role == player 0, so we
            # keep button=0 here to query the policy in role space.
            button=0,
            street=street,
            hole_cards=combo,
            board=board,
            history_id=str(node["history_id"]),
            legal_actions=legal_actions,
            to_call=int(node["current_to_call"]) - int(node["street_contrib"][int(node["current_player"])]),
            pot=int(node["pot"]),
            bucket_id=details.bucket_id,
            bucket_percentile=details.percentile,
        )
        distribution = self.policy_lookup(observation)
        likelihood = float(distribution.get(action, 0.0))
        self._action_prob_cache[cache_key] = likelihood
        return likelihood

    @staticmethod
    def _normalize_or_fallback(
        updated: dict[Combo, float],
        fallback: dict[Combo, float],
    ) -> dict[Combo, float]:
        normalized = _normalize(updated)
        if normalized:
            return {key: float(value) for key, value in normalized.items()}
        return dict(fallback)
