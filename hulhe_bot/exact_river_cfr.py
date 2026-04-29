from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .abstract_game import AbstractGameSpec
from .cards import evaluate_7card
from .config import AbstractHULHEConfig
from .exact_posterior import ExactRiverRangeTranslator, enumerate_legal_hole_combos
from .models import Action, Observation, Street
from .native_river import NativeExactRiverSolver
from .policy import normalize_distribution

Combo = tuple[str, str]


@dataclass(slots=True)
class _ResolvedRiverRoot:
    actions: tuple[Action, ...]
    average_strategy: np.ndarray
    current_strategy: np.ndarray
    combo_index: dict[Combo, int]


class ExactRiverCFRSolver:
    """Exact-card river resolver over the existing public betting subtree.

    This keeps the public tree from the abstract game, but replaces river
    bucket-pair states with exact combo ranges for both players. The ranges are
    translated from the deployed bucketed policy, then solved with a small
    full-width CFR/CFR+/DCFR pass and queried at the acting player's actual
    combo.
    """

    def __init__(
        self,
        config: AbstractHULHEConfig,
        spec: AbstractGameSpec,
        policy_lookup,
    ):
        self.config = config
        self.spec = spec
        self.range_translator = ExactRiverRangeTranslator(config, spec, policy_lookup)
        self.native_solver = NativeExactRiverSolver(config, spec)
        self._solve_cache: dict[tuple[str, tuple[str, ...]], _ResolvedRiverRoot] = {}

    def refine(self, observation: Observation) -> dict[Action, float]:
        if observation.street is not Street.RIVER:
            return {}

        cache_key = (observation.history_id, observation.board)
        solved = self._solve_cache.get(cache_key)
        if solved is None:
            solved = self._solve_root(observation)
            if solved is None:
                return {}
            self._solve_cache[cache_key] = solved

        combo = tuple(sorted(observation.hole_cards))
        combo_index = solved.combo_index.get(combo)
        if combo_index is None:
            return {}

        row = solved.average_strategy[combo_index]
        if float(np.sum(row)) <= 0.0:
            row = solved.current_strategy[combo_index]
        if float(np.sum(row)) <= 0.0:
            return {}

        return normalize_distribution(
            {
                action: float(row[index])
                for index, action in enumerate(solved.actions)
                if action in observation.legal_actions
            }
        )

    def _solve_root(self, observation: Observation) -> _ResolvedRiverRoot | None:
        root_node = self.spec.nodes.get(observation.history_id)
        if root_node is None or root_node["terminal_type"] is not None:
            return None
        if int(root_node["current_player"]) != observation.relative_position:
            return None

        native = self._solve_root_native(observation)
        if native is not None:
            return native

        combos = enumerate_legal_hole_combos(observation.board)
        if not combos:
            return None
        combo_index = {combo: index for index, combo in enumerate(combos)}
        compatibility = self._compatibility_matrix(combos)
        hand_scores = tuple(evaluate_7card(tuple(combo + observation.board)) for combo in combos)
        ranges = self.range_translator.translate_root_ranges(observation)
        range_arrays = [
            self._range_array(ranges[0], combo_index),
            self._range_array(ranges[1], combo_index),
        ]
        if float(np.sum(range_arrays[0])) <= 0.0 or float(np.sum(range_arrays[1])) <= 0.0:
            return None

        algorithm = getattr(self.config, "subgame_cfr_algorithm", "cfr_plus")
        iterations = max(1, int(getattr(self.config, "subgame_cfr_iterations", 200)))
        weight_scale = 1.0 if algorithm == "dcfr" else 0.0
        regrets: dict[str, np.ndarray] = {}
        strategy_sums: dict[str, np.ndarray] = {}
        terminal_cache: dict[str, np.ndarray | float] = {}

        for iteration in range(1, iterations + 1):
            if algorithm == "dcfr":
                self._apply_dcfr_discount(regrets, strategy_sums, iteration)
            average_weight = float(iteration if weight_scale == 0.0 else 1.0)
            for traverser in (0, 1):
                self._cfr(
                    node_id=observation.history_id,
                    traverser=traverser,
                    reach0=range_arrays[0],
                    reach1=range_arrays[1],
                    compatibility=compatibility,
                    hand_scores=hand_scores,
                    regrets=regrets,
                    strategy_sums=strategy_sums,
                    terminal_cache=terminal_cache,
                    algorithm=algorithm,
                    average_weight=average_weight,
                )

        actions = tuple(Action(name) for name in root_node["legal_actions"])
        regrets_root = regrets.get(observation.history_id)
        strategy_sum_root = strategy_sums.get(observation.history_id)
        if regrets_root is None or strategy_sum_root is None:
            return None
        current_strategy = self._current_strategy(regrets_root)
        average_strategy = self._average_strategy(strategy_sum_root, current_strategy)
        return _ResolvedRiverRoot(
            actions=actions,
            average_strategy=average_strategy,
            current_strategy=current_strategy,
            combo_index=combo_index,
        )

    def _solve_root_native(self, observation: Observation) -> _ResolvedRiverRoot | None:
        if not getattr(self.config, "subgame_native_exact_river", True):
            return None
        solved = self.native_solver.solve(observation, self.range_translator)
        if solved is None:
            return None
        return _ResolvedRiverRoot(
            actions=tuple(solved["actions"]),
            average_strategy=np.asarray(solved["average_strategy"], dtype=np.float64),
            current_strategy=np.asarray(solved["current_strategy"], dtype=np.float64),
            combo_index=dict(solved["combo_index"]),
        )

    def _cfr(
        self,
        node_id: str,
        traverser: int,
        reach0: np.ndarray,
        reach1: np.ndarray,
        compatibility: np.ndarray,
        hand_scores: tuple[tuple[int, ...], ...],
        regrets: dict[str, np.ndarray],
        strategy_sums: dict[str, np.ndarray],
        terminal_cache: dict[str, np.ndarray | float],
        algorithm: str,
        average_weight: float,
    ) -> np.ndarray:
        node = self.spec.nodes[node_id]
        terminal_type = node["terminal_type"]
        if terminal_type is not None:
            return self._terminal_values(
                node=node,
                traverser=traverser,
                reach0=reach0,
                reach1=reach1,
                compatibility=compatibility,
                hand_scores=hand_scores,
                terminal_cache=terminal_cache,
            )

        actor = int(node["current_player"])
        actions = tuple(Action(name) for name in node["legal_actions"])
        node_regrets = regrets.get(node_id)
        if node_regrets is None:
            node_regrets = np.zeros((reach0.shape[0], len(actions)), dtype=np.float64)
            regrets[node_id] = node_regrets
        node_strategy_sums = strategy_sums.get(node_id)
        if node_strategy_sums is None:
            node_strategy_sums = np.zeros_like(node_regrets)
            strategy_sums[node_id] = node_strategy_sums

        strategy = self._current_strategy(node_regrets)
        actor_reach = reach0 if actor == 0 else reach1
        node_strategy_sums += average_weight * actor_reach[:, np.newaxis] * strategy

        child_values = []
        for action_index, action in enumerate(actions):
            transition = node["transitions"][action.value]
            if actor == 0:
                next_reach0 = reach0 * strategy[:, action_index]
                next_reach1 = reach1
            else:
                next_reach0 = reach0
                next_reach1 = reach1 * strategy[:, action_index]
            child_values.append(
                self._cfr(
                    node_id=transition["next_node"],
                    traverser=traverser,
                    reach0=next_reach0,
                    reach1=next_reach1,
                    compatibility=compatibility,
                    hand_scores=hand_scores,
                    regrets=regrets,
                    strategy_sums=strategy_sums,
                    terminal_cache=terminal_cache,
                    algorithm=algorithm,
                    average_weight=average_weight,
                )
            )

        child_matrix = np.stack(child_values, axis=1)
        if actor != traverser:
            return np.sum(child_matrix, axis=1)

        node_values = np.sum(strategy * child_matrix, axis=1)
        regret_delta = child_matrix - node_values[:, np.newaxis]
        node_regrets += regret_delta
        if algorithm == "cfr_plus":
            np.maximum(node_regrets, 0.0, out=node_regrets)
        return node_values

    def _terminal_values(
        self,
        node: dict[str, object],
        traverser: int,
        reach0: np.ndarray,
        reach1: np.ndarray,
        compatibility: np.ndarray,
        hand_scores: tuple[tuple[int, ...], ...],
        terminal_cache: dict[str, np.ndarray | float],
    ) -> np.ndarray:
        node_id = str(node["node_id"])
        c0, c1 = node["total_contrib"]
        pot = float(node["pot"])
        reward_unit = float(self.config.reward_unit)

        if node["terminal_type"] == "showdown":
            payoff_matrix = terminal_cache.get(node_id)
            if payoff_matrix is None:
                base = -float(c0) / reward_unit
                scale = pot / reward_unit
                payoff_matrix = np.zeros_like(compatibility)
                for left_index, left_score in enumerate(hand_scores):
                    for right_index, right_score in enumerate(hand_scores):
                        if compatibility[left_index, right_index] <= 0.0:
                            continue
                        if left_score > right_score:
                            share = 1.0
                        elif left_score == right_score:
                            share = 0.5
                        else:
                            share = 0.0
                        payoff_matrix[left_index, right_index] = base + scale * share
                terminal_cache[node_id] = payoff_matrix
            if traverser == 0:
                return payoff_matrix @ reach1
            return (-payoff_matrix.T) @ reach0

        payoff_p0 = terminal_cache.get(node_id)
        if payoff_p0 is None:
            folded_player = int(node["folded_player"])
            p0_share = 0.0 if folded_player == 0 else 1.0
            payoff_p0 = (pot * p0_share - float(c0)) / reward_unit
            terminal_cache[node_id] = payoff_p0
        if traverser == 0:
            return float(payoff_p0) * (compatibility @ reach1)
        return float(-payoff_p0) * (compatibility.T @ reach0)

    @staticmethod
    def _range_array(weights: dict[Combo, float], combo_index: dict[Combo, int]) -> np.ndarray:
        array = np.zeros(len(combo_index), dtype=np.float64)
        for combo, weight in weights.items():
            index = combo_index.get(combo)
            if index is not None:
                array[index] = float(weight)
        return array

    @staticmethod
    def _compatibility_matrix(combos: tuple[Combo, ...]) -> np.ndarray:
        card_order = sorted({card for combo in combos for card in combo})
        card_index = {card: index for index, card in enumerate(card_order)}
        masks = [
            (1 << card_index[combo[0]]) | (1 << card_index[combo[1]])
            for combo in combos
        ]
        size = len(combos)
        compatibility = np.zeros((size, size), dtype=np.float64)
        for left_index, left_mask in enumerate(masks):
            compatibility[left_index, left_index] = 0.0
            for right_index in range(left_index + 1, size):
                if left_mask & masks[right_index]:
                    continue
                compatibility[left_index, right_index] = 1.0
                compatibility[right_index, left_index] = 1.0
        return compatibility

    @staticmethod
    def _current_strategy(regrets: np.ndarray) -> np.ndarray:
        if regrets.size == 0:
            return regrets
        positive = np.maximum(regrets, 0.0)
        row_sums = np.sum(positive, axis=1, keepdims=True)
        uniform = np.full_like(positive, 1.0 / positive.shape[1])
        return np.divide(positive, row_sums, out=uniform, where=row_sums > 0.0)

    @staticmethod
    def _average_strategy(strategy_sums: np.ndarray, current_strategy: np.ndarray) -> np.ndarray:
        if strategy_sums.size == 0:
            return strategy_sums
        row_sums = np.sum(strategy_sums, axis=1, keepdims=True)
        averaged = np.zeros_like(strategy_sums)
        np.divide(strategy_sums, row_sums, out=averaged, where=row_sums > 0.0)
        fallback_rows = (row_sums[:, 0] <= 0.0)
        if np.any(fallback_rows):
            averaged[fallback_rows] = current_strategy[fallback_rows]
        return averaged

    @staticmethod
    def _apply_dcfr_discount(
        regrets: dict[str, np.ndarray],
        strategy_sums: dict[str, np.ndarray],
        iteration: int,
    ) -> None:
        alpha = iteration / (iteration + 1.0)
        for table in (regrets, strategy_sums):
            for values in table.values():
                values *= alpha
