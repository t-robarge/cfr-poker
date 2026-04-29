from __future__ import annotations

from functools import lru_cache
import importlib
from typing import Any

import numpy as np

from .abstract_game import AbstractGameSpec
from .cards import RANK_ORDER, SUITS, evaluate_7card
from .config import AbstractHULHEConfig
from .exact_posterior import ExactRiverRangeTranslator, enumerate_legal_hole_combos
from .models import Action, Observation

Combo = tuple[str, str]

_ACTION_TO_CODE = {
    Action.FOLD: 0,
    Action.CHECK: 1,
    Action.CALL: 2,
    Action.RAISE: 3,
}
_CODE_TO_ACTION = {code: action for action, code in _ACTION_TO_CODE.items()}
_RANK_TO_INDEX = {rank: index for index, rank in enumerate(RANK_ORDER)}
_SUIT_TO_INDEX = {suit: index for index, suit in enumerate(SUITS)}


@lru_cache(maxsize=1)
def _native_module():
    try:
        return importlib.import_module("hulhe_bot._native_river")
    except ImportError:
        return None


def native_exact_river_available() -> bool:
    return _native_module() is not None


def _card_id(card: str) -> int:
    return _RANK_TO_INDEX[card[0]] * 4 + _SUIT_TO_INDEX[card[1]]


def _encode_hand_strength(score: tuple[int, ...]) -> int:
    padded = list(score)[:6] + [0] * max(0, 6 - len(score))
    value = 0
    for component in padded:
        value = value * 15 + int(component)
    return value


class NativeExactRiverSolver:
    """Compact Python<->C++ bridge for exact river resolving."""

    def __init__(self, config: AbstractHULHEConfig, spec: AbstractGameSpec):
        self.config = config
        self.spec = spec
        self._module = _native_module()
        self._tree_cache: dict[str, dict[str, Any]] = {}
        self._board_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    @property
    def available(self) -> bool:
        return self._module is not None

    def solve(
        self,
        observation: Observation,
        range_translator: ExactRiverRangeTranslator,
    ) -> dict[str, Any] | None:
        if self._module is None:
            return None

        root_node = self.spec.nodes.get(observation.history_id)
        if root_node is None or root_node["terminal_type"] is not None:
            return None

        board_payload = self._board_payload(observation.board)
        combo_index = board_payload["combo_index"]
        ranges = range_translator.translate_root_ranges(observation)
        range0 = self._range_array(ranges[0], combo_index)
        range1 = self._range_array(ranges[1], combo_index)
        if float(np.sum(range0)) <= 0.0 or float(np.sum(range1)) <= 0.0:
            return None

        tree_payload = self._tree_payload(observation.history_id)
        solved = self._module.solve_river_root(
            tree_payload,
            board_payload["combo_cards"],
            board_payload["hand_strengths"],
            range0,
            range1,
            int(getattr(self.config, "subgame_cfr_iterations", 200)),
            str(getattr(self.config, "subgame_cfr_algorithm", "cfr_plus")),
        )
        if not isinstance(solved, dict):
            return None

        average_strategy = np.asarray(solved["average_strategy"], dtype=np.float64)
        current_strategy = np.asarray(solved["current_strategy"], dtype=np.float64)
        root_action_codes = tree_payload["action_codes"][tree_payload["root_index"]][: tree_payload["num_actions"][tree_payload["root_index"]]]
        actions = tuple(_CODE_TO_ACTION[int(code)] for code in root_action_codes)
        return {
            "actions": actions,
            "average_strategy": average_strategy,
            "current_strategy": current_strategy,
            "combo_index": combo_index,
        }

    def _tree_payload(self, root_history_id: str) -> dict[str, Any]:
        cached = self._tree_cache.get(root_history_id)
        if cached is not None:
            return cached

        order: list[str] = []
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            order.append(node_id)
            node = self.spec.nodes[node_id]
            for action_name in node["legal_actions"]:
                visit(node["transitions"][action_name]["next_node"])

        visit(root_history_id)
        index = {node_id: offset for offset, node_id in enumerate(order)}
        max_actions = max((len(self.spec.nodes[node_id]["legal_actions"]) for node_id in order), default=0)
        current_player = np.zeros(len(order), dtype=np.int32)
        terminal_type = np.full(len(order), -1, dtype=np.int32)
        folded_player = np.full(len(order), -1, dtype=np.int32)
        pot = np.zeros(len(order), dtype=np.float64)
        contrib0 = np.zeros(len(order), dtype=np.float64)
        contrib1 = np.zeros(len(order), dtype=np.float64)
        num_actions = np.zeros(len(order), dtype=np.int32)
        action_codes = np.full((len(order), max_actions), -1, dtype=np.int32)
        child_index = np.full((len(order), max_actions), -1, dtype=np.int32)

        for offset, node_id in enumerate(order):
            node = self.spec.nodes[node_id]
            current_player[offset] = int(node["current_player"])
            pot[offset] = float(node["pot"])
            contrib0[offset] = float(node["total_contrib"][0])
            contrib1[offset] = float(node["total_contrib"][1])
            if node["terminal_type"] == "fold":
                terminal_type[offset] = 0
                folded_player[offset] = int(node["folded_player"])
            elif node["terminal_type"] == "showdown":
                terminal_type[offset] = 1

            legal_actions = [Action(name) for name in node["legal_actions"]]
            num_actions[offset] = len(legal_actions)
            for action_offset, action in enumerate(legal_actions):
                action_codes[offset, action_offset] = _ACTION_TO_CODE[action]
                child_index[offset, action_offset] = index[node["transitions"][action.value]["next_node"]]

        payload = {
            "root_index": int(index[root_history_id]),
            "reward_unit": float(self.config.reward_unit),
            "current_player": current_player,
            "terminal_type": terminal_type,
            "folded_player": folded_player,
            "pot": pot,
            "contrib0": contrib0,
            "contrib1": contrib1,
            "num_actions": num_actions,
            "action_codes": action_codes,
            "child_index": child_index,
        }
        self._tree_cache[root_history_id] = payload
        return payload

    def _board_payload(self, board: tuple[str, ...]) -> dict[str, Any]:
        cached = self._board_cache.get(board)
        if cached is not None:
            return cached

        combos = enumerate_legal_hole_combos(board)
        combo_index = {combo: offset for offset, combo in enumerate(combos)}
        combo_cards = np.asarray(
            [[_card_id(combo[0]), _card_id(combo[1])] for combo in combos],
            dtype=np.uint8,
        )
        hand_strengths = np.asarray(
            [_encode_hand_strength(evaluate_7card(tuple(combo + board))) for combo in combos],
            dtype=np.int64,
        )
        payload = {
            "combos": combos,
            "combo_index": combo_index,
            "combo_cards": combo_cards,
            "hand_strengths": hand_strengths,
        }
        self._board_cache[board] = payload
        return payload

    @staticmethod
    def _range_array(weights: dict[Combo, float], combo_index: dict[Combo, int]) -> np.ndarray:
        array = np.zeros(len(combo_index), dtype=np.float64)
        for combo, weight in weights.items():
            index = combo_index.get(combo)
            if index is not None:
                array[index] = float(weight)
        return array
