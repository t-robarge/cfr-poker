from __future__ import annotations

import unittest

from hulhe_bot.abstract_game import AbstractGameSpec
from hulhe_bot.bucketing import Bucketer
from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.exact_river_cfr import ExactRiverCFRSolver
from hulhe_bot.models import Action, Observation, Street


def _uniform_policy(observation: Observation) -> dict[Action, float]:
    if not observation.legal_actions:
        return {}
    probability = 1.0 / len(observation.legal_actions)
    return {action: probability for action in observation.legal_actions}


class ExactRiverCFRSolverTests(unittest.TestCase):
    def test_exact_river_solver_is_combo_sensitive_at_root(self) -> None:
        config = AbstractHULHEConfig(
            seed=13,
            subgame_cfr_iterations=200,
            subgame_cfr_algorithm="cfr_plus",
        )
        spec = AbstractGameSpec(
            format_name="hulhe_abstract_game_v1",
            root_node="root",
            config={},
            nodes={
                "root": {
                    "node_id": "root",
                    "street": Street.RIVER.value,
                    "current_player": 0,
                    "history_id": "root",
                    "legal_actions": [Action.CHECK.value, Action.RAISE.value],
                    "pot": 8,
                    "total_contrib": [4, 4],
                    "street_contrib": [0, 0],
                    "current_to_call": 0,
                    "bet_level": 0,
                    "terminal_type": None,
                    "folded_player": None,
                    "transitions": {
                        Action.CHECK.value: {"next_node": "showdown_check", "chance_stage": None},
                        Action.RAISE.value: {"next_node": "raise_response", "chance_stage": None},
                    },
                },
                "raise_response": {
                    "node_id": "raise_response",
                    "street": Street.RIVER.value,
                    "current_player": 1,
                    "history_id": "raise_response",
                    "legal_actions": [Action.FOLD.value, Action.CALL.value],
                    "pot": 12,
                    "total_contrib": [8, 4],
                    "street_contrib": [4, 0],
                    "current_to_call": 4,
                    "bet_level": 1,
                    "terminal_type": None,
                    "folded_player": None,
                    "transitions": {
                        Action.FOLD.value: {"next_node": "raise_fold", "chance_stage": None},
                        Action.CALL.value: {"next_node": "showdown_call", "chance_stage": None},
                    },
                },
                "showdown_check": {
                    "node_id": "showdown_check",
                    "street": Street.RIVER.value,
                    "current_player": 0,
                    "history_id": "showdown_check",
                    "legal_actions": [],
                    "pot": 8,
                    "total_contrib": [4, 4],
                    "street_contrib": [0, 0],
                    "current_to_call": 0,
                    "bet_level": 0,
                    "terminal_type": "showdown",
                    "folded_player": None,
                    "transitions": {},
                },
                "raise_fold": {
                    "node_id": "raise_fold",
                    "street": Street.RIVER.value,
                    "current_player": 0,
                    "history_id": "raise_fold",
                    "legal_actions": [],
                    "pot": 12,
                    "total_contrib": [8, 4],
                    "street_contrib": [4, 0],
                    "current_to_call": 4,
                    "bet_level": 1,
                    "terminal_type": "fold",
                    "folded_player": 1,
                    "transitions": {},
                },
                "showdown_call": {
                    "node_id": "showdown_call",
                    "street": Street.RIVER.value,
                    "current_player": 0,
                    "history_id": "showdown_call",
                    "legal_actions": [],
                    "pot": 16,
                    "total_contrib": [8, 8],
                    "street_contrib": [4, 4],
                    "current_to_call": 4,
                    "bet_level": 1,
                    "terminal_type": "showdown",
                    "folded_player": None,
                    "transitions": {},
                },
            },
            initial_bucket_distribution={"0-0": 1.0},
            street_transitions={},
            river_showdown_share={},
            metadata={},
        )
        board = ("As", "Ks", "Qs", "2d", "3c")
        hero_nuts = ("Js", "Ts")
        hero_bluff = ("8h", "9h")
        villain_range = {
            ("Ac", "Ad"): 0.5,
            ("Kc", "Qd"): 0.5,
        }
        hero_range = {
            hero_nuts: 0.5,
            hero_bluff: 0.5,
        }

        solver = ExactRiverCFRSolver(config, spec, _uniform_policy)
        solver.range_translator.translate_root_ranges = lambda obs: (hero_range, villain_range)
        bucketer = Bucketer(config)

        def make_observation(hole_cards: tuple[str, str]) -> Observation:
            details = bucketer.bucket_details(Street.RIVER, hole_cards, board)
            return Observation(
                acting_player=0,
                button=0,
                street=Street.RIVER,
                hole_cards=hole_cards,
                board=board,
                history_id="root",
                legal_actions=(Action.CHECK, Action.RAISE),
                to_call=0,
                pot=8,
                bucket_id=details.bucket_id,
                bucket_percentile=details.percentile,
            )

        nuts_dist = solver.refine(make_observation(hero_nuts))
        bluff_dist = solver.refine(make_observation(hero_bluff))

        self.assertAlmostEqual(sum(nuts_dist.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(bluff_dist.values()), 1.0, places=6)
        self.assertGreater(nuts_dist[Action.RAISE], bluff_dist[Action.RAISE])


if __name__ == "__main__":
    unittest.main()
