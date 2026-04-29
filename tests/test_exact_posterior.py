from __future__ import annotations

import tempfile
import unittest

from hulhe_bot.abstract_game import AbstractGameBuilder, AbstractGameSpec
from hulhe_bot.bucketing import Bucketer
from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.exact_posterior import ExactRiverRangeTranslator
from hulhe_bot.models import Action, Observation, Street


def _normalize(raw: dict[Action, float]) -> dict[Action, float]:
    total = sum(max(0.0, value) for value in raw.values())
    if total <= 0.0:
        uniform = 1.0 / max(1, len(raw))
        return {key: uniform for key in raw}
    return {key: max(0.0, value) / total for key, value in raw.items()}


def _street_board(board: tuple[str, ...], street: Street) -> tuple[str, ...]:
    if street is Street.PREFLOP:
        return ()
    if street is Street.FLOP:
        return board[:3]
    if street is Street.TURN:
        return board[:4]
    return board[:5]


class ExactRiverRangeTranslatorTests(unittest.TestCase):
    @staticmethod
    def _find_live_node(spec: AbstractGameSpec, street: Street) -> dict[str, object]:
        for node in spec.nodes.values():
            if (
                node["street"] == street.value
                and node["terminal_type"] is None
                and node["legal_actions"]
            ):
                return node
        raise AssertionError(f"No live node found for {street.value}")

    @staticmethod
    def _decode_history_action(token: str) -> Action:
        return {
            "f": Action.FOLD,
            "k": Action.CHECK,
            "c": Action.CALL,
            "r": Action.RAISE,
        }[token]

    @staticmethod
    def _parse_history(history_id: str) -> dict[Street, str]:
        sequences = {street: "" for street in Street}
        if history_id == "root":
            return sequences
        for section in history_id.split("/"):
            code, sequence = section.split(":", 1)
            street = {
                "p": Street.PREFLOP,
                "f": Street.FLOP,
                "t": Street.TURN,
                "r": Street.RIVER,
            }[code]
            sequences[street] = sequence
        return sequences

    def test_translator_reweights_villain_combos_from_bucket_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AbstractHULHEConfig(
                seed=41,
                abstraction_samples=80,
                river_payoff_samples=80,
                flop_rollout_samples=4,
                turn_rollout_samples=4,
                abstraction_file=f"{tmpdir}/abstract.game",
            )
            AbstractGameBuilder(config).build(config)
            spec = AbstractGameSpec.load(config.abstraction_file)
            bucketer = Bucketer(config)
            node = self._find_live_node(spec, Street.RIVER)
            board = ("2c", "7d", "Jh", "Qs", "Tc")
            hero_hole = ("As", "Kd")
            observation = Observation(
                acting_player=node["current_player"],
                button=0,
                street=Street.RIVER,
                hole_cards=hero_hole,
                board=board,
                history_id=node["history_id"],
                legal_actions=tuple(Action(name) for name in node["legal_actions"]),
                to_call=node["current_to_call"] - node["street_contrib"][node["current_player"]],
                pot=node["pot"],
                bucket_id=bucketer.bucket(Street.RIVER, hero_hole, board),
                bucket_percentile=bucketer.bucket_details(Street.RIVER, hero_hole, board).percentile,
            )

            villain_role = 1 - observation.relative_position
            current_node_id = spec.root_node
            target_history_id = None
            target_street = None
            target_action = None
            target_board = None
            history = self._parse_history(observation.history_id)
            for street in Street:
                for token in history[street]:
                    node_at_step = spec.nodes[current_node_id]
                    action = self._decode_history_action(token)
                    if target_history_id is None and (
                        node_at_step["current_player"] == villain_role
                        and len(node_at_step["legal_actions"]) > 1
                    ):
                        target_history_id = node_at_step["history_id"]
                        target_street = street
                        target_action = action
                        target_board = _street_board(board, street)
                    current_node_id = node_at_step["transitions"][action.value]["next_node"]

            self.assertIsNotNone(target_history_id)
            self.assertIsNotNone(target_street)
            self.assertIsNotNone(target_action)
            self.assertIsNotNone(target_board)

            uniform_translator = ExactRiverRangeTranslator(
                config,
                spec,
                lambda obs: (
                    {action: 1.0 / len(obs.legal_actions) for action in obs.legal_actions}
                    if obs.legal_actions
                    else {}
                ),
            )
            baseline_weights = uniform_translator.translate_villain_weights(observation)
            available_target_buckets = sorted(
                {
                    bucketer.bucket(target_street, combo, target_board)
                    for combo in baseline_weights
                }
            )
            self.assertGreaterEqual(len(available_target_buckets), 2)
            preferred_bucket = available_target_buckets[0]
            alternate_bucket = available_target_buckets[1]

            def policy_lookup(obs: Observation) -> dict[Action, float]:
                legal = list(obs.legal_actions)
                if not legal:
                    return {}
                if obs.history_id == target_history_id and obs.street is target_street:
                    preferred = target_action
                    alternate = next((action for action in legal if action != preferred), preferred)
                    raw = {action: 0.05 for action in legal}
                    if obs.bucket_id == preferred_bucket:
                        raw[preferred] = 0.85
                    elif obs.bucket_id == alternate_bucket:
                        raw[alternate] = 0.85
                    else:
                        raw[preferred] = 0.45
                        raw[alternate] = 0.45
                    return _normalize(raw)
                return {action: 1.0 / len(legal) for action in legal}

            translator = ExactRiverRangeTranslator(config, spec, policy_lookup)
            villain_weights = translator.translate_villain_weights(observation)
            self.assertAlmostEqual(sum(villain_weights.values()), 1.0, places=6)
            for combo in villain_weights:
                self.assertTrue(set(combo).isdisjoint(set(board)))
                self.assertTrue(set(combo).isdisjoint(set(hero_hole)))

            baseline_preferred_mass = 0.0
            baseline_alternate_mass = 0.0
            preferred_mass = 0.0
            alternate_mass = 0.0
            for combo, weight in baseline_weights.items():
                bucket_id = bucketer.bucket(target_street, combo, target_board)
                if bucket_id == preferred_bucket:
                    baseline_preferred_mass += weight
                elif bucket_id == alternate_bucket:
                    baseline_alternate_mass += weight
            for combo, weight in villain_weights.items():
                bucket_id = bucketer.bucket(target_street, combo, target_board)
                if bucket_id == preferred_bucket:
                    preferred_mass += weight
                elif bucket_id == alternate_bucket:
                    alternate_mass += weight
            self.assertGreater(preferred_mass, baseline_preferred_mass)
            self.assertLess(alternate_mass, baseline_alternate_mass)

            root_distribution = translator.root_bucket_distribution(observation)
            self.assertAlmostEqual(sum(root_distribution.values()), 1.0, places=6)
            for pair in root_distribution:
                left, right = pair.split("-")
                if observation.relative_position == 0:
                    self.assertEqual(int(left), observation.bucket_id)
                else:
                    self.assertEqual(int(right), observation.bucket_id)


if __name__ == "__main__":
    unittest.main()
