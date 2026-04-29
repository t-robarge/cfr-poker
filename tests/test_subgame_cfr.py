from __future__ import annotations

import tempfile
import unittest

from hulhe_bot.abstract_game import AbstractGameBuilder
from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.models import Action, Observation, Street
from hulhe_bot.subgame_cfr import CFRSubgameSolver


class CFRSubgameSolverTests(unittest.TestCase):
    @staticmethod
    def _find_live_node(spec, street: Street):
        for node in spec.nodes.values():
            if node["street"] == street.value and node["terminal_type"] is None and node["legal_actions"]:
                return node
        raise AssertionError(f"No live node found for {street.value}")

    def test_river_local_cfr_returns_normalized_legal_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AbstractHULHEConfig(
                seed=31,
                abstraction_samples=60,
                river_payoff_samples=60,
                flop_rollout_samples=4,
                turn_rollout_samples=4,
                abstraction_file=f"{tmpdir}/abstract.game",
                use_subgame_solving=True,
                subgame_mode="cfr",
                subgame_blend_weight=1.0,
                turn_subgame_blend_weight=1.0,
                subgame_cfr_iterations=20,
            )
            builder = AbstractGameBuilder(config)
            builder.build(config)

            solver = CFRSubgameSolver(config)
            spec = solver._load_spec()
            node = self._find_live_node(spec, Street.RIVER)
            river_dist = solver._street_distribution(Street.RIVER)
            self.assertTrue(river_dist)
            pair = next(iter(river_dist))
            hero_bucket = int(pair.split("-")[node["current_player"]])
            to_call = node["current_to_call"] - node["street_contrib"][node["current_player"]]

            obs = Observation(
                acting_player=node["current_player"],
                button=0,
                street=Street.RIVER,
                hole_cards=("As", "Kd"),
                board=("2c", "7d", "Jh", "Qs", "Tc"),
                history_id=node["history_id"],
                legal_actions=tuple(Action(name) for name in node["legal_actions"]),
                to_call=to_call,
                pot=node["pot"],
                bucket_id=hero_bucket,
                bucket_percentile=0.75,
            )

            self.assertIn(obs.history_id, spec.nodes)
            base = {action: 1.0 / len(obs.legal_actions) for action in obs.legal_actions}
            dist = solver.refine(obs, base)
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
            self.assertSetEqual(set(dist.keys()), set(obs.legal_actions))

    def test_turn_local_cfr_respects_action_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AbstractHULHEConfig(
                seed=37,
                abstraction_samples=60,
                river_payoff_samples=60,
                flop_rollout_samples=4,
                turn_rollout_samples=4,
                abstraction_file=f"{tmpdir}/abstract.game",
                use_subgame_solving=True,
                subgame_mode="cfr",
                subgame_blend_weight=1.0,
                turn_subgame_blend_weight=1.0,
                subgame_cfr_iterations=20,
            )
            AbstractGameBuilder(config).build(config)
            solver = CFRSubgameSolver(config)
            spec = solver._load_spec()
            node = self._find_live_node(spec, Street.TURN)
            turn_dist = solver._street_distribution(Street.TURN)
            self.assertTrue(turn_dist)
            pair = next(iter(turn_dist))
            hero_bucket = int(pair.split("-")[node["current_player"]])
            to_call = node["current_to_call"] - node["street_contrib"][node["current_player"]]

            obs = Observation(
                acting_player=node["current_player"],
                button=0,
                street=Street.TURN,
                hole_cards=("As", "Kd"),
                board=("2c", "7d", "Jh", "Qs"),
                history_id=node["history_id"],
                legal_actions=tuple(Action(name) for name in node["legal_actions"]),
                to_call=to_call,
                pot=node["pot"],
                bucket_id=hero_bucket,
                bucket_percentile=0.65,
            )
            self.assertIn(obs.history_id, spec.nodes)
            base = {action: 1.0 / len(obs.legal_actions) for action in obs.legal_actions}
            dist = solver.refine(obs, base)
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
            self.assertSetEqual(set(dist.keys()), set(obs.legal_actions))

    def test_exact_river_resolve_path_returns_normalized_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AbstractHULHEConfig(
                seed=43,
                abstraction_samples=60,
                river_payoff_samples=60,
                flop_rollout_samples=4,
                turn_rollout_samples=4,
                abstraction_file=f"{tmpdir}/abstract.game",
                use_subgame_solving=True,
                subgame_mode="cfr",
                subgame_blend_weight=1.0,
                turn_subgame_blend_weight=1.0,
                subgame_cfr_iterations=20,
                subgame_exact_resolve_river=True,
                subgame_exact_posterior_river=False,
            )
            AbstractGameBuilder(config).build(config)
            solver = CFRSubgameSolver(
                config,
                base_policy_lookup=lambda obs: (
                    {action: 1.0 / len(obs.legal_actions) for action in obs.legal_actions}
                    if obs.legal_actions
                    else {}
                ),
            )
            solver._root_distribution = lambda observation: (_ for _ in ()).throw(AssertionError("bucket fallback used"))
            spec = solver._load_spec()
            node = self._find_live_node(spec, Street.RIVER)
            obs = Observation(
                acting_player=node["current_player"],
                button=0,
                street=Street.RIVER,
                hole_cards=("As", "Kd"),
                board=("2c", "7d", "Jh", "Qs", "Tc"),
                history_id=node["history_id"],
                legal_actions=tuple(Action(name) for name in node["legal_actions"]),
                to_call=node["current_to_call"] - node["street_contrib"][node["current_player"]],
                pot=node["pot"],
                bucket_id=13,
                bucket_percentile=0.72,
            )
            base = {action: 1.0 / len(obs.legal_actions) for action in obs.legal_actions}
            dist = solver.refine(obs, base)
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
            self.assertSetEqual(set(dist.keys()), set(obs.legal_actions))


if __name__ == "__main__":
    unittest.main()
