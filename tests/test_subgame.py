from __future__ import annotations

import unittest

from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.models import Action, Observation, Street
from hulhe_bot.subgame import PublicSubgameResolver


class SubgameResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AbstractHULHEConfig(
            use_subgame_solving=True,
            subgame_blend_weight=1.0,
            turn_subgame_blend_weight=1.0,
        )
        self.resolver = PublicSubgameResolver(self.config)

    def test_preflop_passthrough(self) -> None:
        obs = Observation(
            acting_player=0,
            button=0,
            street=Street.PREFLOP,
            hole_cards=("As", "Ah"),
            board=(),
            history_id="root",
            legal_actions=(Action.CALL, Action.RAISE),
            to_call=1,
            pot=3,
            bucket_id=0,
            bucket_percentile=0.95,
        )
        blueprint = {Action.CALL: 0.4, Action.RAISE: 0.6}
        self.assertEqual(self.resolver.refine(obs, blueprint), blueprint)

    def test_turn_strong_hand_prefers_raise(self) -> None:
        obs = Observation(
            acting_player=0,
            button=0,
            street=Street.TURN,
            hole_cards=("As", "Ad"),
            board=("Ac", "7s", "2h", "9d"),
            history_id="f:cr/t:",
            legal_actions=(Action.CHECK, Action.RAISE),
            to_call=0,
            pot=16,
            bucket_id=0,
            bucket_percentile=0.88,
        )
        dist = self.resolver.refine(obs, {Action.CHECK: 0.8, Action.RAISE: 0.2})
        self.assertGreater(dist[Action.RAISE], dist[Action.CHECK])

    def test_turn_weak_hand_facing_bet_prefers_fold(self) -> None:
        obs = Observation(
            acting_player=0,
            button=0,
            street=Street.TURN,
            hole_cards=("7c", "2d"),
            board=("As", "Kd", "9h", "4c"),
            history_id="f:rcr/t:r",
            legal_actions=(Action.FOLD, Action.CALL, Action.RAISE),
            to_call=4,
            pot=20,
            bucket_id=0,
            bucket_percentile=0.05,
        )
        dist = self.resolver.refine(
            obs,
            {Action.FOLD: 0.2, Action.CALL: 0.7, Action.RAISE: 0.1},
        )
        self.assertGreater(dist[Action.FOLD], dist[Action.CALL])


if __name__ == "__main__":
    unittest.main()
