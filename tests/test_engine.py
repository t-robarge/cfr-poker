from __future__ import annotations

import unittest

from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.engine import LimitHoldemGame
from hulhe_bot.models import Action, Street


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AbstractHULHEConfig(seed=13)
        self.game = LimitHoldemGame(self.config)

    def test_preflop_call_then_check_advances_to_flop(self) -> None:
        state = self.game.new_hand(seed=13)
        state = self.game.apply_action(state, Action.CALL)
        state = self.game.apply_action(state, Action.CHECK)
        self.assertEqual(state.street, Street.FLOP)
        self.assertEqual(len(state.board), 3)
        self.assertFalse(state.terminal)

    def test_raise_cap_is_enforced(self) -> None:
        state = self.game.new_hand(seed=17)
        state = self.game.apply_action(state, Action.RAISE)
        state = self.game.apply_action(state, Action.RAISE)
        state = self.game.apply_action(state, Action.RAISE)
        self.assertNotIn(Action.RAISE, self.game.legal_actions(state))

    def test_fold_is_terminal_and_zero_sum(self) -> None:
        state = self.game.new_hand(seed=21)
        state = self.game.apply_action(state, Action.FOLD)
        self.assertTrue(state.terminal)
        self.assertAlmostEqual(sum(state.payoffs), 0.0)


if __name__ == "__main__":
    unittest.main()

