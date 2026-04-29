from __future__ import annotations

import math
import random

from .baselines import Agent
from .bucketing import Bucketer
from .config import AbstractHULHEConfig
from .engine import GameState, LimitHoldemGame
from .models import MatchResult, Observation


class Evaluator:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()
        self.game = LimitHoldemGame(self.config)
        self.bucketer = Bucketer(self.config)

    def make_observation(self, state: GameState) -> Observation:
        player = state.current_player
        details = self.bucketer.bucket_details(state.street, state.hole_cards[player], state.board)
        return Observation(
            acting_player=player,
            button=state.button,
            street=state.street,
            hole_cards=state.hole_cards[player],
            board=state.board,
            history_id=state.history_id(),
            legal_actions=self.game.legal_actions(state),
            to_call=state.current_to_call - state.street_contrib[player],
            pot=state.pot,
            bucket_id=details.bucket_id,
            bucket_percentile=details.percentile,
        )

    def play_hand(self, agent_a: Agent, agent_b: Agent, seed: int, button: int) -> float:
        rng = random.Random(seed)
        state = self.game.new_hand(seed=seed, button=button)
        agents = {0: agent_a, 1: agent_b}
        while not state.terminal:
            observation = self.make_observation(state)
            action = agents[state.current_player].select_action(observation, rng)
            state = self.game.apply_action(state, action)
        return state.payoffs[0]

    def match(self, agent_a: Agent, agent_b: Agent, hands: int, seed: int) -> MatchResult:
        per_hand_results: list[float] = []
        total = 0.0
        checkpoint = max(1, hands // 10)  # Report every 10%
        for hand_index in range(hands):
            hand_seed = seed * 1_000_003 + hand_index
            button = hand_index % 2
            result = self.play_hand(agent_a, agent_b, hand_seed, button)
            total += result
            per_hand_results.append(result)
            if (hand_index + 1) % checkpoint == 0:
                current_sb_per_100 = (total / (hand_index + 1)) * 100.0
                print(f"      [MATCH] {hand_index + 1}/{hands} hands, sb/100: {current_sb_per_100:.2f}")
        sb_per_100 = (total / max(1, hands)) * 100.0
        return MatchResult(
            hands=hands,
            total_small_bets=total,
            sb_per_100=sb_per_100,
            per_hand_results=per_hand_results,
            metadata={"stdev": self._stdev(per_hand_results)},
        )

    @staticmethod
    def _stdev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)
