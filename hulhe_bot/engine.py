from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .cards import compare_hands, shuffled_deck
from .config import AbstractHULHEConfig
from .models import Action, Street


@dataclass(frozen=True, slots=True)
class ActionRecord:
    street: Street
    player: int
    action: Action
    amount: int


@dataclass(frozen=True, slots=True)
class GameState:
    street: Street
    current_player: int
    button: int
    hole_cards: tuple[tuple[str, str], tuple[str, str]]
    board: tuple[str, ...]
    deck: tuple[str, ...]
    deck_index: int
    pot: int
    stacks: tuple[int, int]
    total_contrib: tuple[int, int]
    street_contrib: tuple[int, int]
    current_to_call: int
    bet_level: int
    acted_this_street: tuple[int, ...]
    last_aggressor: int | None
    history_by_street: tuple[str, str, str, str]
    action_history: tuple[ActionRecord, ...]
    terminal: bool = False
    folded_player: int | None = None
    payoffs: tuple[float, float] = (0.0, 0.0)

    def history_id(self) -> str:
        parts = []
        for street, sequence in zip(Street, self.history_by_street):
            if sequence:
                parts.append(f"{street.value[0]}:{sequence}")
        return "/".join(parts) or "root"


class LimitHoldemGame:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()

    def new_hand(self, seed: int | None = None, button: int = 0) -> GameState:
        seed = self.config.seed if seed is None else seed
        deck = shuffled_deck(seed)
        hole_cards = (
            (deck[0], deck[2]),
            (deck[1], deck[3]),
        )
        deck_index = 4
        sb_player = button
        bb_player = 1 - button
        stacks = [self.config.starting_stack, self.config.starting_stack]
        total_contrib = [0, 0]
        street_contrib = [0, 0]
        for player, blind in ((sb_player, self.config.small_blind), (bb_player, self.config.big_blind)):
            stacks[player] -= blind
            total_contrib[player] += blind
            street_contrib[player] += blind
        return GameState(
            street=Street.PREFLOP,
            current_player=sb_player,
            button=button,
            hole_cards=hole_cards,
            board=(),
            deck=tuple(deck),
            deck_index=deck_index,
            pot=self.config.small_blind + self.config.big_blind,
            stacks=tuple(stacks),
            total_contrib=tuple(total_contrib),
            street_contrib=tuple(street_contrib),
            current_to_call=self.config.big_blind,
            bet_level=1,
            acted_this_street=(),
            last_aggressor=None,
            history_by_street=("", "", "", ""),
            action_history=(),
        )

    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        if state.terminal:
            return ()
        to_call = state.current_to_call - state.street_contrib[state.current_player]
        actions: list[Action] = [Action.FOLD] if to_call > 0 else []
        actions.append(Action.CALL if to_call > 0 else Action.CHECK)
        if state.bet_level < self.config.bet_cap:
            actions.append(Action.RAISE)
        return tuple(actions)

    def apply_action(self, state: GameState, action: Action) -> GameState:
        if state.terminal:
            raise ValueError("Cannot act on a terminal state.")
        if action not in self.legal_actions(state):
            raise ValueError(f"Illegal action {action} for {state.history_id()}")
        player = state.current_player
        opponent = 1 - player
        to_call = state.current_to_call - state.street_contrib[player]
        stacks = list(state.stacks)
        total_contrib = list(state.total_contrib)
        street_contrib = list(state.street_contrib)
        history_by_street = list(state.history_by_street)
        acted = set(state.acted_this_street)
        action_history = list(state.action_history)
        pot = state.pot
        last_aggressor = state.last_aggressor
        current_to_call = state.current_to_call
        bet_level = state.bet_level

        def commit(amount: int) -> None:
            nonlocal pot
            if amount < 0:
                raise ValueError("Cannot commit a negative amount.")
            if amount > stacks[player]:
                raise ValueError("Starting stack is too shallow for limit betting assumptions.")
            stacks[player] -= amount
            total_contrib[player] += amount
            street_contrib[player] += amount
            pot += amount

        street_index = list(Street).index(state.street)
        code = {
            Action.FOLD: "f",
            Action.CHECK: "k",
            Action.CALL: "c",
            Action.RAISE: "r",
        }[action]
        amount_committed = 0

        if action == Action.FOLD:
            history_by_street[street_index] += code
            action_history.append(ActionRecord(state.street, player, action, 0))
            return self._settle_fold(
                state,
                folded_player=player,
                history_by_street=tuple(history_by_street),
                action_history=tuple(action_history),
            )

        if action == Action.CHECK:
            history_by_street[street_index] += code
            acted.add(player)
            action_history.append(ActionRecord(state.street, player, action, 0))
            if len(acted) == 2:
                return self._advance_street(
                    replace(
                        state,
                        pot=pot,
                        stacks=tuple(stacks),
                        total_contrib=tuple(total_contrib),
                        street_contrib=tuple(street_contrib),
                        history_by_street=tuple(history_by_street),
                        action_history=tuple(action_history),
                    )
                )
            return replace(
                state,
                current_player=opponent,
                pot=pot,
                stacks=tuple(stacks),
                total_contrib=tuple(total_contrib),
                street_contrib=tuple(street_contrib),
                acted_this_street=tuple(sorted(acted)),
                history_by_street=tuple(history_by_street),
                action_history=tuple(action_history),
            )

        if action == Action.CALL:
            amount_committed = max(0, to_call)
            commit(amount_committed)
            history_by_street[street_index] += code
            acted.add(player)
            action_history.append(ActionRecord(state.street, player, action, amount_committed))
            if state.last_aggressor is not None:
                return self._advance_street(
                    replace(
                        state,
                        pot=pot,
                        stacks=tuple(stacks),
                        total_contrib=tuple(total_contrib),
                        street_contrib=tuple(street_contrib),
                        history_by_street=tuple(history_by_street),
                        action_history=tuple(action_history),
                    )
                )
            if len(acted) == 2:
                return self._advance_street(
                    replace(
                        state,
                        pot=pot,
                        stacks=tuple(stacks),
                        total_contrib=tuple(total_contrib),
                        street_contrib=tuple(street_contrib),
                        history_by_street=tuple(history_by_street),
                        action_history=tuple(action_history),
                    )
                )
            return replace(
                state,
                current_player=opponent,
                pot=pot,
                stacks=tuple(stacks),
                total_contrib=tuple(total_contrib),
                street_contrib=tuple(street_contrib),
                acted_this_street=tuple(sorted(acted)),
                history_by_street=tuple(history_by_street),
                action_history=tuple(action_history),
            )

        bet_size = self.config.small_bet if state.street in {Street.PREFLOP, Street.FLOP} else self.config.big_bet
        amount_committed = max(0, to_call) + bet_size
        commit(amount_committed)
        current_to_call += bet_size
        bet_level += 1
        acted = {player}
        last_aggressor = player
        history_by_street[street_index] += code
        action_history.append(ActionRecord(state.street, player, action, amount_committed))
        return replace(
            state,
            current_player=opponent,
            pot=pot,
            stacks=tuple(stacks),
            total_contrib=tuple(total_contrib),
            street_contrib=tuple(street_contrib),
            current_to_call=current_to_call,
            bet_level=bet_level,
            acted_this_street=tuple(sorted(acted)),
            last_aggressor=last_aggressor,
            history_by_street=tuple(history_by_street),
            action_history=tuple(action_history),
        )

    def _advance_street(self, state: GameState) -> GameState:
        if state.street == Street.RIVER:
            return self._settle_showdown(state)
        board = list(state.board)
        deck_index = state.deck_index
        if state.street == Street.PREFLOP:
            board.extend(state.deck[deck_index : deck_index + 3])
            deck_index += 3
            next_street = Street.FLOP
        elif state.street == Street.FLOP:
            board.append(state.deck[deck_index])
            deck_index += 1
            next_street = Street.TURN
        else:
            board.append(state.deck[deck_index])
            deck_index += 1
            next_street = Street.RIVER
        return replace(
            state,
            street=next_street,
            current_player=1 - state.button,
            board=tuple(board),
            deck_index=deck_index,
            street_contrib=(0, 0),
            current_to_call=0,
            bet_level=0,
            acted_this_street=(),
            last_aggressor=None,
        )

    def _settle_fold(
        self,
        state: GameState,
        folded_player: int,
        history_by_street: tuple[str, str, str, str],
        action_history: tuple[ActionRecord, ...],
    ) -> GameState:
        winner = 1 - folded_player
        payoffs = self._compute_payoffs(state.total_contrib, state.pot, 1.0 if winner == 0 else 0.0)
        return replace(
            state,
            terminal=True,
            folded_player=folded_player,
            history_by_street=history_by_street,
            action_history=action_history,
            payoffs=payoffs,
        )

    def _settle_showdown(self, state: GameState) -> GameState:
        hero_cards = tuple(state.hole_cards[0] + state.board)
        villain_cards = tuple(state.hole_cards[1] + state.board)
        result = compare_hands(hero_cards, villain_cards)
        if result > 0:
            p0_share = 1.0
        elif result < 0:
            p0_share = 0.0
        else:
            p0_share = 0.5
        payoffs = self._compute_payoffs(state.total_contrib, state.pot, p0_share)
        return replace(state, terminal=True, payoffs=payoffs)

    def _compute_payoffs(
        self,
        total_contrib: Iterable[int],
        pot: int,
        p0_share: float,
    ) -> tuple[float, float]:
        c0, c1 = total_contrib
        p0 = (pot * p0_share - c0) / self.config.reward_unit
        p1 = (pot * (1.0 - p0_share) - c1) / self.config.reward_unit
        return (p0, p1)
