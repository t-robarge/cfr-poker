"""Turn/river public-state refinement.

The original implementation only adjusted river play using exact equity.
This module expands that idea into a lightweight public-state resolver
for both the turn and river:

* **Turn**: use current showdown equity, pot odds, hand class, draw
  potential, and board texture to approximate a depth-limited resolve.
* **River**: use exact equity together with pot-odds-inspired thresholds
  for value betting, bluffing, bluff-catching, and value raising.

The resolver is intentionally lightweight: it does not build a full local
game tree like Libratus / DeepStack / ReBeL, but it moves this project in
that direction by performing street-aware public-state refinement rather
than applying a single static blueprint everywhere.
"""
from __future__ import annotations

from collections import Counter

from .cards import draw_class, made_hand_class
from .config import AbstractHULHEConfig
from .models import Action, Observation, Street
from .policy import normalize_distribution


class PublicSubgameResolver:
    """Lightweight turn/river public-state resolver.

    The goal is not to perfectly solve a local subgame, but to move from a
    static blueprint toward a depth-limited, street-aware refinement.

    On the **turn**, we treat current equity as a noisy proxy for future EV,
    then modulate it by hand strength, draw potential, pot odds, and board
    texture. This approximates a small local resolve with heuristic leaf
    values.

    On the **river**, we use exact equity and analytic pot-odds logic similar
    to the original implementation, but with slightly stronger value / bluff /
    bluff-catch differentiation.

    The result is blended with the incoming ``blueprint_dist`` using
    a configurable street-dependent weight to preserve an opt-out guarantee.
    """

    def __init__(self, config: AbstractHULHEConfig):
        self.config = config
        self.river_blend_weight: float = getattr(config, "subgame_blend_weight", 0.5)
        self.turn_blend_weight: float = getattr(
            config, "turn_subgame_blend_weight", self.river_blend_weight
        )

    def refine(
        self,
        observation: Observation,
        blueprint_dist: dict[Action, float],
    ) -> dict[Action, float]:
        """Return a refined distribution for the turn/river (or pass-through)."""
        if observation.street not in {Street.TURN, Street.RIVER}:
            return blueprint_dist

        legal_actions = list(observation.legal_actions)
        if observation.street is Street.RIVER:
            subgame_prefs = self._compute_river_preferences(observation, legal_actions)
            blend_weight = self.river_blend_weight
        else:
            subgame_prefs = self._compute_turn_preferences(observation, legal_actions)
            blend_weight = self.turn_blend_weight

        subgame_dist = self._prefs_to_dist(subgame_prefs, legal_actions)

        # Opt-out blending with blueprint
        blended = {
            a: (1.0 - blend_weight) * blueprint_dist.get(a, 0.0)
            + blend_weight * subgame_dist.get(a, 0.0)
            for a in legal_actions
        }
        return normalize_distribution(blended)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_river_preferences(
        self,
        observation: Observation,
        legal_actions: list[Action],
    ) -> dict[Action, float]:
        equity = observation.bucket_percentile
        pot = observation.pot
        to_call = observation.to_call
        bet_size = self.config.big_bet
        prefs: dict[Action, float] = {}

        if to_call == 0:
            # Not facing a bet ─ can CHECK or BET (represented as RAISE).
            value_threshold = 1.0 - bet_size / max(1, pot + 2 * bet_size)
            bluff_ratio = bet_size / max(1, pot + bet_size)

            if equity >= value_threshold:
                prefs[Action.RAISE] = 4.0
                prefs[Action.CHECK] = 0.10
            elif equity <= bluff_ratio * 0.5:
                # Worst hands → low-frequency bluffing region.
                prefs[Action.RAISE] = bluff_ratio * 0.9
                prefs[Action.CHECK] = 1.0 - bluff_ratio
            else:
                # Medium hands → mostly check behind.
                prefs[Action.CHECK] = 1.0
                prefs[Action.RAISE] = 0.05
        else:
            # Facing a bet ─ can FOLD, CALL, or RAISE.
            pot_odds = to_call / max(1, pot + to_call)
            strong_value_threshold = max(0.82, pot_odds + 0.25)

            if equity >= strong_value_threshold and Action.RAISE in legal_actions:
                prefs[Action.RAISE] = 2.5
                prefs[Action.CALL] = 0.4
                prefs[Action.FOLD] = 0.0
            elif equity >= pot_odds:
                call_str = (equity - pot_odds) / max(0.01, 1.0 - pot_odds)
                prefs[Action.CALL] = 1.0 + 0.25 * call_str
                prefs[Action.FOLD] = 0.0
                if Action.RAISE in legal_actions:
                    prefs[Action.RAISE] = call_str * 0.5
            else:
                fold_str = (pot_odds - equity) / max(0.01, pot_odds)
                prefs[Action.FOLD] = max(0.0, fold_str)
                prefs[Action.CALL] = max(0.0, 1.0 - fold_str)
                if Action.RAISE in legal_actions:
                    prefs[Action.RAISE] = 0.0

        return prefs

    def _compute_turn_preferences(
        self,
        observation: Observation,
        legal_actions: list[Action],
    ) -> dict[Action, float]:
        equity = observation.bucket_percentile
        pot = observation.pot
        to_call = observation.to_call
        bet_size = self.config.big_bet

        made_class = made_hand_class(observation.hole_cards, observation.board)
        draw_strength = draw_class(observation.hole_cards, observation.board)
        board_wetness = self._board_wetness(observation.board)

        # Future realization adjustment: strong made hands realize better, but
        # strong draws also deserve extra turn aggression as semi-bluffs.
        realization = (
            0.06 * made_class
            + 0.05 * draw_strength
            - 0.03 * board_wetness
        )
        effective_equity = min(max(equity + realization, 0.0), 1.0)

        prefs: dict[Action, float] = {}
        if to_call == 0:
            value_threshold = max(0.62, 0.72 - 0.04 * made_class)
            semibluff_threshold = max(0.24, 0.42 - 0.05 * draw_strength)

            if effective_equity >= value_threshold:
                prefs[Action.RAISE] = 3.0 + 0.4 * made_class
                prefs[Action.CHECK] = 0.2
            elif draw_strength >= 1 and effective_equity >= semibluff_threshold:
                prefs[Action.RAISE] = 1.2 + 0.5 * draw_strength
                prefs[Action.CHECK] = 0.9
            else:
                prefs[Action.CHECK] = 1.0 + 0.2 * made_class
                prefs[Action.RAISE] = 0.05 if effective_equity > 0.55 else 0.0
            return prefs

        pot_odds = to_call / max(1, pot + to_call)
        pressure = to_call / max(1, bet_size)
        call_threshold = max(0.0, pot_odds - 0.04 * draw_strength - 0.03 * made_class)
        raise_threshold = min(0.95, pot_odds + 0.18 - 0.05 * draw_strength)

        if effective_equity >= raise_threshold and Action.RAISE in legal_actions:
            prefs[Action.RAISE] = 1.8 + 0.4 * made_class + 0.3 * draw_strength
            prefs[Action.CALL] = 0.6
            prefs[Action.FOLD] = 0.0
            return prefs

        if effective_equity >= call_threshold:
            call_bias = 1.0 + max(0.0, effective_equity - pot_odds)
            prefs[Action.CALL] = call_bias
            prefs[Action.FOLD] = 0.1 if effective_equity < pot_odds + 0.03 else 0.0
            if Action.RAISE in legal_actions and draw_strength >= 1 and pressure <= 1.5:
                prefs[Action.RAISE] = 0.35 + 0.25 * draw_strength
            return prefs

        # Against heavy turn pressure, fold substantially more with weak,
        # low-realization holdings, while still preserving calls with draws.
        fold_pressure = max(0.0, pot_odds - effective_equity) * (1.3 + 0.2 * pressure)
        air_penalty = 0.2 if draw_strength == 0 and made_class <= 1 else 0.0
        prefs[Action.FOLD] = min(1.0, max(0.25, fold_pressure + air_penalty))
        prefs[Action.CALL] = max(
            0.0,
            0.7 - prefs[Action.FOLD] + 0.2 * draw_strength + 0.1 * min(made_class, 2),
        )
        if Action.RAISE in legal_actions and draw_strength >= 2 and pressure <= 1.0:
            prefs[Action.RAISE] = 0.25
        return prefs

    @staticmethod
    def _board_wetness(board: tuple[str, ...]) -> int:
        if not board:
            return 0
        suit_counts = Counter(card[1] for card in board)
        rank_counts = Counter(card[0] for card in board)
        max_suit = max(suit_counts.values(), default=0)
        paired = 1 if max(rank_counts.values(), default=1) >= 2 else 0

        ranks = sorted({"23456789TJQKA".index(card[0]) for card in board})
        straightiness = 0
        for idx in range(len(ranks) - 1):
            if ranks[idx + 1] - ranks[idx] <= 2:
                straightiness += 1

        wet = 0
        if max_suit >= 2:
            wet += 1
        if max_suit >= 3:
            wet += 1
        if straightiness >= 2:
            wet += 1
        wet += paired
        return wet

    @staticmethod
    def _prefs_to_dist(
        prefs: dict[Action, float], legal_actions: list[Action]
    ) -> dict[Action, float]:
        only_legal = {a: max(0.0, prefs.get(a, 0.0)) for a in legal_actions}
        total = sum(only_legal.values())
        if total > 0:
            return {a: v / total for a, v in only_legal.items()}
        # Fallback: uniform over legal actions
        n = len(legal_actions)
        return {a: 1.0 / n for a in legal_actions}


# Backward-compatible export name used elsewhere in the project.
RiverSubgameSolver = PublicSubgameResolver
