"""River subgame refinement using exact equity.

At the river all community cards are dealt, so we know our *exact*
showdown equity (stored in ``bucket_percentile``).  This module uses
that equity together with pot odds and game-theoretic betting
frequencies to produce a refined action distribution, then blends it
with the blueprint for an opt-out safety guarantee.

Activated by setting ``use_subgame_solving = true`` in the config JSON.
Only applies on the **river** street; earlier streets fall through to
the blueprint + residual strategy unchanged.
"""
from __future__ import annotations

import math
from typing import Any

from .config import AbstractHULHEConfig
from .models import Action, Observation, Street
from .policy import normalize_distribution


class RiverSubgameSolver:
    """Lightweight river subgame refinement (GTO-inspired analytical solver).

    Given exact equity *e*, current pot *P*, amount to call *c*, and the
    big-bet size *B*, the solver computes GTO-inspired frequencies:

    **Not facing a bet** (to_call == 0):
        * Value-bet threshold  = 1 − B / (P + 2B)
        * Bluff ratio          = B / (P + B)          (makes opponent indifferent)
        * Medium hands         → check

    **Facing a bet** (to_call > 0):
        * Pot odds             = c / (P + c)
        * Equity ≫ pot odds   → raise for value
        * Equity > pot odds    → call
        * Equity < pot odds    → fold (with small bluff-catch frequency)

    The result is blended with the incoming ``blueprint_dist`` using
    ``config.subgame_blend_weight`` to preserve an opt-out guarantee
    similar to the one used in Libratus.
    """

    def __init__(self, config: AbstractHULHEConfig):
        self.config = config
        self.blend_weight: float = getattr(config, "subgame_blend_weight", 0.5)

    def refine(
        self,
        observation: Observation,
        blueprint_dist: dict[Action, float],
    ) -> dict[Action, float]:
        """Return a refined distribution for the river (or pass-through)."""
        if observation.street is not Street.RIVER:
            return blueprint_dist

        equity = observation.bucket_percentile
        pot = observation.pot
        to_call = observation.to_call
        legal_actions = list(observation.legal_actions)
        bet_size = self.config.big_bet

        subgame_prefs = self._compute_preferences(
            equity, pot, to_call, legal_actions, bet_size
        )
        subgame_dist = self._prefs_to_dist(subgame_prefs, legal_actions)

        # Opt-out blending with blueprint
        w = self.blend_weight
        blended = {
            a: (1.0 - w) * blueprint_dist.get(a, 0.0)
            + w * subgame_dist.get(a, 0.0)
            for a in legal_actions
        }
        return normalize_distribution(blended)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_preferences(
        self,
        equity: float,
        pot: int,
        to_call: int,
        legal_actions: list[Action],
        bet_size: int,
    ) -> dict[Action, float]:
        prefs: dict[Action, float] = {}

        if to_call == 0:
            # Not facing a bet ─ can CHECK or BET (represented as RAISE)
            value_threshold = 1.0 - bet_size / max(1, pot + 2 * bet_size)
            bluff_ratio = bet_size / max(1, pot + bet_size)

            if equity >= value_threshold:
                prefs[Action.RAISE] = 3.0
                prefs[Action.CHECK] = 0.15
            elif equity <= bluff_ratio * 0.5:
                # Worst hands → bluff with optimal frequency
                prefs[Action.RAISE] = bluff_ratio
                prefs[Action.CHECK] = 1.0 - bluff_ratio
            else:
                # Medium hands → mostly check
                prefs[Action.CHECK] = 1.0
                prefs[Action.RAISE] = 0.05
        else:
            # Facing a bet ─ can FOLD, CALL, or RAISE
            pot_odds = to_call / max(1, pot + to_call)

            if equity >= 0.85 and Action.RAISE in legal_actions:
                prefs[Action.RAISE] = 2.0
                prefs[Action.CALL] = 0.5
                prefs[Action.FOLD] = 0.0
            elif equity >= pot_odds:
                call_str = (equity - pot_odds) / max(0.01, 1.0 - pot_odds)
                prefs[Action.CALL] = 1.0
                prefs[Action.FOLD] = 0.0
                if Action.RAISE in legal_actions:
                    prefs[Action.RAISE] = call_str * 0.3
            else:
                fold_str = (pot_odds - equity) / max(0.01, pot_odds)
                prefs[Action.FOLD] = fold_str
                prefs[Action.CALL] = 1.0 - fold_str
                if Action.RAISE in legal_actions:
                    prefs[Action.RAISE] = 0.0

        return prefs

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
