from __future__ import annotations

from functools import lru_cache

from .cards import (
    bucket_from_percentile,
    canonical_state_key,
    exact_showdown_share,
    monte_carlo_showdown_share,
    preflop_class_index,
    stable_seed,
)
from .config import AbstractHULHEConfig
from .models import BucketingResult, Street


class Bucketer:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()

    def bucket(self, street: Street, hole_cards: tuple[str, str], board: tuple[str, ...]) -> int:
        return self.bucket_details(street, hole_cards, board).bucket_id

    @lru_cache(maxsize=200_000)
    def bucket_details(
        self,
        street: Street,
        hole_cards: tuple[str, str],
        board: tuple[str, ...],
    ) -> BucketingResult:
        canonical_key = canonical_state_key(street, hole_cards, board)
        if street == Street.PREFLOP:
            percentile = monte_carlo_showdown_share(
                hole_cards,
                (),
                samples=32,
                seed=stable_seed((self.config.seed, "preflop", canonical_key)),
            )
            return BucketingResult(
                bucket_id=preflop_class_index(hole_cards),
                percentile=percentile,
                canonical_key=canonical_key,
            )
        if street == Street.FLOP:
            percentile = monte_carlo_showdown_share(
                hole_cards,
                board,
                samples=self.config.flop_rollout_samples,
                seed=stable_seed((self.config.seed, "flop", canonical_key)),
            )
            return BucketingResult(
                bucket_id=bucket_from_percentile(percentile, self.config.flop_buckets),
                percentile=percentile,
                canonical_key=canonical_key,
            )
        if street == Street.TURN:
            percentile = monte_carlo_showdown_share(
                hole_cards,
                board,
                samples=self.config.turn_rollout_samples,
                seed=stable_seed((self.config.seed, "turn", canonical_key)),
            )
            return BucketingResult(
                bucket_id=bucket_from_percentile(percentile, self.config.turn_buckets),
                percentile=percentile,
                canonical_key=canonical_key,
            )
        percentile = exact_showdown_share(hole_cards, board)
        return BucketingResult(
            bucket_id=bucket_from_percentile(percentile, self.config.river_buckets),
            percentile=percentile,
            canonical_key=canonical_key,
        )

