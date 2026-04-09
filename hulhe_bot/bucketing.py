from __future__ import annotations

from functools import lru_cache

from .cards import (
    bucket_from_percentile,
    canonical_state_key,
    draw_class,
    exact_showdown_share,
    made_hand_class,
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
                feature_bucket=0,
            )
        if street == Street.FLOP:
            percentile = monte_carlo_showdown_share(
                hole_cards,
                board,
                samples=self.config.flop_rollout_samples,
                seed=stable_seed((self.config.seed, "flop", canonical_key)),
            )
            equity_bucket = bucket_from_percentile(percentile, self.config.flop_buckets)
            feature_bucket = self._feature_bucket(hole_cards, board)
            return BucketingResult(
                bucket_id=equity_bucket + self.config.flop_buckets * feature_bucket,
                percentile=percentile,
                canonical_key=canonical_key,
                feature_bucket=feature_bucket,
            )
        if street == Street.TURN:
            percentile = monte_carlo_showdown_share(
                hole_cards,
                board,
                samples=self.config.turn_rollout_samples,
                seed=stable_seed((self.config.seed, "turn", canonical_key)),
            )
            equity_bucket = bucket_from_percentile(percentile, self.config.turn_buckets)
            feature_bucket = self._feature_bucket(hole_cards, board)
            return BucketingResult(
                bucket_id=equity_bucket + self.config.turn_buckets * feature_bucket,
                percentile=percentile,
                canonical_key=canonical_key,
                feature_bucket=feature_bucket,
            )
        percentile = exact_showdown_share(hole_cards, board)
        equity_bucket = bucket_from_percentile(percentile, self.config.river_buckets)
        feature_bucket = made_hand_class(hole_cards, board)
        return BucketingResult(
            bucket_id=equity_bucket + self.config.river_buckets * feature_bucket,
            percentile=percentile,
            canonical_key=canonical_key,
            feature_bucket=feature_bucket,
        )

    @staticmethod
    def _feature_bucket(hole_cards: tuple[str, str], board: tuple[str, ...]) -> int:
        made_bucket = made_hand_class(hole_cards, board)
        draws = draw_class(hole_cards, board)
        return made_bucket * 3 + draws
