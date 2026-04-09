from __future__ import annotations

import unittest

from hulhe_bot.bucketing import Bucketer
from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.models import Street


class BucketingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bucketer = Bucketer(AbstractHULHEConfig(seed=5, flop_rollout_samples=4, turn_rollout_samples=4))

    def test_preflop_suit_invariance(self) -> None:
        hearts = self.bucketer.bucket_details(Street.PREFLOP, ("Ah", "Kh"), ())
        spades = self.bucketer.bucket_details(Street.PREFLOP, ("As", "Ks"), ())
        self.assertEqual(hearts.bucket_id, spades.bucket_id)
        self.assertAlmostEqual(hearts.percentile, spades.percentile)

    def test_river_bucket_is_deterministic(self) -> None:
        result_a = self.bucketer.bucket_details(Street.RIVER, ("Ah", "Kh"), ("Qh", "Jh", "Th", "2c", "3d"))
        result_b = self.bucketer.bucket_details(Street.RIVER, ("Ah", "Kh"), ("Qh", "Jh", "Th", "2c", "3d"))
        self.assertEqual(result_a.bucket_id, result_b.bucket_id)
        self.assertAlmostEqual(result_a.percentile, result_b.percentile)

    def test_flop_feature_bucket_distinguishes_draw_pressure(self) -> None:
        flush_draw = self.bucketer.bucket_details(Street.FLOP, ("Ah", "Kh"), ("Qh", "7h", "2c"))
        dry_overcards = self.bucketer.bucket_details(Street.FLOP, ("Ad", "Kc"), ("Qs", "7h", "2d"))
        self.assertNotEqual(flush_draw.feature_bucket, dry_overcards.feature_bucket)

    def test_flop_feature_bucket_distinguishes_pair_quality(self) -> None:
        top_pair = self.bucketer.bucket_details(Street.FLOP, ("Ah", "Kd"), ("Ac", "7s", "2d"))
        weak_pair = self.bucketer.bucket_details(Street.FLOP, ("7h", "Qd"), ("Ac", "7s", "2d"))
        self.assertNotEqual(top_pair.feature_bucket, weak_pair.feature_bucket)


if __name__ == "__main__":
    unittest.main()
