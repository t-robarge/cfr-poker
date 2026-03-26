from __future__ import annotations

import unittest

from hulhe_bot.abstract_game import AbstractGameBuilder
from hulhe_bot.config import AbstractHULHEConfig


class AbstractGameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AbstractHULHEConfig(
            seed=11,
            abstraction_samples=120,
            river_payoff_samples=120,
            flop_rollout_samples=4,
            turn_rollout_samples=4,
        )
        self.spec = AbstractGameBuilder(self.config).build_spec()

    def test_initial_distribution_sums_to_one(self) -> None:
        total = sum(self.spec.initial_bucket_distribution.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_transition_probabilities_sum_to_one(self) -> None:
        for street, sources in self.spec.street_transitions.items():
            for source, destinations in list(sources.items())[:10]:
                self.assertAlmostEqual(sum(destinations.values()), 1.0, places=6, msg=f"{street}:{source}")

    def test_river_payoffs_are_antisymmetric(self) -> None:
        checked = 0
        for pair, share in self.spec.river_showdown_share.items():
            left, right = pair.split("-")
            opposite = f"{right}-{left}"
            if opposite in self.spec.river_showdown_share:
                self.assertAlmostEqual(share + self.spec.river_showdown_share[opposite], 1.0, places=6)
                checked += 1
            if checked >= 10:
                break
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

