from __future__ import annotations

import tempfile
import unittest

from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.experiments import ExperimentRunner


class ExperimentTests(unittest.TestCase):
    def test_experiment_runner_creates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AbstractHULHEConfig(
                seed=23,
                abstraction_samples=40,
                river_payoff_samples=40,
                flop_rollout_samples=4,
                turn_rollout_samples=4,
                training_iterations=12,
                smoke_iterations=6,
                checkpoint_every=6,
                fine_tune_hands=8,
                fine_tune_validation_hands=6,
                fine_tune_eval_interval=4,
                default_eval_hands=8,
                default_eval_seeds=1,
                abstraction_file=f"{tmpdir}/abstract.game",
                blueprint_file=f"{tmpdir}/blueprint.json",
                tuned_file=f"{tmpdir}/tuned.json",
                experiment_report_file=f"{tmpdir}/report.json",
            )
            runner = ExperimentRunner(config)
            report = runner.run(rebuild=True, eval_hands=8, eval_seeds=1)
            self.assertIn("evaluations", report)
            self.assertIn("blueprint", report["evaluations"])
            self.assertIn("random", report["evaluations"]["blueprint"])


if __name__ == "__main__":
    unittest.main()
