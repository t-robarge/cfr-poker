from __future__ import annotations

import tempfile
import unittest

from hulhe_bot.abstract_game import AbstractGameBuilder
from hulhe_bot.baselines import RandomAgent
from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.evaluator import Evaluator
from hulhe_bot.policy import PolicyRuntime
from hulhe_bot.rl import ResidualFineTuner
from hulhe_bot.trainer import LiteEFGTrainer
from hulhe_bot.baselines import PolicyAgent


class PipelineTests(unittest.TestCase):
    def test_smoke_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AbstractHULHEConfig(
                seed=19,
                abstraction_samples=80,
                river_payoff_samples=80,
                flop_rollout_samples=4,
                turn_rollout_samples=4,
                training_iterations=30,
                smoke_iterations=10,
                checkpoint_every=10,
                fine_tune_hands=20,
                fine_tune_validation_hands=10,
                fine_tune_eval_interval=10,
                abstraction_file=f"{tmpdir}/abstract.game",
                blueprint_file=f"{tmpdir}/blueprint.json",
                tuned_file=f"{tmpdir}/tuned.json",
            )

            builder = AbstractGameBuilder(config)
            game_file = builder.build(config)

            trainer = LiteEFGTrainer(config)
            blueprint = trainer.train(game_file, config=config, iterations=20, algorithm="mccfr")
            self.assertTrue(blueprint.policy_table)

            fine_tuner = ResidualFineTuner(config)
            tuned = fine_tuner.train(blueprint, episodes=10)

            evaluator = Evaluator(config)
            agent = PolicyAgent(PolicyRuntime(tuned, config, evaluator.bucketer), name="candidate")
            result = evaluator.match(agent, RandomAgent(), hands=20, seed=23)
            self.assertEqual(result.hands, 20)
            self.assertTrue(result.sb_per_100 == result.sb_per_100)


if __name__ == "__main__":
    unittest.main()

