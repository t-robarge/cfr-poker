from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .abstract_game import AbstractGameBuilder
from .baselines import LoosePassiveAgent, PolicyAgent, RandomAgent, TightAggressiveAgent
from .config import AbstractHULHEConfig
from .evaluator import Evaluator
from .models import PolicyArtifact
from .policy import PolicyRuntime
from .rl import ResidualFineTuner
from .trainer import LiteEFGTrainer


def summarize_seed_results(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values) if values else 0.0
    ci95 = 0.0
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        ci95 = 1.96 * math.sqrt(variance / len(values))
    return {
        "sb_per_100_mean": mean,
        "sb_per_100_ci95": ci95,
        "seed_results": values,
    }


class ExperimentRunner:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()
        self.builder = AbstractGameBuilder(self.config)
        self.trainer = LiteEFGTrainer(self.config)
        self.fine_tuner = ResidualFineTuner(self.config)
        self.evaluator = Evaluator(self.config)

    def run(
        self,
        rebuild: bool = False,
        eval_hands: int | None = None,
        eval_seeds: int | None = None,
        run_ablation: bool = True,
        run_fine_tune: bool = True,
        report_path: str | None = None,
    ) -> dict[str, Any]:
        hands = eval_hands or self.config.default_eval_hands
        seeds = eval_seeds or self.config.default_eval_seeds
        abstraction_path = Path(self.config.abstraction_file)
        if rebuild or not abstraction_path.exists():
            self.builder.build(self.config)

        blueprint = self.trainer.train(self.config.abstraction_file, config=self.config, algorithm="mccfr")
        blueprint.save(self.config.blueprint_file)

        artifacts: dict[str, PolicyArtifact] = {"blueprint": blueprint}
        artifact_paths: dict[str, str] = {"blueprint": self.config.blueprint_file}

        if run_ablation:
            ablation = self.trainer.train_ablation(
                self.config.abstraction_file,
                config=self.config,
                iterations=self.config.smoke_iterations,
                algorithm=self.config.ablation_algorithm,
            )
            ablation_path = str(Path(self.config.blueprint_file).with_name(f"{Path(self.config.blueprint_file).stem}_{ablation.algorithm}.json"))
            ablation.save(ablation_path)
            artifacts["ablation"] = ablation
            artifact_paths["ablation"] = ablation_path

        if run_fine_tune:
            tuned = self.fine_tuner.train(blueprint, episodes=self.config.fine_tune_hands)
            tuned.save(self.config.tuned_file)
            artifacts["tuned"] = tuned
            artifact_paths["tuned"] = self.config.tuned_file

        evaluations = {
            label: self.evaluate_policy(artifact, hands=hands, seeds=seeds)
            for label, artifact in artifacts.items()
        }

        head_to_head = {}
        if "tuned" in artifacts:
            head_to_head["tuned_vs_blueprint"] = self.evaluate_head_to_head(
                artifacts["tuned"],
                artifacts["blueprint"],
                hands=hands,
                seeds=seeds,
            )
        if "ablation" in artifacts:
            head_to_head["blueprint_vs_ablation"] = self.evaluate_head_to_head(
                artifacts["blueprint"],
                artifacts["ablation"],
                hands=hands,
                seeds=seeds,
            )

        report = {
            "artifacts": artifact_paths,
            "dependency_status": self.trainer.dependency_status(),
            "solver_check": self.trainer.solver_check(iterations=50),
            "evaluations": evaluations,
            "head_to_head": head_to_head,
            "config": self.config.to_dict(),
            "simulation_notes": {
                "deck_sampling": "fresh shuffled deck per hand",
                "button_assignment": "alternates by hand index",
                "metric": "small bets won per 100 hands",
                "opponents": ["random", "loose_passive", "tight_aggressive", "mirror"],
            },
        }
        target = Path(report_path or self.config.experiment_report_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True))
        return report

    def evaluate_policy(self, artifact: PolicyArtifact, hands: int, seeds: int) -> dict[str, Any]:
        runtime = PolicyRuntime(artifact, self.config, self.evaluator.bucketer)
        candidate = PolicyAgent(runtime, name="candidate")
        summaries: dict[str, Any] = {}
        for opponent_name in ("random", "loose_passive", "tight_aggressive", "mirror"):
            values = []
            for seed_index in range(seeds):
                seed = self.config.seed + 1_000 * seed_index
                opponent = self._build_opponent(opponent_name, runtime)
                result = self.evaluator.match(candidate, opponent, hands=hands, seed=seed)
                values.append(result.sb_per_100)
            summaries[opponent_name] = summarize_seed_results(values)
        return summaries

    def evaluate_head_to_head(
        self,
        left: PolicyArtifact,
        right: PolicyArtifact,
        hands: int,
        seeds: int,
    ) -> dict[str, Any]:
        left_agent = PolicyAgent(PolicyRuntime(left, self.config, self.evaluator.bucketer), name="left")
        right_agent = PolicyAgent(PolicyRuntime(right, self.config, self.evaluator.bucketer), name="right")
        values = []
        for seed_index in range(seeds):
            seed = self.config.seed + 5_000 + seed_index
            result = self.evaluator.match(left_agent, right_agent, hands=hands, seed=seed)
            values.append(result.sb_per_100)
        return summarize_seed_results(values)

    @staticmethod
    def _build_opponent(name: str, runtime: PolicyRuntime):
        if name == "random":
            return RandomAgent()
        if name == "loose_passive":
            return LoosePassiveAgent()
        if name == "tight_aggressive":
            return TightAggressiveAgent()
        if name == "mirror":
            return PolicyAgent(runtime, name="mirror")
        raise ValueError(f"Unknown opponent {name}")
