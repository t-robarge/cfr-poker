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


def _make_fine_tuner(config: AbstractHULHEConfig):
    """Factory: pick tabular or NN fine-tuner based on config."""
    mode = getattr(config, "fine_tune_mode", "tabular")
    if mode == "nn":
        from .nn_rl import NNResidualFineTuner
        return NNResidualFineTuner(config)
    return ResidualFineTuner(config)


def summarize_seed_results(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values) if values else 0.0
    ci95 = 0.0
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        ci95 = 1.96 * math.sqrt(variance / len(values))
    lower = mean - ci95
    upper = mean + ci95
    if lower > 0.0:
        verdict = "win"
    elif upper < 0.0:
        verdict = "loss"
    else:
        verdict = "uncertain"
    return {
        "sb_per_100_mean": mean,
        "sb_per_100_ci95": ci95,
        "ci95_interval": [lower, upper],
        "verdict": verdict,
        "seed_results": values,
    }


class ExperimentRunner:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()
        self.builder = AbstractGameBuilder(self.config)
        self.trainer = LiteEFGTrainer(self.config)
        self.fine_tuner = _make_fine_tuner(self.config)
        self.evaluator = Evaluator(self.config)

    def run(
        self,
        rebuild: bool = False,
        eval_hands: int | None = None,
        eval_seeds: int | None = None,
        run_ablation: bool = True,
        run_fine_tune: bool = True,
        reuse_blueprint: bool = False,
        tuned_only: bool = False,
        report_path: str | None = None,
    ) -> dict[str, Any]:
        print("[HULHE] Starting experiment run...")
        hands = eval_hands or self.config.default_eval_hands
        seeds = eval_seeds or self.config.default_eval_seeds
        abstraction_path = Path(self.config.abstraction_file)
        blueprint_path = Path(self.config.blueprint_file)

        needs_abstract_game = (not reuse_blueprint) or run_ablation
        if needs_abstract_game and (rebuild or not abstraction_path.exists()):
            print(f"[BUILD] Creating abstract game at {abstraction_path}...")
            self.builder.build(self.config)
            print(f"[BUILD] Abstract game created.")

        if reuse_blueprint:
            if not blueprint_path.exists():
                raise FileNotFoundError(
                    f"Blueprint artifact not found at {blueprint_path}. Run training first or disable reuse."
                )
            print(f"[TRAIN] Loading blueprint from {blueprint_path}...")
            blueprint = PolicyArtifact.load(blueprint_path)
            print(f"[TRAIN] Blueprint loaded.")
        else:
            print(f"[TRAIN] Training blueprint (MCCFR, {self.config.training_iterations} iterations)...")
            blueprint = self.trainer.train(self.config.abstraction_file, config=self.config, algorithm="mccfr")
            print(f"[TRAIN] Saving blueprint to {self.config.blueprint_file}...")
            blueprint.save(self.config.blueprint_file)
            print(f"[TRAIN] Blueprint trained and saved.")

        artifacts: dict[str, PolicyArtifact] = {"blueprint": blueprint}
        artifact_paths: dict[str, str] = {"blueprint": self.config.blueprint_file}

        if run_ablation:
            print(f"[ABLATION] Training ablation ({self.config.ablation_algorithm}, {self.config.smoke_iterations} iterations)...")
            ablation = self.trainer.train_ablation(
                self.config.abstraction_file,
                config=self.config,
                iterations=self.config.smoke_iterations,
                algorithm=self.config.ablation_algorithm,
            )
            ablation_path = str(Path(self.config.blueprint_file).with_name(f"{Path(self.config.blueprint_file).stem}_{ablation.algorithm}.json"))
            print(f"[ABLATION] Saving ablation to {ablation_path}...")
            ablation.save(ablation_path)
            print(f"[ABLATION] Ablation complete.")
            artifacts["ablation"] = ablation
            artifact_paths["ablation"] = ablation_path

        if run_fine_tune:
            print(f"[FINE-TUNE] Fine-tuning blueprint ({self.config.fine_tune_hands} hands)...")
            tuned = self.fine_tuner.train(blueprint, episodes=self.config.fine_tune_hands)
            print(f"[FINE-TUNE] Saving tuned policy to {self.config.tuned_file}...")
            tuned.save(self.config.tuned_file)
            print(f"[FINE-TUNE] Fine-tuning complete.")
            artifacts["tuned"] = tuned
            artifact_paths["tuned"] = self.config.tuned_file

        print(f"[EVAL] Evaluating {len(artifacts)} policies ({hands} hands × {seeds} seeds each)...")
        evaluations = {}
        for label, artifact in artifacts.items():
            if tuned_only and label != "tuned":
                print(f"  [EVAL] Skipping {label} (--tuned-only)")
                continue
            print(f"  [EVAL] Evaluating {label}...")
            evaluations[label] = self.evaluate_policy(artifact, hands=hands, seeds=seeds)
            print(f"  [EVAL] {label} evaluation complete.")

        head_to_head = {}
        if "tuned" in artifacts and not tuned_only:
            print(f"[H2H] Running head-to-head: tuned vs blueprint...")
            head_to_head["tuned_vs_blueprint"] = self.evaluate_head_to_head(
                artifacts["tuned"],
                artifacts["blueprint"],
                hands=hands,
                seeds=seeds,
            )
            print(f"[H2H] tuned vs blueprint complete.")
        if "ablation" in artifacts and not tuned_only:
            print(f"[H2H] Running head-to-head: blueprint vs ablation...")
            head_to_head["blueprint_vs_ablation"] = self.evaluate_head_to_head(
                artifacts["blueprint"],
                artifacts["ablation"],
                hands=hands,
                seeds=seeds,
            )
            print(f"[H2H] blueprint vs ablation complete.")

        report = {
            "artifacts": artifact_paths,
            "dependency_status": self.trainer.dependency_status(),
            "solver_check": self.trainer.solver_check(iterations=50),
            "evaluations": evaluations,
            "head_to_head": head_to_head,
            "summary": self._build_summary(evaluations, head_to_head),
            "config": self.config.to_dict(),
            "execution": {
                "reuse_blueprint": reuse_blueprint,
                "run_ablation": run_ablation,
                "run_fine_tune": run_fine_tune,
            },
            "simulation_notes": {
                "deck_sampling": "fresh shuffled deck per hand",
                "button_assignment": "alternates by hand index",
                "metric": "small bets won per 100 hands",
                "opponents": ["random", "loose_passive", "tight_aggressive", "mirror"],
            },
        }
        target = Path(report_path or self.config.experiment_report_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[REPORT] Writing final report to {target}...")
        target.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"[COMPLETE] Experiment run finished. Report: {target}")
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
    def _build_summary(
        evaluations: dict[str, dict[str, Any]],
        head_to_head: dict[str, Any],
    ) -> dict[str, Any]:
        matchup_summary = {
            policy_name: {
                opponent_name: metrics["verdict"]
                for opponent_name, metrics in policy_results.items()
            }
            for policy_name, policy_results in evaluations.items()
        }
        head_to_head_summary = {
            matchup_name: metrics["verdict"]
            for matchup_name, metrics in head_to_head.items()
        }
        return {
            "matchups": matchup_summary,
            "head_to_head": head_to_head_summary,
        }

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
