from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .abstract_game import AbstractGameBuilder
from .baselines import LoosePassiveAgent, PolicyAgent, RandomAgent, TightAggressiveAgent
from .config import AbstractHULHEConfig
from .evaluator import Evaluator
from .experiments import ExperimentRunner
from .models import PolicyArtifact
from .policy import PolicyRuntime
from .rl import ResidualFineTuner
from .trainer import LiteEFGTrainer


def load_config(path: str | None) -> AbstractHULHEConfig:
    return AbstractHULHEConfig.load(path)


def command_build(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.output:
        config.abstraction_file = args.output
    builder = AbstractGameBuilder(config)
    path = builder.build(config)
    print(json.dumps({"abstraction_file": path}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    game_file = args.game_file or config.abstraction_file
    trainer = LiteEFGTrainer(config)
    artifact = trainer.train(
        game_file,
        config=config,
        iterations=args.iterations or config.training_iterations,
        algorithm=args.algorithm,
    )
    output = args.output or config.blueprint_file
    artifact.save(output)
    payload = {
        "policy_file": output,
        "backend": artifact.backend,
        "algorithm": artifact.algorithm,
        "metadata": artifact.metadata,
    }
    if args.run_ablation:
        ablation = trainer.train_ablation(
            game_file,
            config=config,
            iterations=args.ablation_iterations or config.smoke_iterations,
            algorithm=args.ablation_algorithm or config.ablation_algorithm,
        )
        ablation_path = str(Path(output).with_name(f"{Path(output).stem}_{ablation.algorithm}.json"))
        ablation.save(ablation_path)
        payload["ablation_file"] = ablation_path
    print(json.dumps(payload, indent=2))


def command_fine_tune(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    blueprint = PolicyArtifact.load(args.policy or config.blueprint_file)
    fine_tuner = ResidualFineTuner(config)
    tuned = fine_tuner.train(blueprint, episodes=args.hands or config.fine_tune_hands)
    output = args.output or config.tuned_file
    tuned.save(output)
    print(json.dumps({"policy_file": output, "metadata": tuned.metadata}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    evaluator = Evaluator(config)
    agent = PolicyAgent(PolicyRuntime(PolicyArtifact.load(args.policy), config, evaluator.bucketer), name="candidate")
    opponents = {
        "random": RandomAgent(),
        "loose_passive": LoosePassiveAgent(),
        "tight_aggressive": TightAggressiveAgent(),
        "mirror": PolicyAgent(PolicyRuntime(PolicyArtifact.load(args.policy), config, evaluator.bucketer), name="mirror"),
    }
    if args.opponent not in opponents:
        raise ValueError(f"Unknown opponent {args.opponent}")
    results = []
    for seed_index in range(args.seeds):
        match = evaluator.match(agent, opponents[args.opponent], hands=args.hands, seed=config.seed + seed_index)
        results.append(match.sb_per_100)
    mean = sum(results) / len(results)
    ci95 = 0.0
    if len(results) > 1:
        variance = sum((value - mean) ** 2 for value in results) / (len(results) - 1)
        ci95 = 1.96 * math.sqrt(variance / len(results))
    print(
        json.dumps(
            {
                "opponent": args.opponent,
                "hands_per_seed": args.hands,
                "seeds": args.seeds,
                "sb_per_100_mean": mean,
                "sb_per_100_ci95": ci95,
                "seed_results": results,
            },
            indent=2,
        )
    )


def command_solver_check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    trainer = LiteEFGTrainer(config)
    print(json.dumps(trainer.solver_check(iterations=args.iterations), indent=2))


def command_run_experiments(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    runner = ExperimentRunner(config)
    report = runner.run(
        rebuild=args.rebuild,
        eval_hands=args.hands or config.default_eval_hands,
        eval_seeds=args.seeds or config.default_eval_seeds,
        run_ablation=not args.skip_ablation,
        run_fine_tune=not args.skip_fine_tune,
        reuse_blueprint=args.reuse_blueprint,
        tuned_only=args.tuned_only,
        report_path=args.output or config.experiment_report_file,
    )
    summary: dict[str, Any] = {
        "report_file": args.output or config.experiment_report_file,
        "solver_check": report["solver_check"],
    }
    evals = report["evaluations"]
    # Show the most relevant policy's results
    for label in ("tuned", "blueprint"):
        if label in evals:
            summary[f"{label}_random"] = evals[label]["random"]
            summary[f"{label}_tight_aggressive"] = evals[label]["tight_aggressive"]
            break
    print(json.dumps(summary, indent=2))


def command_play_ui(args: argparse.Namespace) -> None:
    from .ui import launch_ui

    launch_ui(
        config_path=args.config,
        policy_path=args.policy,
        host=args.host,
        port=args.port,
        human_seat=args.human_seat,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HULHE abstraction and training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser("build_abstract_game")
    build_parser_cmd.add_argument("--config")
    build_parser_cmd.add_argument("--output")
    build_parser_cmd.set_defaults(func=command_build)

    train_parser = subparsers.add_parser("train_blueprint")
    train_parser.add_argument("--config")
    train_parser.add_argument("--game-file")
    train_parser.add_argument("--output")
    train_parser.add_argument("--iterations", type=int)
    train_parser.add_argument("--algorithm", default="mccfr", choices=["mccfr", "cfr_plus", "dcfr"])
    train_parser.add_argument("--run-ablation", action="store_true")
    train_parser.add_argument("--ablation-algorithm", choices=["cfr_plus", "dcfr"])
    train_parser.add_argument("--ablation-iterations", type=int)
    train_parser.set_defaults(func=command_train)

    tune_parser = subparsers.add_parser("fine_tune")
    tune_parser.add_argument("--config")
    tune_parser.add_argument("--policy")
    tune_parser.add_argument("--output")
    tune_parser.add_argument("--hands", type=int)
    tune_parser.set_defaults(func=command_fine_tune)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config")
    evaluate_parser.add_argument("--policy", required=True)
    evaluate_parser.add_argument("--opponent", default="random", choices=["random", "loose_passive", "tight_aggressive", "mirror"])
    evaluate_parser.add_argument("--hands", type=int, default=2000)
    evaluate_parser.add_argument("--seeds", type=int, default=1)
    evaluate_parser.set_defaults(func=command_evaluate)

    solver_parser = subparsers.add_parser("solver_check")
    solver_parser.add_argument("--config")
    solver_parser.add_argument("--iterations", type=int, default=200)
    solver_parser.set_defaults(func=command_solver_check)

    experiment_parser = subparsers.add_parser("run_experiments")
    experiment_parser.add_argument("--config")
    experiment_parser.add_argument("--hands", type=int)
    experiment_parser.add_argument("--seeds", type=int)
    experiment_parser.add_argument("--output")
    experiment_parser.add_argument("--rebuild", action="store_true")
    experiment_parser.add_argument("--reuse-blueprint", action="store_true")
    experiment_parser.add_argument("--skip-ablation", action="store_true")
    experiment_parser.add_argument("--skip-fine-tune", action="store_true")
    experiment_parser.add_argument("--tuned-only", action="store_true",
                                   help="Only evaluate the tuned policy (skip blueprint/ablation eval and head-to-head)")
    experiment_parser.set_defaults(func=command_run_experiments)

    play_parser = subparsers.add_parser("play_ui")
    play_parser.add_argument("--config")
    play_parser.add_argument("--policy")
    play_parser.add_argument("--host", default="127.0.0.1")
    play_parser.add_argument("--port", type=int, default=8765)
    play_parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1])
    play_parser.set_defaults(func=command_play_ui)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
