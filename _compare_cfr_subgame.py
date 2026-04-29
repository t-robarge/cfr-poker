from __future__ import annotations

import argparse
import json

from hulhe_bot.baselines import LoosePassiveAgent, PolicyAgent, RandomAgent, TightAggressiveAgent
from hulhe_bot.config import AbstractHULHEConfig
from hulhe_bot.evaluator import Evaluator
from hulhe_bot.models import PolicyArtifact
from hulhe_bot.policy import PolicyRuntime

CONFIGS = {
    "heuristic": "configs/tag_heavy_045.json",
    "cfr_subgame": "configs/tag_heavy_045_cfr_subgame.json",
}
POLICY_PATH = "artifacts/tag_heavy_045_tuned.json"
DEFAULT_HANDS = 500
DEFAULT_SEEDS = 2
OPPONENTS = ("random", "loose_passive", "tight_aggressive", "mirror")


def make_opponent(name: str, runtime: PolicyRuntime):
    if name == "random":
        return RandomAgent()
    if name == "loose_passive":
        return LoosePassiveAgent()
    if name == "tight_aggressive":
        return TightAggressiveAgent()
    if name == "mirror":
        return PolicyAgent(runtime, name="mirror")
    raise ValueError(name)


def resolve_config_path(raw: str) -> str:
    return CONFIGS.get(raw, raw)


def evaluate(
    config_path: str,
    hands: int = DEFAULT_HANDS,
    seeds: int = DEFAULT_SEEDS,
    cfr_iterations: int | None = None,
    policy_path: str = POLICY_PATH,
) -> dict[str, object]:
    # This is the experiment program referenced in the config patch request.
    config = AbstractHULHEConfig.load(config_path)
    if cfr_iterations is not None:
        config.subgame_cfr_iterations = cfr_iterations
    evaluator = Evaluator(config)
    artifact = PolicyArtifact.load(policy_path)
    runtime = PolicyRuntime(artifact, config, evaluator.bucketer)
    agent = PolicyAgent(runtime, name="candidate")

    results: dict[str, object] = {
        "config": config_path,
        "policy": policy_path,
        "hands_per_seed": hands,
        "seeds": seeds,
        "subgame_mode": getattr(config, "subgame_mode", "heuristic"),
        "subgame_algorithm": getattr(config, "subgame_cfr_algorithm", None),
        "subgame_cfr_iterations": getattr(config, "subgame_cfr_iterations", None),
        "matchups": {},
    }

    for opponent_name in OPPONENTS:
        values: list[float] = []
        for seed_index in range(seeds):
            seed = config.seed + seed_index
            opponent = make_opponent(opponent_name, runtime)
            match = evaluator.match(agent, opponent, hands=hands, seed=seed)
            values.append(match.sb_per_100)
        results["matchups"][opponent_name] = {
            "mean": sum(values) / len(values),
            "seed_results": values,
        }
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare subgame-solving configurations.")
    parser.add_argument(
        "--config",
        default="cfr_subgame",
        help="Config key from CONFIGS or a direct JSON config path.",
    )
    parser.add_argument(
        "--policy",
        default=POLICY_PATH,
        help="Policy artifact path to evaluate.",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=1000,
        help="Hands per opponent per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds per opponent.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[200, 500, 1000],
        help="Subgame CFR iteration settings to compare.",
    )
    return parser



# Example experiment matrix for river CFR subgame solver:
if __name__ == "__main__":
    args = build_parser().parse_args()
    config_path = resolve_config_path(args.config)
    matrix = args.iters
    summary = {}
    for iters in matrix:
        label = f"cfr_subgame_{iters}iters"
        summary[label] = evaluate(
            config_path,
            hands=args.hands,
            seeds=args.seeds,
            cfr_iterations=iters,
            policy_path=args.policy,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
