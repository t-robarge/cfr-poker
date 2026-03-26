from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import importlib
import os
from pathlib import Path
import random
from typing import Any

from .abstract_game import AbstractGameSpec
from .config import AbstractHULHEConfig
from .models import Action, InfoSetKey, PolicyArtifact


def _normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(values))
        return {key: uniform for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}


class LocalExternalSamplingTrainer:
    def __init__(self, spec: AbstractGameSpec, config: AbstractHULHEConfig, algorithm: str = "mccfr"):
        self.spec = spec
        self.config = config
        self.algorithm = algorithm
        self.nodes = spec.nodes
        self.regrets: dict[str, dict[str, float]] = {}
        self.strategy_sums: dict[str, dict[str, float]] = {}

    def train(self, iterations: int, checkpoint_every: int) -> PolicyArtifact:
        rng = random.Random(self.config.seed)
        checkpoints: list[dict[str, Any]] = []
        for iteration in range(1, iterations + 1):
            weight = float(iteration)
            for traverser in (0, 1):
                bucket_pair = self._sample_distribution(self.spec.initial_bucket_distribution, rng)
                self._traverse(self.spec.root_node, bucket_pair, traverser, rng, weight)
            if self.algorithm == "dcfr" and iteration % 1000 == 0:
                self._apply_dcfr_discount(iteration)
            if checkpoint_every > 0 and iteration % checkpoint_every == 0:
                checkpoints.append(
                    {
                        "iteration": iteration,
                        "infosets": len(self.strategy_sums),
                    }
                )
        policy = {
            key: _normalize_probabilities(actions)
            for key, actions in self.strategy_sums.items()
        }
        return PolicyArtifact(
            algorithm=self.algorithm,
            backend="local_external_sampling",
            policy_table=policy,
            metadata={
                "iterations": iterations,
                "checkpoint_every": checkpoint_every,
                "checkpoints": checkpoints,
                "format_name": self.spec.format_name,
            },
        )

    def _traverse(
        self,
        node_id: str,
        bucket_pair: str,
        traverser: int,
        rng: random.Random,
        weight: float,
    ) -> float:
        node = self.nodes[node_id]
        if node["terminal_type"] == "fold":
            return self._terminal_payoff(node, bucket_pair, traverser)
        if node["terminal_type"] == "showdown":
            return self._terminal_payoff(node, bucket_pair, traverser)

        actor = int(node["current_player"])
        legal = [Action(name) for name in node["legal_actions"]]
        info_key = self._infoset_key(node, bucket_pair, actor)
        strategy = self._current_strategy(info_key, legal)
        self._accumulate_strategy(info_key, strategy, weight)

        if actor == traverser:
            action_values: dict[str, float] = {}
            node_value = 0.0
            for action in legal:
                edge = node["transitions"][action.value]
                next_pair = self._advance_bucket_pair(bucket_pair, edge["chance_stage"], rng)
                value = self._traverse(edge["next_node"], next_pair, traverser, rng, weight)
                action_values[action.value] = value
                node_value += strategy[action.value] * value
            regrets = self.regrets.setdefault(info_key, {action.value: 0.0 for action in legal})
            for action in legal:
                updated = regrets[action.value] + action_values[action.value] - node_value
                if self.algorithm == "cfr_plus":
                    updated = max(0.0, updated)
                regrets[action.value] = updated
            return node_value

        sampled_action = self._sample_action(strategy, rng)
        edge = node["transitions"][sampled_action]
        next_pair = self._advance_bucket_pair(bucket_pair, edge["chance_stage"], rng)
        return self._traverse(edge["next_node"], next_pair, traverser, rng, weight)

    def _terminal_payoff(self, node: dict[str, Any], bucket_pair: str, traverser: int) -> float:
        c0, c1 = node["total_contrib"]
        pot = node["pot"]
        if node["terminal_type"] == "fold":
            folded_player = node["folded_player"]
            p0_share = 0.0 if folded_player == 0 else 1.0
        else:
            p0_share = float(self.spec.river_showdown_share.get(bucket_pair, 0.5))
        p0 = (pot * p0_share - c0) / self.config.reward_unit
        p1 = (pot * (1.0 - p0_share) - c1) / self.config.reward_unit
        return p0 if traverser == 0 else p1

    def _infoset_key(self, node: dict[str, Any], bucket_pair: str, actor: int) -> str:
        left, right = bucket_pair.split("-")
        bucket_id = int(left if actor == 0 else right)
        return InfoSetKey(
            street=node["street"],
            position=actor,
            history_id=node["history_id"],
            bucket_id=bucket_id,
        ).encode()

    def _current_strategy(self, info_key: str, legal: list[Action]) -> dict[str, float]:
        regrets = self.regrets.get(info_key)
        if regrets is None:
            return {action.value: 1.0 / len(legal) for action in legal}
        positives = {action.value: max(0.0, regrets.get(action.value, 0.0)) for action in legal}
        total = sum(positives.values())
        if total <= 0:
            return {action.value: 1.0 / len(legal) for action in legal}
        return {action.value: positives[action.value] / total for action in legal}

    def _accumulate_strategy(self, info_key: str, strategy: dict[str, float], weight: float) -> None:
        strategy_sum = self.strategy_sums.setdefault(info_key, {})
        for action, probability in strategy.items():
            strategy_sum[action] = strategy_sum.get(action, 0.0) + weight * probability

    def _sample_action(self, strategy: dict[str, float], rng: random.Random) -> str:
        threshold = rng.random()
        cumulative = 0.0
        for action, probability in strategy.items():
            cumulative += probability
            if threshold <= cumulative:
                return action
        return next(iter(strategy))

    def _sample_distribution(self, distribution: dict[str, float], rng: random.Random) -> str:
        if not distribution:
            raise ValueError("Cannot sample from an empty distribution.")
        threshold = rng.random()
        cumulative = 0.0
        for key, probability in distribution.items():
            cumulative += probability
            if threshold <= cumulative:
                return key
        return next(iter(distribution))

    def _advance_bucket_pair(self, current_pair: str, chance_stage: str | None, rng: random.Random) -> str:
        if chance_stage is None:
            return current_pair
        transitions = self.spec.street_transitions.get(chance_stage, {})
        destination = transitions.get(current_pair)
        if not destination:
            return current_pair
        return self._sample_distribution(destination, rng)

    def _apply_dcfr_discount(self, iteration: int) -> None:
        alpha = iteration / (iteration + 1.0)
        for table in (self.regrets, self.strategy_sums):
            for actions in table.values():
                for action in list(actions):
                    actions[action] *= alpha


class LiteEFGTrainer:
    def __init__(self, config: AbstractHULHEConfig | None = None):
        self.config = config or AbstractHULHEConfig()

    def dependency_status(self) -> dict[str, Any]:
        return {
            "LiteEFG": importlib.util.find_spec("LiteEFG") is not None,
            "pyspiel": importlib.util.find_spec("pyspiel") is not None,
        }

    def solver_check(self, iterations: int = 200) -> dict[str, Any]:
        status = self.dependency_status()
        if not (status["LiteEFG"] and status["pyspiel"]):
            return {
                "available": False,
                "status": status,
                "message": "LiteEFG/OpenSpiel are not installed in the current environment.",
            }

        import LiteEFG as leg
        import pyspiel

        class KuhnCFR(leg.Graph):
            def __init__(self) -> None:
                super().__init__()
                with leg.backward(is_static=True):
                    ev = leg.const(size=1, val=0.0)
                    self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
                    self.regret = leg.const(self.action_set_size, 0.0)
                with leg.backward():
                    gradient = leg.aggregate(ev, aggregator="sum")
                    gradient.inplace(gradient + self.utility)
                    ev.inplace(leg.dot(gradient, self.strategy))
                    self.regret.inplace(self.regret + gradient - ev)
                    self.strategy.inplace(leg.normalize(self.regret, p_norm=1.0, ignore_negative=True))

            def update_graph(self, env: Any) -> None:
                env.update(self.strategy)

            def current_strategy(self) -> Any:
                return self.strategy

        def scalarize(value: Any) -> float:
            if isinstance(value, (list, tuple)):
                if not value:
                    return 0.0
                return float(sum(value) / len(value))
            return float(value)

        scratch_home = Path(self.config.artifact_dir).resolve()
        scratch_home.mkdir(parents=True, exist_ok=True)
        previous_home = os.environ.get("HOME")
        os.environ["HOME"] = str(scratch_home)
        try:
            env = leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker"), traverse_type="External")
            algorithm = KuhnCFR()
            env.set_graph(algorithm)
            initial_exploitability = None
            final_exploitability = None
            for iteration in range(iterations):
                algorithm.update_graph(env)
                env.update_strategy(algorithm.current_strategy())
                if iteration == 0:
                    initial_exploitability = scalarize(
                        env.exploitability(algorithm.current_strategy(), "avg-iterate")
                    )
                final_exploitability = scalarize(
                    env.exploitability(algorithm.current_strategy(), "avg-iterate")
                )
        finally:
            if previous_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = previous_home
        return {
            "available": True,
            "status": status,
            "initial_exploitability": initial_exploitability,
            "final_exploitability": final_exploitability,
            "improved": final_exploitability is not None
            and initial_exploitability is not None
            and final_exploitability <= initial_exploitability,
        }

    def train(
        self,
        game_file_path: str,
        config: AbstractHULHEConfig | None = None,
        iterations: int | None = None,
        algorithm: str = "mccfr",
    ) -> PolicyArtifact:
        cfg = config or self.config
        spec = AbstractGameSpec.load(game_file_path)
        trainer = LocalExternalSamplingTrainer(spec, cfg, algorithm=algorithm)
        artifact = trainer.train(
            iterations=iterations or cfg.training_iterations,
            checkpoint_every=cfg.checkpoint_every,
        )
        artifact.metadata["dependency_status"] = self.dependency_status()
        artifact.metadata["requested_backend"] = cfg.trainer_backend
        return artifact

    def train_ablation(
        self,
        game_file_path: str,
        config: AbstractHULHEConfig | None = None,
        iterations: int | None = None,
        algorithm: str | None = None,
    ) -> PolicyArtifact:
        cfg = config or self.config
        algo = algorithm or cfg.ablation_algorithm
        return self.train(game_file_path, cfg, iterations=iterations or cfg.smoke_iterations, algorithm=algo)
