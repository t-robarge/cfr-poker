from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AbstractHULHEConfig:
    seed: int = 7
    small_blind: int = 1
    big_blind: int = 2
    small_bet: int = 2
    big_bet: int = 4
    bet_cap: int = 4
    starting_stack: int = 400
    preflop_buckets: int = 169
    flop_buckets: int = 12
    turn_buckets: int = 12
    river_buckets: int = 10
    flop_rollout_samples: int = 16
    turn_rollout_samples: int = 16
    abstraction_samples: int = 5000
    river_payoff_samples: int = 4000
    training_iterations: int = 200000
    smoke_iterations: int = 5000
    checkpoint_every: int = 10000
    fine_tune_hands: int = 100000
    fine_tune_validation_hands: int = 2000
    fine_tune_eval_interval: int = 5000
    fine_tune_alpha: float = 0.15
    fine_tune_epsilon: float = 0.1
    residual_mix_weight: float = 0.15
    residual_temperature: float = 0.5
    artifact_dir: str = "artifacts"
    abstraction_file: str = "artifacts/hulhe_abstract.game"
    blueprint_file: str = "artifacts/blueprint_policy.json"
    tuned_file: str = "artifacts/tuned_policy.json"
    default_eval_hands: int = 50000
    default_eval_seeds: int = 5
    experiment_report_file: str = "artifacts/experiment_report.json"
    trainer_backend: str = "auto"
    ablation_algorithm: str = "cfr_plus"

    @property
    def reward_unit(self) -> int:
        return self.small_bet

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AbstractHULHEConfig":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str | Path | None) -> "AbstractHULHEConfig":
        if path is None:
            return cls()
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if config_path.suffix.lower() not in {".json"}:
            raise ValueError("Only JSON config files are supported without extra dependencies.")
        return cls.from_dict(json.loads(config_path.read_text()))

    def save(self, path: str | Path) -> None:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
