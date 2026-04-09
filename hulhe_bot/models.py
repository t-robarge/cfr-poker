from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class InfoSetKey:
    street: str
    position: int
    history_id: str
    bucket_id: int

    def encode(self) -> str:
        return f"{self.street}|p{self.position}|{self.history_id}|b{self.bucket_id}"

    @classmethod
    def decode(cls, raw: str) -> "InfoSetKey":
        street, pos, history_id, bucket = raw.split("|")
        return cls(
            street=street,
            position=int(pos[1:]),
            history_id=history_id,
            bucket_id=int(bucket[1:]),
        )


@dataclass(slots=True)
class BucketingResult:
    bucket_id: int
    percentile: float
    canonical_key: str
    feature_bucket: int = 0


@dataclass(slots=True)
class Observation:
    acting_player: int
    street: Street
    hole_cards: tuple[str, str]
    board: tuple[str, ...]
    history_id: str
    legal_actions: tuple[Action, ...]
    to_call: int
    pot: int
    bucket_id: int
    bucket_percentile: float


@dataclass(slots=True)
class MatchResult:
    hands: int
    total_small_bets: float
    sb_per_100: float
    per_hand_results: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyArtifact:
    algorithm: str
    backend: str
    policy_table: dict[str, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    residual_table: dict[str, dict[str, float]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        payload = {
            "algorithm": self.algorithm,
            "backend": self.backend,
            "policy_table": self.policy_table,
            "metadata": self.metadata,
            "residual_table": self.residual_table,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "PolicyArtifact":
        payload = json.loads(Path(path).read_text())
        return cls(
            algorithm=payload["algorithm"],
            backend=payload["backend"],
            policy_table=payload["policy_table"],
            metadata=payload.get("metadata", {}),
            residual_table=payload.get("residual_table", {}),
        )
