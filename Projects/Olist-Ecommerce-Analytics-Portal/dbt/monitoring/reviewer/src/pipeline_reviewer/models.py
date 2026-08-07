from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

EvaluationResult = Literal[
    "PASS",
    "TRIGGERED",
    "NOT_EVALUATED",
]


def freeze_value(value: Any) -> Any:
    """Recursively convert mutable containers into immutable equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: freeze_value(item) for key, item in value.items()}
        )

    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)

    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)

    return value


def thaw_value(value: Any) -> Any:
    """Convert immutable containers back into JSON-serializable containers."""
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}

    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]

    if isinstance(value, frozenset):
        return [thaw_value(item) for item in value]

    return value


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    rule_id: str
    result: EvaluationResult
    severity: str | None
    entity_type: str
    entity_id: str | None
    evidence_source: str
    evidence: Mapping[str, Any]
    reason: str

    def __post_init__(self) -> None:
        # frozen=True blocks normal assignment, so object.__setattr__ is used
        # once during construction to store an immutable evidence snapshot.
        object.__setattr__(
            self,
            "evidence",
            freeze_value(self.evidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "result": self.result,
            "severity": self.severity,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "evidence_source": self.evidence_source,
            "evidence": thaw_value(self.evidence),
            "reason": self.reason,
        }
