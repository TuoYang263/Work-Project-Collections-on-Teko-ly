from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


EvaluationResult = Literal["PASS", "TRIGGERED", "NOT_EVALUATED"]


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    rule_id: str
    result: EvaluationResult
    severity: str | None
    entity_type: str
    entity_id: str | None
    evidence_source: str
    evidence: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)