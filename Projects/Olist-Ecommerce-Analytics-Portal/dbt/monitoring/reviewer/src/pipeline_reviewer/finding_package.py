from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from .models import RuleEvaluation


def _build_finding_id(
    monitoring_run_id: str,
    evaluation: RuleEvaluation,
) -> str:
    entity_id = evaluation.entity_id or "pipeline"

    return (
        f"{monitoring_run_id}:"
        f"{evaluation.rule_id}:"
        f"{evaluation.entity_type}:"
        f"{entity_id}"
    )


def build_finding_package(
    monitoring_run_id: str,
    evaluations: Sequence[RuleEvaluation],
) -> dict[str, Any]:
    result_counts = Counter(evaluation.result for evaluation in evaluations)

    summary = {
        "total_evaluations": len(evaluations),
        "pass": result_counts["PASS"],
        "triggered": result_counts["TRIGGERED"],
        "not_evaluated": result_counts["NOT_EVALUATED"],
    }

    findings = [
        {
            "finding_id": _build_finding_id(
                monitoring_run_id,
                evaluation,
            ),
            **evaluation.to_dict(),
        }
        for evaluation in evaluations
        if evaluation.result == "TRIGGERED"
    ]

    return {
        "monitoring_run_id": monitoring_run_id,
        "summary": summary,
        "findings": findings,
    }
