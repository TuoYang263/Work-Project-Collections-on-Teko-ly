from __future__ import annotations

from dataclasses import dataclass

from .evaluator import DeterministicEvaluator
from .evidence_loader import BigQueryEvidenceLoader
from .models import RuleEvaluation


@dataclass(frozen=True, slots=True)
class StatusReview:
    monitoring_run_id: str
    evaluations: tuple[RuleEvaluation, ...]


class StatusReviewService:
    def __init__(
        self,
        loader: BigQueryEvidenceLoader,
        evaluator: DeterministicEvaluator,
    ) -> None:
        self._loader = loader
        self._evaluator = evaluator

    def review_run(self, monitoring_run_id: str) -> StatusReview:
        bundle = self._loader.load_status_evidence(monitoring_run_id)
        evaluations = self._evaluator.evaluate_status_rules(
            selected_run_id=bundle.monitoring_run_id,
            evidence=bundle.evidence,
        )
        return StatusReview(
            monitoring_run_id=bundle.monitoring_run_id,
            evaluations=tuple(evaluations),
        )

    def review_latest(
        self,
        job_name: str,
        environment: str,
    ) -> StatusReview:
        bundle = self._loader.load_latest_status_evidence(
            job_name=job_name,
            environment=environment,
        )
        evaluations = self._evaluator.evaluate_status_rules(
            selected_run_id=bundle.monitoring_run_id,
            evidence=bundle.evidence,
        )
        return StatusReview(
            monitoring_run_id=bundle.monitoring_run_id,
            evaluations=tuple(evaluations),
        )