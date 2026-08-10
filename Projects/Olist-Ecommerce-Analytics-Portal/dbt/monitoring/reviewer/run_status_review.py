from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REVIEWER_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REVIEWER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer import (
    BigQueryEvidenceLoader,
    BigQueryQueryExecutor,
    DeterministicEvaluator,
    StatusReviewService,
    load_rule_catalog,
)
from pipeline_reviewer.ai_explainer import build_explanation_report
from pipeline_reviewer.finding_package import build_finding_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic M9 status rules against BigQuery evidence."
    )
    parser.add_argument(
        "--monitoring-run-id",
        help="Review one explicit monitoring run. Defaults to the latest run.",
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("GCP_PROJECT_ID"),
        help="GCP project containing the monitoring dataset.",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("MONITORING_DATASET_ID", "olist_monitoring"),
    )
    parser.add_argument(
        "--job-name",
        default=os.getenv("MONITORING_JOB_NAME", "olist-dbt-build-job"),
    )
    parser.add_argument(
        "--environment",
        default=os.getenv("MONITORING_ENVIRONMENT", "prod"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REVIEWER_ROOT / "config" / "rule_catalog.yml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.project_id:
        raise SystemExit(
            "GCP project is required. Set GCP_PROJECT_ID or pass --project-id."
        )

    catalog = load_rule_catalog(args.catalog)
    evaluator = DeterministicEvaluator(catalog)
    executor = BigQueryQueryExecutor(project_id=args.project_id)
    loader = BigQueryEvidenceLoader(
        executor=executor,
        project_id=args.project_id,
        dataset_id=args.dataset_id,
    )
    service = StatusReviewService(loader=loader, evaluator=evaluator)

    if args.monitoring_run_id:
        review = service.review_run(args.monitoring_run_id)
    else:
        review = service.review_latest(
            job_name=args.job_name,
            environment=args.environment,
        )

    finding_package = build_finding_package(
        monitoring_run_id=review.monitoring_run_id,
        evaluations=review.evaluations,
    )

    explanation_report = build_explanation_report(
        finding_package=finding_package,
        project_id=args.project_id,
    )

    payload = {
        "monitoring_run_id": review.monitoring_run_id,
        "evaluations": [item.to_dict() for item in review.evaluations],
        "finding_package": finding_package,
        "explanation_report": explanation_report,
    }
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
