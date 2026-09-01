from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REVIEWER_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REVIEWER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer.review_persistence import (
    ReviewPersistenceError,
    persist_review_snapshot,
    prepare_review_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Persist one deterministic M9 review snapshot to BigQuery.")
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--project-id",
        default=(os.getenv("GCP_PROJECT_ID") or os.getenv("DBT_PROJECT_ID")),
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv(
            "MONITORING_DATASET_ID",
            "olist_monitoring",
        ),
    )
    parser.add_argument(
        "--location",
        default=os.getenv(
            "DBT_LOCATION",
            "EU",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.project_id:
        print(
            "ERROR: project ID is required.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = json.loads(args.review_json.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            raise ReviewPersistenceError("Review JSON root must be an object.")

        snapshot = prepare_review_snapshot(payload)

        persist_review_snapshot(
            snapshot,
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            location=args.location,
        )
    except (
        OSError,
        json.JSONDecodeError,
        ReviewPersistenceError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 3

    print(f"review_id={snapshot.review_id}")
    print(f"monitoring_run_id={snapshot.monitoring_run_id}")
    print(f"total_evaluations={snapshot.total_evaluations}")
    print(f"pass_count={snapshot.pass_count}")
    print(f"triggered_count={snapshot.triggered_count}")
    print("not_evaluated_count=" f"{snapshot.not_evaluated_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
