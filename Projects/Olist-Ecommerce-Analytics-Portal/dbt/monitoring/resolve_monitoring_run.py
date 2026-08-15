from __future__ import annotations

import argparse
import os
import sys

from google.cloud import bigquery

from monitoring_run_resolver import (
    BigQueryMonitoringRunResolver,
    MonitoringRunResolutionError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the exact M8 monitoring_run_id " "for one M10 control attempt."
        )
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
        default=os.getenv("DBT_LOCATION", "EU"),
    )
    parser.add_argument(
        "--control-attempt-id",
        default=os.getenv("CONTROL_ATTEMPT_ID"),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.project_id:
        print("ERROR: project ID is required.", file=sys.stderr)
        return 2

    if args.control_attempt_id is None or not args.control_attempt_id.strip():
        print(
            "ERROR: control attempt ID is required.",
            file=sys.stderr,
        )
        return 2

    client = bigquery.Client(
        project=args.project_id,
        location=args.location,
    )
    resolver = BigQueryMonitoringRunResolver(
        client,
        dataset_id=args.dataset_id,
    )

    try:
        monitoring_run_id = resolver.resolve(
            control_attempt_id=args.control_attempt_id,
        )
    except (MonitoringRunResolutionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(monitoring_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
