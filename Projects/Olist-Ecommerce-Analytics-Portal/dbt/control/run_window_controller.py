from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone

from google.cloud import bigquery

from window_controller.controller import (
    ControlStateNotInitializedError,
    execute_new_window,
    execute_retry_window,
)
from window_controller.models import PipelineState
from window_controller.repository import (
    BigQueryWindowControlRepository,
    ConcurrentStateUpdateError,
    ControlStateIntegrityError,
)


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")

    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include a timezone")

    return parsed.astimezone(timezone.utc)


def build_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run one M10 governed Olist pipeline window.")
    )

    parser.add_argument(
        "--project-id",
        required=True,
    )

    parser.add_argument(
        "--dataset-id",
        default="olist_control",
    )

    parser.add_argument(
        "--pipeline-name",
        default="olist-dbt-build-job",
    )

    parser.add_argument(
        "--environment",
        default="prod",
    )

    parser.add_argument(
        "--dbt-dataset",
        default="olist",
        help=(
            "Base dbt dataset. "
            "For isolated validation runs, use "
            "'olist_validation'."
        ),
    )

    parser.add_argument(
        "--location",
        default="EU",
    )

    parser.add_argument(
        "--source-start",
        type=parse_datetime,
        help=(
            "Inclusive start of the bounded historical "
            "production cycle. Required for new-window "
            "execution."
        ),
    )

    parser.add_argument(
        "--source-end",
        type=parse_datetime,
        help=(
            "Exclusive end of the bounded historical "
            "production cycle. Required for new-window "
            "execution."
        ),
    )

    parser.add_argument(
        "--retry",
        action="store_true",
        help=(
            "Retry the currently FAILED or WAITING_RETRY "
            "window instead of claiming a new window."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.retry:
        if (
            args.source_start is None
            or args.source_end is None
        ):
            print(
                "ERROR: --source-start and --source-end "
                "are required for a new window.",
                file=sys.stderr,
            )
            return 2

    if args.environment != "prod" and args.dbt_dataset == "olist":
        print(
            "ERROR: non-prod environments must use an "
            "isolated --dbt-dataset instead of 'olist'.",
            file=sys.stderr,
        )
        return 2

    # The existing dbt workload requires this environment
    # variable. For Olist, the dbt and control-plane project
    # are currently the same GCP project.
    os.environ["DBT_PROJECT_ID"] = args.project_id

    os.environ["DBT_LOCATION"] = args.location

    os.environ["DBT_TARGET"] = args.environment

    os.environ["DBT_DATASET"] = args.dbt_dataset

    os.environ["MONITORING_JOB_NAME"] = args.pipeline_name

    os.environ["MONITORING_ENVIRONMENT"] = args.environment

    client = bigquery.Client(
        project=args.project_id,
        location=args.location,
    )

    repository = BigQueryWindowControlRepository(
        client,
        dataset_id=args.dataset_id,
    )

    attempt_id = build_id("attempt")
    waiting_event_id = build_id("event-retry-waiting")
    started_event_id = build_id("event-started")
    final_event_id = build_id("event-final")

    print("Starting governed pipeline execution...")
    print(f"pipeline_name={args.pipeline_name}")
    print(f"environment={args.environment}")
    print(f"dbt_dataset={args.dbt_dataset}")
    print(f"attempt_id={attempt_id}")

    try:
        if args.retry:
            print("execution_mode=retry")

            final_state = execute_retry_window(
                repository,
                pipeline_name=args.pipeline_name,
                environment=args.environment,
                attempt_id=attempt_id,
                waiting_event_id=waiting_event_id,
                started_event_id=started_event_id,
                final_event_id=final_event_id,
            )

        else:
            print("execution_mode=new_window")

            # main() has already validated this invariant.
            assert args.source_start is not None
            assert args.source_end is not None

            final_state = execute_new_window(
                repository,
                pipeline_name=args.pipeline_name,
                environment=args.environment,
                source_start=args.source_start,
                source_end=args.source_end,
                attempt_id=attempt_id,
                started_event_id=started_event_id,
                final_event_id=final_event_id,
            )

    except ControlStateNotInitializedError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    except ConcurrentStateUpdateError as exc:
        print(
            f"ERROR: concurrent control-state update: {exc}",
            file=sys.stderr,
        )
        return 3

    except ControlStateIntegrityError as exc:
        print(
            f"ERROR: control-state integrity failure: {exc}",
            file=sys.stderr,
        )
        return 4

    except ValueError as exc:
        print(
            f"ERROR: invalid control transition: {exc}",
            file=sys.stderr,
        )
        return 5

    print(f"final_state={final_state.state.value}")
    print(f"cycle_id={final_state.cycle_id}")
    print(f"control_version={final_state.control_version}")

    if final_state.last_successful_window is not None:
        print(
            "last_successful_window="
            f"{final_state.last_successful_window.start.isoformat()}"
            " -> "
            f"{final_state.last_successful_window.end.isoformat()}"
        )

    if final_state.state == PipelineState.IDLE:
        print("Governed pipeline execution completed successfully.")
        return 0

    if final_state.last_error_code:
        print(
            f"error_code={final_state.last_error_code}",
            file=sys.stderr,
        )

    if final_state.last_error_message:
        print(
            f"error_message={final_state.last_error_message}",
            file=sys.stderr,
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
