from __future__ import annotations

import argparse
import sys

from google.cloud import bigquery

from window_controller.models import PipelineState
from window_controller.repository import (
    BigQueryWindowControlRepository,
    ControlStateAlreadyInitializedError,
    ControlStateIntegrityError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly initialize M10 window-control "
            "state for one pipeline environment."
        )
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
        required=True,
    )

    parser.add_argument(
        "--environment",
        required=True,
    )

    parser.add_argument(
        "--location",
        default="EU",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = bigquery.Client(
        project=args.project_id,
        location=args.location,
    )

    repository = BigQueryWindowControlRepository(
        client,
        dataset_id=args.dataset_id,
    )

    try:
        repository.initialize_state(
            pipeline_name=args.pipeline_name,
            environment=args.environment,
        )

    except ControlStateAlreadyInitializedError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    except ControlStateIntegrityError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 3

    persisted_state = repository.load_state(
        pipeline_name=args.pipeline_name,
        environment=args.environment,
    )

    if persisted_state is None:
        raise RuntimeError(
            "control state initialization completed "
            "but no persisted state could be loaded"
        )

    if (
        persisted_state.state != PipelineState.IDLE
        or persisted_state.control_version != 0
        or persisted_state.active_attempt is not None
    ):
        raise RuntimeError(
            "persisted initial control state does not "
            "match the expected IDLE/version-0 state"
        )

    print("Control state initialized successfully.")
    print(f"pipeline_name={persisted_state.pipeline_name}")
    print(f"environment={persisted_state.environment}")
    print(f"state={persisted_state.state.value}")
    print(f"control_version=" f"{persisted_state.control_version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
