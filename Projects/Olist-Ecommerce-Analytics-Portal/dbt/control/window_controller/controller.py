from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .models import (
    ControlState,
    PipelineState,
)
from .repository import (
    BigQueryWindowControlRepository,
)
from .service import (
    complete_current_window,
    derive_next_monthly_window,
    fail_current_window,
    move_to_waiting_retry,
    start_new_window,
    start_retry,
)


class ControlStateNotInitializedError(RuntimeError):
    """Raised when runtime starts before explicit bootstrap."""


WorkloadRunner = Callable[
    [dict[str, str]],
    int,
]


def claim_new_window(
    repository: BigQueryWindowControlRepository,
    *,
    pipeline_name: str,
    environment: str,
    source_start: datetime,
    source_end: datetime,
    attempt_id: str,
    event_id: str,
) -> ControlState:
    current_state = repository.load_state(
        pipeline_name=pipeline_name,
        environment=environment,
    )

    if current_state is None:
        raise ControlStateNotInitializedError(
            "control state is not initialized for "
            f"pipeline_name={pipeline_name!r}, "
            f"environment={environment!r}"
        )

    cycle_id, window = derive_next_monthly_window(
        current_state,
        source_start=source_start,
        source_end=source_end,
    )

    running_state = start_new_window(
        current_state,
        cycle_id=cycle_id,
        window=window,
        attempt_id=attempt_id,
    )

    repository.persist_transition(
        previous_state=current_state,
        new_state=running_state,
        event_id=event_id,
        event_type="WINDOW_STARTED",
        metadata={
            "operation": "claim_new_window",
            "cycle_id": cycle_id,
        },
    )

    return running_state


def claim_retry_window(
    repository: BigQueryWindowControlRepository,
    *,
    pipeline_name: str,
    environment: str,
    attempt_id: str,
    waiting_event_id: str,
    started_event_id: str,
) -> ControlState:
    current_state = repository.load_state(
        pipeline_name=pipeline_name,
        environment=environment,
    )

    if current_state is None:
        raise ControlStateNotInitializedError(
            "control state is not initialized for "
            f"pipeline_name={pipeline_name!r}, "
            f"environment={environment!r}"
        )

    if current_state.state == PipelineState.FAILED:
        waiting_state = move_to_waiting_retry(current_state)

        repository.persist_transition(
            previous_state=current_state,
            new_state=waiting_state,
            event_id=waiting_event_id,
            event_type="WINDOW_RETRY_SCHEDULED",
            metadata={
                "operation": "claim_retry_window",
            },
        )

    elif current_state.state == PipelineState.WAITING_RETRY:
        # Resume safely if the previous runtime stopped after
        # persisting WAITING_RETRY but before starting the retry.
        waiting_state = current_state

    else:
        raise ValueError(
            "retry requires control state FAILED "
            "or WAITING_RETRY; "
            f"current state is {current_state.state.value}"
        )

    running_state = start_retry(
        waiting_state,
        attempt_id=attempt_id,
    )

    repository.persist_transition(
        previous_state=waiting_state,
        new_state=running_state,
        event_id=started_event_id,
        event_type="WINDOW_RETRY_STARTED",
        metadata={
            "operation": "claim_retry_window",
        },
    )

    return running_state


def execute_new_window(
    repository: BigQueryWindowControlRepository,
    *,
    pipeline_name: str,
    environment: str,
    source_start: datetime,
    source_end: datetime,
    attempt_id: str,
    started_event_id: str,
    final_event_id: str,
    workload_runner: WorkloadRunner | None = None,
) -> ControlState:
    running_state = claim_new_window(
        repository,
        pipeline_name=pipeline_name,
        environment=environment,
        source_start=source_start,
        source_end=source_end,
        attempt_id=attempt_id,
        event_id=started_event_id,
    )

    return _execute_running_window(
        repository,
        running_state=running_state,
        environment=environment,
        final_event_id=final_event_id,
        workload_runner=workload_runner,
    )


def execute_retry_window(
    repository: BigQueryWindowControlRepository,
    *,
    pipeline_name: str,
    environment: str,
    attempt_id: str,
    waiting_event_id: str,
    started_event_id: str,
    final_event_id: str,
    workload_runner: WorkloadRunner | None = None,
) -> ControlState:
    running_state = claim_retry_window(
        repository,
        pipeline_name=pipeline_name,
        environment=environment,
        attempt_id=attempt_id,
        waiting_event_id=waiting_event_id,
        started_event_id=started_event_id,
    )

    return _execute_running_window(
        repository,
        running_state=running_state,
        environment=environment,
        final_event_id=final_event_id,
        workload_runner=workload_runner,
    )


def _execute_running_window(
    repository: BigQueryWindowControlRepository,
    *,
    running_state: ControlState,
    environment: str,
    final_event_id: str,
    workload_runner: WorkloadRunner | None = None,
) -> ControlState:
    attempt = running_state.active_attempt

    if attempt is None:
        raise RuntimeError("RUNNING state has no active attempt")

    execution_env = os.environ.copy()

    execution_env["CONTROL_ATTEMPT_ID"] = attempt.attempt_id

    execution_env["CONTROL_WINDOW_START"] = attempt.window.start.isoformat()

    execution_env["CONTROL_WINDOW_END"] = attempt.window.end.isoformat()

    execution_env["MONITORING_ENVIRONMENT"] = environment

    runner = workload_runner if workload_runner is not None else _run_dbt_workload

    try:
        return_code = runner(execution_env)

    except OSError as exc:
        failed_state = fail_current_window(
            running_state,
            error_code="WORKLOAD_START_FAILED",
            error_message=str(exc),
        )

        repository.persist_transition(
            previous_state=running_state,
            new_state=failed_state,
            event_id=final_event_id,
            event_type="WINDOW_FAILED",
            metadata={
                "failure_stage": "workload_start",
            },
        )

        return failed_state

    if return_code == 0:
        completed_state = complete_current_window(running_state)

        repository.persist_transition(
            previous_state=running_state,
            new_state=completed_state,
            event_id=final_event_id,
            event_type="WINDOW_SUCCEEDED",
            metadata={
                "workload_return_code": 0,
            },
        )

        return completed_state

    failed_state = fail_current_window(
        running_state,
        error_code="WORKLOAD_FAILED",
        error_message=("run_dbt_job.sh exited with " f"return code {return_code}"),
    )

    repository.persist_transition(
        previous_state=running_state,
        new_state=failed_state,
        event_id=final_event_id,
        event_type="WINDOW_FAILED",
        metadata={
            "workload_return_code": return_code,
        },
    )

    return failed_state


def _run_dbt_workload(
    execution_env: dict[str, str],
) -> int:
    dbt_dir = Path(__file__).resolve().parents[2]

    script_path = dbt_dir / "run_dbt_job.sh"

    result = subprocess.run(
        [
            "bash",
            str(script_path),
        ],
        cwd=dbt_dir,
        env=execution_env,
        check=False,
    )

    return result.returncode
