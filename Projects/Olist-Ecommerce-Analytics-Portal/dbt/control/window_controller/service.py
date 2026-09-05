from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from .models import (
    Attempt,
    ControlState,
    PipelineState,
    Window,
)
from .transitions import validate_transition


def derive_next_window(
    control_state: ControlState,
    *,
    initial_start: datetime,
    window_size: timedelta,
) -> Window:
    if window_size <= timedelta(0):
        raise ValueError("window_size must be greater than zero")

    if control_state.last_successful_window is None:
        window_start = initial_start
    else:
        window_start = control_state.last_successful_window.end

    return Window(
        start=window_start,
        end=window_start + window_size,
    )

def derive_next_monthly_window(
    control_state: ControlState,
    *,
    source_start: datetime,
    source_end: datetime,
) -> tuple[int, Window]:
    _validate_source_bounds(
        source_start=source_start,
        source_end=source_end,
    )

    if control_state.cycle_id == 0:
        cycle_id = 1
        window_start = source_start

    elif control_state.last_successful_window is None:
        cycle_id = control_state.cycle_id
        window_start = source_start

    else:
        last_window_end = (
            control_state.last_successful_window.end
        )

        if last_window_end < source_end:
            cycle_id = control_state.cycle_id
            window_start = last_window_end

        elif last_window_end == source_end:
            cycle_id = control_state.cycle_id + 1
            window_start = source_start

        else:
            raise ValueError(
                "last successful window extends "
                "beyond configured source end"
            )

    _validate_month_boundary(
        "window_start",
        window_start,
    )

    window_end = _next_calendar_month(
        window_start
    )

    if window_end > source_end:
        raise ValueError(
            "derived calendar-month window "
            "extends beyond configured source end"
        )

    return (
        cycle_id,
        Window(
            start=window_start,
            end=window_end,
        ),
    )

def start_new_window(
    control_state: ControlState,
    *,
    window: Window,
    attempt_id: str,
    cycle_id: int | None = None,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.RUNNING,
    )

    effective_cycle_id = (
        control_state.cycle_id
        if cycle_id is None
        else cycle_id
    )

    if effective_cycle_id < control_state.cycle_id:
        raise ValueError(
            "new window cannot move to an earlier cycle"
        )

    if (
        effective_cycle_id
        > control_state.cycle_id + 1
    ):
        raise ValueError(
            "new window cannot skip processing cycles"
        )

    if (
        effective_cycle_id == control_state.cycle_id
        and control_state.last_successful_window
        is not None
    ):
        expected_start = (
            control_state.last_successful_window.end
        )

        if window.start != expected_start:
            raise ValueError(
                "new window must start at the "
                "last successful window end"
            )

    attempt = Attempt(
        attempt_id=attempt_id,
        attempt_number=1,
        window=window,
    )

    return replace(
        control_state,
        state=PipelineState.RUNNING,
        cycle_id=effective_cycle_id,
        active_attempt=attempt,
        control_version=control_state.control_version + 1,
        last_error_code=None,
        last_error_message=None,
    )


def complete_current_window(
    control_state: ControlState,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.IDLE,
    )

    attempt = _require_active_attempt(control_state)

    return replace(
        control_state,
        state=PipelineState.IDLE,
        last_successful_window=attempt.window,
        active_attempt=None,
        control_version=control_state.control_version + 1,
        last_error_code=None,
        last_error_message=None,
    )


def fail_current_window(
    control_state: ControlState,
    *,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.FAILED,
    )

    _require_active_attempt(control_state)

    return replace(
        control_state,
        state=PipelineState.FAILED,
        control_version=control_state.control_version + 1,
        last_error_code=error_code,
        last_error_message=error_message,
    )


def move_to_waiting_retry(
    control_state: ControlState,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.WAITING_RETRY,
    )

    _require_active_attempt(control_state)

    return replace(
        control_state,
        state=PipelineState.WAITING_RETRY,
        control_version=control_state.control_version + 1,
    )


def start_retry(
    control_state: ControlState,
    *,
    attempt_id: str,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.RUNNING,
    )

    previous_attempt = _require_active_attempt(control_state)

    retry_attempt = Attempt(
        attempt_id=attempt_id,
        attempt_number=previous_attempt.attempt_number + 1,
        window=previous_attempt.window,
        retry_of_attempt_id=previous_attempt.attempt_id,
    )

    return replace(
        control_state,
        state=PipelineState.RUNNING,
        active_attempt=retry_attempt,
        control_version=control_state.control_version + 1,
        last_error_code=None,
        last_error_message=None,
    )


def quarantine(
    control_state: ControlState,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.QUARANTINED,
    )

    return replace(
        control_state,
        state=PipelineState.QUARANTINED,
        control_version=control_state.control_version + 1,
    )


def _require_active_attempt(
    control_state: ControlState,
) -> Attempt:
    if control_state.active_attempt is None:
        raise ValueError("control state has no active attempt")

    return control_state.active_attempt


def _next_calendar_month(
    value: datetime,
) -> datetime:
    if value.month == 12:
        return value.replace(
            year=value.year + 1,
            month=1,
        )

    return value.replace(
        month=value.month + 1,
    )


def _validate_month_boundary(
    field_name: str,
    value: datetime,
) -> None:
    if value.tzinfo is None:
        raise ValueError(
            f"{field_name} must be timezone-aware"
        )

    if (
        value.day != 1
        or value.hour != 0
        or value.minute != 0
        or value.second != 0
        or value.microsecond != 0
    ):
        raise ValueError(
            f"{field_name} must be a "
            "calendar-month boundary"
        )


def _validate_source_bounds(
    *,
    source_start: datetime,
    source_end: datetime,
) -> None:
    _validate_month_boundary(
        "source_start",
        source_start,
    )

    _validate_month_boundary(
        "source_end",
        source_end,
    )

    if source_end <= source_start:
        raise ValueError(
            "source_end must be greater than source_start"
        )
