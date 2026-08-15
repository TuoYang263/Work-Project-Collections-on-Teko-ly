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


def start_new_window(
    control_state: ControlState,
    *,
    window: Window,
    attempt_id: str,
) -> ControlState:
    validate_transition(
        control_state.state,
        PipelineState.RUNNING,
    )

    if control_state.last_successful_window is not None:
        expected_start = control_state.last_successful_window.end

        if window.start != expected_start:
            raise ValueError("new window must start at the last successful window end")

    attempt = Attempt(
        attempt_id=attempt_id,
        attempt_number=1,
        window=window,
    )

    return replace(
        control_state,
        state=PipelineState.RUNNING,
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
