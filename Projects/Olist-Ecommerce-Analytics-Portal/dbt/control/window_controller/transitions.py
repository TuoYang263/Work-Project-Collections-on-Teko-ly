from __future__ import annotations

from .models import PipelineState

_ALLOWED_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.IDLE: {
        PipelineState.RUNNING,
    },
    PipelineState.RUNNING: {
        PipelineState.IDLE,
        PipelineState.FAILED,
    },
    PipelineState.FAILED: {
        PipelineState.WAITING_RETRY,
        PipelineState.QUARANTINED,
    },
    PipelineState.WAITING_RETRY: {
        PipelineState.RUNNING,
        PipelineState.QUARANTINED,
    },
    PipelineState.QUARANTINED: set(),
}


def is_transition_allowed(
    current_state: PipelineState,
    next_state: PipelineState,
) -> bool:
    return next_state in _ALLOWED_TRANSITIONS[current_state]


def validate_transition(
    current_state: PipelineState,
    next_state: PipelineState,
) -> None:
    if not is_transition_allowed(current_state, next_state):
        raise ValueError(
            f"invalid pipeline state transition: " f"{current_state} -> {next_state}"
        )
