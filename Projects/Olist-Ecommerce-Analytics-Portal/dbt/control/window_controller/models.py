from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PipelineState(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    WAITING_RETRY = "WAITING_RETRY"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class Window:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("window end must be greater than window start")


@dataclass(frozen=True, slots=True)
class Attempt:
    attempt_id: str
    attempt_number: int
    window: Window
    retry_of_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("attempt_id must be non-empty")

        if self.attempt_number < 1:
            raise ValueError("attempt_number must be greater than or equal to 1")


@dataclass(frozen=True, slots=True)
class ControlState:
    pipeline_name: str
    environment: str
    state: PipelineState

    cycle_id: int = 1

    last_successful_window: Window | None = None
    active_attempt: Attempt | None = None

    control_version: int = 0

    last_error_code: str | None = None
    last_error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.pipeline_name.strip():
            raise ValueError("pipeline_name must be non-empty")

        if not self.environment.strip():
            raise ValueError("environment must be non-empty")

        if self.control_version < 0:
            raise ValueError("control_version must be greater than or equal to 0")

        if self.cycle_id < 0:
            raise ValueError(
                "cycle_id must be greater than or equal to 0"
            )

        if self.state == PipelineState.RUNNING and self.active_attempt is None:
            raise ValueError("RUNNING state requires an active attempt")

        if (
            self.state
            in {
                PipelineState.FAILED,
                PipelineState.WAITING_RETRY,
            }
            and self.active_attempt is None
        ):
            raise ValueError(f"{self.state} state requires an active attempt")

        if self.state == PipelineState.IDLE and self.active_attempt is not None:
            raise ValueError("IDLE state cannot have an active attempt")

    @property
    def active_window(self) -> Window | None:
        if self.active_attempt is None:
            return None

        return self.active_attempt.window
