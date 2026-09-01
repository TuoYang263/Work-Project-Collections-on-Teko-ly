import unittest

from window_controller.models import PipelineState
from window_controller.transitions import (
    is_transition_allowed,
    validate_transition,
)


class TestStateTransitions(unittest.TestCase):
    def test_idle_can_start_running(self) -> None:
        self.assertTrue(
            is_transition_allowed(
                PipelineState.IDLE,
                PipelineState.RUNNING,
            )
        )

    def test_running_can_finish_successfully(self) -> None:
        self.assertTrue(
            is_transition_allowed(
                PipelineState.RUNNING,
                PipelineState.IDLE,
            )
        )

    def test_running_can_fail(self) -> None:
        self.assertTrue(
            is_transition_allowed(
                PipelineState.RUNNING,
                PipelineState.FAILED,
            )
        )

    def test_failed_can_wait_for_retry(self) -> None:
        self.assertTrue(
            is_transition_allowed(
                PipelineState.FAILED,
                PipelineState.WAITING_RETRY,
            )
        )

    def test_waiting_retry_can_run_again(self) -> None:
        self.assertTrue(
            is_transition_allowed(
                PipelineState.WAITING_RETRY,
                PipelineState.RUNNING,
            )
        )

    def test_failed_cannot_run_directly(self) -> None:
        self.assertFalse(
            is_transition_allowed(
                PipelineState.FAILED,
                PipelineState.RUNNING,
            )
        )

    def test_idle_cannot_wait_for_retry(self) -> None:
        self.assertFalse(
            is_transition_allowed(
                PipelineState.IDLE,
                PipelineState.WAITING_RETRY,
            )
        )

    def test_quarantined_cannot_run_automatically(self) -> None:
        self.assertFalse(
            is_transition_allowed(
                PipelineState.QUARANTINED,
                PipelineState.RUNNING,
            )
        )

    def test_invalid_transition_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_transition(
                PipelineState.FAILED,
                PipelineState.RUNNING,
            )

    def test_valid_transition_does_not_raise(self) -> None:
        validate_transition(
            PipelineState.IDLE,
            PipelineState.RUNNING,
        )


if __name__ == "__main__":
    unittest.main()
