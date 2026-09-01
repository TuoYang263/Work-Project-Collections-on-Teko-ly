import unittest
from datetime import datetime, timedelta, timezone

from window_controller.models import (
    ControlState,
    PipelineState,
    Window,
)
from window_controller.service import (
    complete_current_window,
    derive_next_window,
    fail_current_window,
    move_to_waiting_retry,
    start_new_window,
    start_retry,
)


class TestWindowControllerService(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_window = Window(
            start=datetime(2026, 8, 8, tzinfo=timezone.utc),
            end=datetime(2026, 8, 9, tzinfo=timezone.utc),
        )

        self.next_window = Window(
            start=datetime(2026, 8, 9, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        self.idle_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            last_successful_window=self.previous_window,
        )

    def test_derive_next_window_from_watermark(self) -> None:
        window = derive_next_window(
            self.idle_state,
            initial_start=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            window_size=timedelta(days=1),
        )

        self.assertEqual(window, self.next_window)

    def test_success_advances_watermark(self) -> None:
        running = start_new_window(
            self.idle_state,
            window=self.next_window,
            attempt_id="attempt-001",
        )

        completed = complete_current_window(running)

        self.assertEqual(
            completed.last_successful_window,
            self.next_window,
        )
        self.assertEqual(completed.state, PipelineState.IDLE)
        self.assertIsNone(completed.active_attempt)

    def test_failure_does_not_advance_watermark(self) -> None:
        running = start_new_window(
            self.idle_state,
            window=self.next_window,
            attempt_id="attempt-001",
        )

        failed = fail_current_window(
            running,
            error_code="DBT_BUILD_FAILED",
            error_message="dbt build returned non-zero",
        )

        self.assertEqual(
            failed.last_successful_window,
            self.previous_window,
        )
        self.assertEqual(failed.state, PipelineState.FAILED)

    def test_retry_reuses_same_window(self) -> None:
        running = start_new_window(
            self.idle_state,
            window=self.next_window,
            attempt_id="attempt-001",
        )

        failed = fail_current_window(running)

        waiting = move_to_waiting_retry(failed)

        retrying = start_retry(
            waiting,
            attempt_id="attempt-002",
        )

        self.assertEqual(
            retrying.active_attempt.window,
            self.next_window,
        )
        self.assertEqual(
            retrying.active_attempt.attempt_number,
            2,
        )
        self.assertEqual(
            retrying.active_attempt.retry_of_attempt_id,
            "attempt-001",
        )

    def test_new_window_cannot_skip_watermark(self) -> None:
        skipped_window = Window(
            start=datetime(2026, 8, 10, tzinfo=timezone.utc),
            end=datetime(2026, 8, 11, tzinfo=timezone.utc),
        )

        with self.assertRaises(ValueError):
            start_new_window(
                self.idle_state,
                window=skipped_window,
                attempt_id="attempt-001",
            )


if __name__ == "__main__":
    unittest.main()
