import unittest
from datetime import datetime, timezone

from window_controller.models import (
    Attempt,
    ControlState,
    PipelineState,
    Window,
)


class TestPipelineState(unittest.TestCase):
    def test_expected_states_exist(self) -> None:
        self.assertEqual(PipelineState.IDLE, "IDLE")
        self.assertEqual(PipelineState.RUNNING, "RUNNING")
        self.assertEqual(PipelineState.FAILED, "FAILED")
        self.assertEqual(PipelineState.WAITING_RETRY, "WAITING_RETRY")
        self.assertEqual(PipelineState.QUARANTINED, "QUARANTINED")


class TestWindow(unittest.TestCase):
    def test_valid_window(self) -> None:
        window = Window(
            start=datetime(2026, 8, 9, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        self.assertLess(window.start, window.end)

    def test_rejects_equal_start_and_end(self) -> None:
        timestamp = datetime(2026, 8, 10, tzinfo=timezone.utc)

        with self.assertRaises(ValueError):
            Window(
                start=timestamp,
                end=timestamp,
            )

    def test_rejects_end_before_start(self) -> None:
        with self.assertRaises(ValueError):
            Window(
                start=datetime(2026, 8, 10, tzinfo=timezone.utc),
                end=datetime(2026, 8, 9, tzinfo=timezone.utc),
            )


class TestAttempt(unittest.TestCase):
    def setUp(self) -> None:
        self.window = Window(
            start=datetime(2026, 8, 9, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

    def test_valid_attempt(self) -> None:
        attempt = Attempt(
            attempt_id="attempt-001",
            attempt_number=1,
            window=self.window,
        )

        self.assertEqual(attempt.attempt_number, 1)
        self.assertEqual(attempt.window, self.window)
        self.assertIsNone(attempt.retry_of_attempt_id)

    def test_rejects_empty_attempt_id(self) -> None:
        with self.assertRaises(ValueError):
            Attempt(
                attempt_id="",
                attempt_number=1,
                window=self.window,
            )

    def test_rejects_attempt_number_below_one(self) -> None:
        with self.assertRaises(ValueError):
            Attempt(
                attempt_id="attempt-001",
                attempt_number=0,
                window=self.window,
            )


class TestControlState(unittest.TestCase):
    def setUp(self) -> None:
        self.window = Window(
            start=datetime(2026, 8, 9, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )

        self.attempt = Attempt(
            attempt_id="attempt-001",
            attempt_number=1,
            window=self.window,
        )

    def test_valid_idle_state(self) -> None:
        control_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
        )

        self.assertEqual(control_state.state, PipelineState.IDLE)
        self.assertEqual(control_state.cycle_id, 1)
        self.assertIsNone(control_state.active_attempt)
        self.assertIsNone(control_state.active_window)

    def test_valid_running_state(self) -> None:
        control_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.RUNNING,
            active_attempt=self.attempt,
        )

        self.assertEqual(control_state.active_attempt, self.attempt)
        self.assertEqual(control_state.active_window, self.window)

    def test_running_requires_active_attempt(self) -> None:
        with self.assertRaises(ValueError):
            ControlState(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                state=PipelineState.RUNNING,
            )

    def test_failed_requires_active_attempt(self) -> None:
        with self.assertRaises(ValueError):
            ControlState(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                state=PipelineState.FAILED,
            )

    def test_waiting_retry_requires_active_attempt(self) -> None:
        with self.assertRaises(ValueError):
            ControlState(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                state=PipelineState.WAITING_RETRY,
            )

    def test_idle_rejects_active_attempt(self) -> None:
        with self.assertRaises(ValueError):
            ControlState(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                state=PipelineState.IDLE,
                active_attempt=self.attempt,
            )

    def test_cycle_zero_is_valid_for_legacy_history(self) -> None:
        control_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=0,
        )

        self.assertEqual(control_state.cycle_id, 0)

    def test_rejects_negative_cycle_id(self) -> None:
        with self.assertRaises(ValueError):
            ControlState(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                state=PipelineState.IDLE,
                cycle_id=-1,
            )

    def test_rejects_negative_control_version(self) -> None:
        with self.assertRaises(ValueError):
            ControlState(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                state=PipelineState.IDLE,
                control_version=-1,
            )


if __name__ == "__main__":
    unittest.main()