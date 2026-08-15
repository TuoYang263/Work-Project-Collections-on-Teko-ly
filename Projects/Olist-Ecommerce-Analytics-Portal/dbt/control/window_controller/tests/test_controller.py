import unittest
from datetime import datetime, timedelta, timezone

from window_controller.controller import (
    ControlStateNotInitializedError,
    claim_new_window,
    claim_retry_window,
    execute_new_window,
    execute_retry_window,
)
from window_controller.models import (
    Attempt,
    ControlState,
    PipelineState,
    Window,
)


class FakeRepository:
    def __init__(self, state):
        self.state = state
        self.persist_calls = []

    def load_state(
        self,
        *,
        pipeline_name,
        environment,
    ):
        return self.state

    def persist_transition(
        self,
        *,
        previous_state,
        new_state,
        event_id,
        event_type,
        metadata=None,
    ):
        self.persist_calls.append(
            {
                "previous_state": previous_state,
                "new_state": new_state,
                "event_id": event_id,
                "event_type": event_type,
                "metadata": metadata,
            }
        )


class TestWindowController(unittest.TestCase):
    def _previous_window(self):
        return Window(
            start=datetime(
                2026,
                8,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                8,
                2,
                tzinfo=timezone.utc,
            ),
        )

    def _failed_state(self):
        previous_window = self._previous_window()
        failed_window = Window(
            start=previous_window.end,
            end=previous_window.end + timedelta(days=1),
        )

        return ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.FAILED,
            last_successful_window=previous_window,
            active_attempt=Attempt(
                attempt_id="attempt-001",
                attempt_number=1,
                window=failed_window,
            ),
            control_version=2,
            last_error_code="WORKLOAD_FAILED",
            last_error_message="simulated failure",
        )

    def test_claim_new_window_from_initialized_idle_state(self):
        initial_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            control_version=0,
        )

        repository = FakeRepository(initial_state)

        initial_start = datetime(
            2026,
            8,
            1,
            tzinfo=timezone.utc,
        )

        running_state = claim_new_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            initial_start=initial_start,
            window_size=timedelta(days=1),
            attempt_id="attempt-001",
            event_id="event-001",
        )

        self.assertEqual(
            running_state.state,
            PipelineState.RUNNING,
        )
        self.assertEqual(
            running_state.control_version,
            1,
        )
        self.assertEqual(
            running_state.active_attempt.attempt_id,
            "attempt-001",
        )
        self.assertEqual(
            running_state.active_attempt.window.start,
            initial_start,
        )
        self.assertEqual(
            running_state.active_attempt.window.end,
            initial_start + timedelta(days=1),
        )
        self.assertEqual(
            repository.persist_calls[0]["event_type"],
            "WINDOW_STARTED",
        )

    def test_claim_requires_explicit_bootstrap(self):
        repository = FakeRepository(None)

        with self.assertRaises(ControlStateNotInitializedError):
            claim_new_window(
                repository,
                pipeline_name="olist-dbt-build-job",
                environment="prod",
                initial_start=datetime(
                    2026,
                    8,
                    1,
                    tzinfo=timezone.utc,
                ),
                window_size=timedelta(days=1),
                attempt_id="attempt-001",
                event_id="event-001",
            )

    def test_execute_new_window_success_advances_watermark(self):
        initial_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            control_version=0,
        )

        repository = FakeRepository(initial_state)

        captured_env = {}

        def successful_runner(env):
            captured_env.update(env)
            return 0

        initial_start = datetime(
            2026,
            8,
            1,
            tzinfo=timezone.utc,
        )

        final_state = execute_new_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            initial_start=initial_start,
            window_size=timedelta(days=1),
            attempt_id="attempt-001",
            started_event_id="event-start-001",
            final_event_id="event-success-001",
            workload_runner=successful_runner,
        )

        self.assertEqual(
            final_state.state,
            PipelineState.IDLE,
        )
        self.assertEqual(
            final_state.control_version,
            2,
        )
        self.assertEqual(
            final_state.last_successful_window.start,
            initial_start,
        )
        self.assertEqual(
            final_state.last_successful_window.end,
            initial_start + timedelta(days=1),
        )
        self.assertIsNone(final_state.active_attempt)
        self.assertEqual(
            captured_env["CONTROL_ATTEMPT_ID"],
            "attempt-001",
        )
        self.assertEqual(
            len(repository.persist_calls),
            2,
        )
        self.assertEqual(
            repository.persist_calls[0]["event_type"],
            "WINDOW_STARTED",
        )
        self.assertEqual(
            repository.persist_calls[1]["event_type"],
            "WINDOW_SUCCEEDED",
        )

    def test_execute_new_window_failure_keeps_watermark(self):
        initial_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            control_version=0,
        )

        repository = FakeRepository(initial_state)

        def failing_runner(env):
            return 17

        final_state = execute_new_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            initial_start=datetime(
                2026,
                8,
                1,
                tzinfo=timezone.utc,
            ),
            window_size=timedelta(days=1),
            attempt_id="attempt-001",
            started_event_id="event-start-001",
            final_event_id="event-failed-001",
            workload_runner=failing_runner,
        )

        self.assertEqual(
            final_state.state,
            PipelineState.FAILED,
        )
        self.assertEqual(
            final_state.control_version,
            2,
        )
        self.assertIsNone(final_state.last_successful_window)
        self.assertIsNotNone(final_state.active_attempt)
        self.assertEqual(
            final_state.active_attempt.attempt_id,
            "attempt-001",
        )
        self.assertEqual(
            final_state.last_error_code,
            "WORKLOAD_FAILED",
        )
        self.assertEqual(
            len(repository.persist_calls),
            2,
        )
        self.assertEqual(
            repository.persist_calls[1]["event_type"],
            "WINDOW_FAILED",
        )

    def test_claim_retry_from_failed_reuses_window_and_links_attempt(self):
        failed_state = self._failed_state()
        repository = FakeRepository(failed_state)

        running_state = claim_retry_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            attempt_id="attempt-002",
            waiting_event_id="event-waiting-001",
            started_event_id="event-retry-started-001",
        )

        self.assertEqual(
            running_state.state,
            PipelineState.RUNNING,
        )
        self.assertEqual(
            running_state.control_version,
            4,
        )
        self.assertEqual(
            running_state.active_attempt.window,
            failed_state.active_attempt.window,
        )
        self.assertEqual(
            running_state.active_attempt.attempt_number,
            2,
        )
        self.assertEqual(
            running_state.active_attempt.retry_of_attempt_id,
            "attempt-001",
        )
        self.assertIsNone(running_state.last_error_code)
        self.assertIsNone(running_state.last_error_message)
        self.assertEqual(
            [call["event_type"] for call in repository.persist_calls],
            [
                "WINDOW_RETRY_SCHEDULED",
                "WINDOW_RETRY_STARTED",
            ],
        )

    def test_claim_retry_resumes_from_waiting_retry(self):
        failed_state = self._failed_state()
        waiting_state = ControlState(
            pipeline_name=failed_state.pipeline_name,
            environment=failed_state.environment,
            state=PipelineState.WAITING_RETRY,
            last_successful_window=(failed_state.last_successful_window),
            active_attempt=failed_state.active_attempt,
            control_version=3,
            last_error_code=failed_state.last_error_code,
            last_error_message=(failed_state.last_error_message),
        )

        repository = FakeRepository(waiting_state)

        running_state = claim_retry_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            attempt_id="attempt-002",
            waiting_event_id="unused-event-waiting",
            started_event_id="event-retry-started-001",
        )

        self.assertEqual(
            running_state.state,
            PipelineState.RUNNING,
        )
        self.assertEqual(
            running_state.control_version,
            4,
        )
        self.assertEqual(
            len(repository.persist_calls),
            1,
        )
        self.assertEqual(
            repository.persist_calls[0]["event_type"],
            "WINDOW_RETRY_STARTED",
        )

    def test_execute_retry_success_advances_failed_window(self):
        failed_state = self._failed_state()
        repository = FakeRepository(failed_state)
        captured_env = {}

        def successful_runner(env):
            captured_env.update(env)
            return 0

        final_state = execute_retry_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            attempt_id="attempt-002",
            waiting_event_id="event-waiting-001",
            started_event_id="event-retry-started-001",
            final_event_id="event-success-001",
            workload_runner=successful_runner,
        )

        self.assertEqual(
            final_state.state,
            PipelineState.IDLE,
        )
        self.assertEqual(
            final_state.control_version,
            5,
        )
        self.assertEqual(
            final_state.last_successful_window,
            failed_state.active_attempt.window,
        )
        self.assertIsNone(final_state.active_attempt)
        self.assertEqual(
            captured_env["CONTROL_ATTEMPT_ID"],
            "attempt-002",
        )
        self.assertEqual(
            captured_env["CONTROL_WINDOW_START"],
            failed_state.active_attempt.window.start.isoformat(),
        )
        self.assertEqual(
            [call["event_type"] for call in repository.persist_calls],
            [
                "WINDOW_RETRY_SCHEDULED",
                "WINDOW_RETRY_STARTED",
                "WINDOW_SUCCEEDED",
            ],
        )

    def test_execute_retry_failure_keeps_previous_watermark(self):
        failed_state = self._failed_state()
        repository = FakeRepository(failed_state)

        def failing_runner(env):
            return 23

        final_state = execute_retry_window(
            repository,
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            attempt_id="attempt-002",
            waiting_event_id="event-waiting-001",
            started_event_id="event-retry-started-001",
            final_event_id="event-failed-002",
            workload_runner=failing_runner,
        )

        self.assertEqual(
            final_state.state,
            PipelineState.FAILED,
        )
        self.assertEqual(
            final_state.control_version,
            5,
        )
        self.assertEqual(
            final_state.last_successful_window,
            failed_state.last_successful_window,
        )
        self.assertEqual(
            final_state.active_attempt.attempt_id,
            "attempt-002",
        )
        self.assertEqual(
            final_state.active_attempt.attempt_number,
            2,
        )
        self.assertEqual(
            final_state.active_attempt.retry_of_attempt_id,
            "attempt-001",
        )
        self.assertEqual(
            final_state.last_error_code,
            "WORKLOAD_FAILED",
        )
        self.assertEqual(
            [call["event_type"] for call in repository.persist_calls],
            [
                "WINDOW_RETRY_SCHEDULED",
                "WINDOW_RETRY_STARTED",
                "WINDOW_FAILED",
            ],
        )


if __name__ == "__main__":
    unittest.main()
