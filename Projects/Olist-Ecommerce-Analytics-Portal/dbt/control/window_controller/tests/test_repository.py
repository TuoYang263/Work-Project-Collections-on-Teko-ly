import unittest
from datetime import datetime, timezone
from google.api_core.exceptions import BadRequest

from window_controller.models import (
    Attempt,
    ControlState,
    PipelineState,
    Window,
)
from window_controller.repository import (
    BigQueryWindowControlRepository,
    ConcurrentStateUpdateError,
    ControlStateAlreadyInitializedError,
    ControlStateIntegrityError,
)


class FakeQueryJob:
    def __init__(
        self,
        rows=None,
        *,
        error=None,
    ):
        self._rows = rows or []
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error

        return self._rows


class FakeBigQueryClient:
    project = "olist-test-project"

    def __init__(
        self,
        rows=None,
        *,
        error=None,
    ):
        self._rows = rows or []
        self._error = error

        self.last_query = None
        self.last_job_config = None

    def query(
        self,
        query,
        *,
        job_config,
    ):
        self.last_query = query
        self.last_job_config = job_config

        return FakeQueryJob(
            self._rows,
            error=self._error,
        )


class TestBigQueryWindowControlRepository(unittest.TestCase):
    def _build_state_pair(self):
        last_successful_window = Window(
            start=datetime(
                2026,
                8,
                8,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                8,
                9,
                tzinfo=timezone.utc,
            ),
        )

        active_window = Window(
            start=datetime(
                2026,
                8,
                9,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                8,
                10,
                tzinfo=timezone.utc,
            ),
        )

        attempt = Attempt(
            attempt_id="attempt-001",
            attempt_number=1,
            window=active_window,
        )

        previous_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.RUNNING,
            last_successful_window=(last_successful_window),
            active_attempt=attempt,
            control_version=7,
        )

        new_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.FAILED,
            last_successful_window=(last_successful_window),
            active_attempt=attempt,
            control_version=8,
            last_error_code="DBT_BUILD_FAILED",
            last_error_message=("dbt build returned non-zero"),
        )

        return previous_state, new_state

    def test_missing_state_returns_none(self):
        client = FakeBigQueryClient([])

        repository = BigQueryWindowControlRepository(client)

        state = repository.load_state(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
        )

        self.assertIsNone(state)

    def test_loads_idle_state(self):
        client = FakeBigQueryClient(
            [
                {
                    "pipeline_name": "olist-dbt-build-job",
                    "environment": "prod",
                    "state": "IDLE",
                    "last_successful_window_start": datetime(
                        2026,
                        8,
                        9,
                        tzinfo=timezone.utc,
                    ),
                    "last_successful_window_end": datetime(
                        2026,
                        8,
                        10,
                        tzinfo=timezone.utc,
                    ),
                    "active_window_start": None,
                    "active_window_end": None,
                    "active_attempt_id": None,
                    "active_attempt_number": None,
                    "active_retry_of_attempt_id": None,
                    "control_version": 7,
                    "last_error_code": None,
                    "last_error_message": None,
                }
            ]
        )

        repository = BigQueryWindowControlRepository(client)

        state = repository.load_state(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
        )

        self.assertIsNotNone(state)
        self.assertEqual(
            state.state,
            PipelineState.IDLE,
        )
        self.assertEqual(state.control_version, 7)
        self.assertIsNone(state.active_attempt)

        self.assertEqual(
            state.last_successful_window.end,
            datetime(
                2026,
                8,
                10,
                tzinfo=timezone.utc,
            ),
        )

    def test_loads_active_attempt(self):
        client = FakeBigQueryClient(
            [
                {
                    "pipeline_name": "olist-dbt-build-job",
                    "environment": "prod",
                    "state": "FAILED",
                    "last_successful_window_start": datetime(
                        2026,
                        8,
                        8,
                        tzinfo=timezone.utc,
                    ),
                    "last_successful_window_end": datetime(
                        2026,
                        8,
                        9,
                        tzinfo=timezone.utc,
                    ),
                    "active_window_start": datetime(
                        2026,
                        8,
                        9,
                        tzinfo=timezone.utc,
                    ),
                    "active_window_end": datetime(
                        2026,
                        8,
                        10,
                        tzinfo=timezone.utc,
                    ),
                    "active_attempt_id": "attempt-002",
                    "active_attempt_number": 2,
                    "active_retry_of_attempt_id": "attempt-001",
                    "control_version": 11,
                    "last_error_code": "DBT_BUILD_FAILED",
                    "last_error_message": "dbt build returned non-zero",
                }
            ]
        )

        repository = BigQueryWindowControlRepository(client)

        state = repository.load_state(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
        )

        self.assertEqual(
            state.state,
            PipelineState.FAILED,
        )

        self.assertEqual(
            state.active_attempt.attempt_id,
            "attempt-002",
        )

        self.assertEqual(
            state.active_attempt.attempt_number,
            2,
        )

        self.assertEqual(
            state.active_attempt.retry_of_attempt_id,
            "attempt-001",
        )

    def test_duplicate_state_rows_are_rejected(self):
        row = {
            "pipeline_name": "olist-dbt-build-job",
            "environment": "prod",
            "state": "IDLE",
            "last_successful_window_start": None,
            "last_successful_window_end": None,
            "active_window_start": None,
            "active_window_end": None,
            "active_attempt_id": None,
            "active_attempt_number": None,
            "active_retry_of_attempt_id": None,
            "control_version": 0,
            "last_error_code": None,
            "last_error_message": None,
        }

        client = FakeBigQueryClient([row, row])

        repository = BigQueryWindowControlRepository(client)

        with self.assertRaises(ControlStateIntegrityError):
            repository.load_state(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
            )

    def test_partial_last_successful_window_is_rejected(self):
        row = {
            "pipeline_name": "olist-dbt-build-job",
            "environment": "prod",
            "state": "IDLE",
            "last_successful_window_start": datetime(2026, 8, 9, tzinfo=timezone.utc),
            "last_successful_window_end": None,
            "active_window_start": None,
            "active_window_end": None,
            "active_attempt_id": None,
            "active_attempt_number": None,
            "active_retry_of_attempt_id": None,
            "control_version": 7,
            "last_error_code": None,
            "last_error_message": None,
        }

        repository = BigQueryWindowControlRepository(FakeBigQueryClient([row]))

        with self.assertRaises(ControlStateIntegrityError):
            repository.load_state(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
            )

    def test_partial_active_attempt_is_rejected(self):
        row = {
            "pipeline_name": "olist-dbt-build-job",
            "environment": "prod",
            "state": "FAILED",
            "last_successful_window_start": None,
            "last_successful_window_end": None,
            "active_window_start": datetime(2026, 8, 9, tzinfo=timezone.utc),
            "active_window_end": datetime(2026, 8, 10, tzinfo=timezone.utc),
            "active_attempt_id": "attempt-001",
            "active_attempt_number": None,
            "active_retry_of_attempt_id": None,
            "control_version": 8,
            "last_error_code": "FAILED",
            "last_error_message": "test",
        }

        repository = BigQueryWindowControlRepository(FakeBigQueryClient([row]))

        with self.assertRaises(ControlStateIntegrityError):
            repository.load_state(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
            )

    def test_persist_transition_is_atomic(self):
        client = FakeBigQueryClient()

        repository = BigQueryWindowControlRepository(client)

        previous_state, new_state = self._build_state_pair()

        repository.persist_transition(
            previous_state=previous_state,
            new_state=new_state,
            event_id="event-001",
            event_type="WINDOW_FAILED",
            metadata={
                "source": "unit-test",
            },
        )

        self.assertIn(
            "BEGIN TRANSACTION",
            client.last_query,
        )

        self.assertIn(
            "@expected_control_version",
            client.last_query,
        )

        self.assertIn(
            "pipeline_window_events",
            client.last_query,
        )

        self.assertIn(
            "COMMIT TRANSACTION",
            client.last_query,
        )

        self.assertIn(
            "ROLLBACK TRANSACTION",
            client.last_query,
        )

    def test_persist_transition_rejects_stale_version(self):
        client = FakeBigQueryClient(error=BadRequest("M10_STALE_CONTROL_VERSION"))

        repository = BigQueryWindowControlRepository(client)

        previous_state, new_state = self._build_state_pair()

        with self.assertRaises(ConcurrentStateUpdateError):
            repository.persist_transition(
                previous_state=previous_state,
                new_state=new_state,
                event_id="event-001",
                event_type="WINDOW_FAILED",
            )

    def test_persist_transition_rejects_invalid_row_count(self):
        client = FakeBigQueryClient(
            error=BadRequest("M10_CONTROL_STATE_INTEGRITY_ERROR")
        )

        repository = BigQueryWindowControlRepository(client)

        previous_state, new_state = self._build_state_pair()

        with self.assertRaises(ControlStateIntegrityError):
            repository.persist_transition(
                previous_state=previous_state,
                new_state=new_state,
                event_id="event-001",
                event_type="WINDOW_FAILED",
            )

    def test_persist_transition_rejects_version_jump(self):
        client = FakeBigQueryClient()

        repository = BigQueryWindowControlRepository(client)

        previous_state, new_state = self._build_state_pair()

        invalid_state = ControlState(
            pipeline_name=new_state.pipeline_name,
            environment=new_state.environment,
            state=new_state.state,
            last_successful_window=(new_state.last_successful_window),
            active_attempt=new_state.active_attempt,
            control_version=9,
            last_error_code=new_state.last_error_code,
            last_error_message=new_state.last_error_message,
        )

        with self.assertRaises(ValueError):
            repository.persist_transition(
                previous_state=previous_state,
                new_state=invalid_state,
                event_id="event-001",
                event_type="WINDOW_FAILED",
            )

    def test_persist_transition_rejects_identity_change(self):
        client = FakeBigQueryClient()

        repository = BigQueryWindowControlRepository(client)

        previous_state, new_state = self._build_state_pair()

        invalid_state = ControlState(
            pipeline_name="different-pipeline",
            environment=new_state.environment,
            state=new_state.state,
            last_successful_window=(new_state.last_successful_window),
            active_attempt=new_state.active_attempt,
            control_version=8,
            last_error_code=new_state.last_error_code,
            last_error_message=new_state.last_error_message,
        )

        with self.assertRaises(ValueError):
            repository.persist_transition(
                previous_state=previous_state,
                new_state=invalid_state,
                event_id="event-001",
                event_type="WINDOW_FAILED",
            )

    def test_completed_transition_uses_previous_attempt(self):
        previous_state, _ = self._build_state_pair()

        completed_state = ControlState(
            pipeline_name=previous_state.pipeline_name,
            environment=previous_state.environment,
            state=PipelineState.IDLE,
            last_successful_window=(previous_state.active_attempt.window),
            active_attempt=None,
            control_version=8,
        )

        client = FakeBigQueryClient()

        repository = BigQueryWindowControlRepository(client)

        repository.persist_transition(
            previous_state=previous_state,
            new_state=completed_state,
            event_id="event-002",
            event_type="WINDOW_SUCCEEDED",
        )

        parameters = {
            parameter.name: parameter.value
            for parameter in client.last_job_config.query_parameters
        }

        self.assertEqual(
            parameters["event_attempt_id"],
            "attempt-001",
        )

        self.assertEqual(
            parameters["event_attempt_number"],
            1,
        )

    def test_initialize_state_creates_idle_version_zero(self):
        client = FakeBigQueryClient()

        repository = BigQueryWindowControlRepository(client)

        state = repository.initialize_state(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
        )

        self.assertEqual(
            state.state,
            PipelineState.IDLE,
        )

        self.assertEqual(
            state.control_version,
            0,
        )

        self.assertIsNone(state.last_successful_window)

        self.assertIsNone(state.active_attempt)

        self.assertIn(
            "BEGIN TRANSACTION",
            client.last_query,
        )

        self.assertIn(
            "INSERT INTO",
            client.last_query,
        )

        self.assertIn(
            "COMMIT TRANSACTION",
            client.last_query,
        )

    def test_initialize_state_rejects_existing_state(self):
        client = FakeBigQueryClient(
            error=BadRequest("M10_CONTROL_STATE_ALREADY_INITIALIZED")
        )

        repository = BigQueryWindowControlRepository(client)

        with self.assertRaises(ControlStateAlreadyInitializedError):
            repository.initialize_state(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
            )

    def test_initialize_state_rejects_duplicate_rows(self):
        client = FakeBigQueryClient(
            error=BadRequest("M10_CONTROL_STATE_INTEGRITY_ERROR")
        )

        repository = BigQueryWindowControlRepository(client)

        with self.assertRaises(ControlStateIntegrityError):
            repository.initialize_state(
                pipeline_name="olist-dbt-build-job",
                environment="prod",
            )


if __name__ == "__main__":
    unittest.main()
