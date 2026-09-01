from __future__ import annotations

import json
from typing import Any

from google.api_core.exceptions import BadRequest
from google.cloud import bigquery

from .models import (
    Attempt,
    ControlState,
    PipelineState,
    Window,
)


class ControlStateIntegrityError(RuntimeError):
    """Raised when persisted control state violates repository invariants."""


class ConcurrentStateUpdateError(RuntimeError):
    """Raised when a stale control-state version tries to write."""


class ControlStateAlreadyInitializedError(RuntimeError):
    """Raised when control state has already been initialized."""


class BigQueryWindowControlRepository:
    def __init__(
        self,
        client: bigquery.Client,
        *,
        dataset_id: str = "olist_control",
        state_table_id: str = "pipeline_control_state",
        event_table_id: str = "pipeline_window_events",
    ) -> None:
        self._client = client

        self._state_table_fqn = f"{client.project}.{dataset_id}.{state_table_id}"

        self._event_table_fqn = f"{client.project}.{dataset_id}.{event_table_id}"

    def load_state(
        self,
        *,
        pipeline_name: str,
        environment: str,
    ) -> ControlState | None:
        query = f"""
        SELECT
            pipeline_name,
            environment,
            state,
            last_successful_window_start,
            last_successful_window_end,
            active_window_start,
            active_window_end,
            active_attempt_id,
            active_attempt_number,
            active_retry_of_attempt_id,
            control_version,
            last_error_code,
            last_error_message
        FROM `{self._state_table_fqn}`
        WHERE pipeline_name = @pipeline_name
          AND environment = @environment
        LIMIT 2
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "pipeline_name",
                    "STRING",
                    pipeline_name,
                ),
                bigquery.ScalarQueryParameter(
                    "environment",
                    "STRING",
                    environment,
                ),
            ]
        )

        rows = list(
            self._client.query(
                query,
                job_config=job_config,
            ).result()
        )

        if not rows:
            return None

        if len(rows) > 1:
            raise ControlStateIntegrityError(
                "multiple control-state rows found for "
                f"pipeline_name={pipeline_name!r}, "
                f"environment={environment!r}"
            )

        return self._row_to_control_state(rows[0])

    def initialize_state(
        self,
        *,
        pipeline_name: str,
        environment: str,
    ) -> ControlState:
        initial_state = ControlState(
            pipeline_name=pipeline_name,
            environment=environment,
            state=PipelineState.IDLE,
            last_successful_window=None,
            active_attempt=None,
            control_version=0,
            last_error_code=None,
            last_error_message=None,
        )

        query = f"""
        DECLARE existing_rows INT64 DEFAULT 0;

        BEGIN

        BEGIN TRANSACTION;

        SET existing_rows = (
            SELECT COUNT(*)
            FROM `{self._state_table_fqn}`
            WHERE pipeline_name = @pipeline_name
            AND environment = @environment
        );

        IF existing_rows = 1 THEN
            RAISE USING MESSAGE =
                'M10_CONTROL_STATE_ALREADY_INITIALIZED';
        END IF;

        IF existing_rows > 1 THEN
            RAISE USING MESSAGE =
                'M10_CONTROL_STATE_INTEGRITY_ERROR';
        END IF;

        INSERT INTO `{self._state_table_fqn}`
        (
            pipeline_name,
            environment,
            state,

            last_successful_window_start,
            last_successful_window_end,

            active_window_start,
            active_window_end,
            active_attempt_id,
            active_attempt_number,
            active_retry_of_attempt_id,

            control_version,

            last_error_code,
            last_error_message,

            updated_at
        )
        VALUES
        (
            @pipeline_name,
            @environment,
            'IDLE',

            NULL,
            NULL,

            NULL,
            NULL,
            NULL,
            NULL,
            NULL,

            0,

            NULL,
            NULL,

            CURRENT_TIMESTAMP()
        );

        COMMIT TRANSACTION;

        EXCEPTION WHEN ERROR THEN

        ROLLBACK TRANSACTION;
        RAISE;

        END;
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "pipeline_name",
                    "STRING",
                    pipeline_name,
                ),
                bigquery.ScalarQueryParameter(
                    "environment",
                    "STRING",
                    environment,
                ),
            ]
        )

        try:
            self._client.query(
                query,
                job_config=job_config,
            ).result()

        except BadRequest as exc:
            message = str(exc)

            if "M10_CONTROL_STATE_ALREADY_INITIALIZED" in message:
                raise ControlStateAlreadyInitializedError(
                    "control state has already been "
                    "initialized for "
                    f"pipeline_name={pipeline_name!r}, "
                    f"environment={environment!r}"
                ) from exc

            if "M10_CONTROL_STATE_INTEGRITY_ERROR" in message:
                raise ControlStateIntegrityError(
                    "multiple control-state rows exist " "during initialization"
                ) from exc

            raise

        return initial_state

    def persist_transition(
        self,
        *,
        previous_state: ControlState,
        new_state: ControlState,
        event_id: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._validate_state_update(
            previous_state=previous_state,
            new_state=new_state,
        )

        if not event_id.strip():
            raise ValueError("event_id must be non-empty")

        if not event_type.strip():
            raise ValueError("event_type must be non-empty")

        event_attempt = self._select_event_attempt(
            previous_state=previous_state,
            new_state=new_state,
        )

        last_successful_window = new_state.last_successful_window

        active_attempt = new_state.active_attempt

        metadata_json = (
            json.dumps(
                metadata,
                ensure_ascii=False,
            )
            if metadata is not None
            else None
        )

        query = f"""
        DECLARE affected_rows INT64 DEFAULT 0;

        BEGIN

        BEGIN TRANSACTION;

        UPDATE `{self._state_table_fqn}`
        SET
            state = @state,

            last_successful_window_start =
                @last_successful_window_start,
            last_successful_window_end =
                @last_successful_window_end,

            active_window_start =
                @active_window_start,
            active_window_end =
                @active_window_end,

            active_attempt_id =
                @active_attempt_id,
            active_attempt_number =
                @active_attempt_number,
            active_retry_of_attempt_id =
                @active_retry_of_attempt_id,

            control_version =
                @new_control_version,

            last_error_code =
                @last_error_code,
            last_error_message =
                @last_error_message,

            updated_at = CURRENT_TIMESTAMP()

        WHERE pipeline_name = @pipeline_name
            AND environment = @environment
            AND control_version =
                @expected_control_version;

        SET affected_rows = @@row_count;

        IF affected_rows = 0 THEN
            RAISE USING MESSAGE =
                'M10_STALE_CONTROL_VERSION';
        END IF;

        IF affected_rows != 1 THEN
            RAISE USING MESSAGE =
                'M10_CONTROL_STATE_INTEGRITY_ERROR';
        END IF;

        INSERT INTO `{self._event_table_fqn}`
        (
            event_id,
            attempt_id,

            pipeline_name,
            environment,

            window_start,
            window_end,

            attempt_number,

            from_state,
            to_state,

            from_control_version,
            to_control_version,

            event_type,
            event_time,

            retry_of_attempt_id,

            error_code,
            error_message,

            metadata_json
        )
        VALUES
        (
            @event_id,
            @event_attempt_id,

            @pipeline_name,
            @environment,

            @event_window_start,
            @event_window_end,

            @event_attempt_number,

            @from_state,
            @to_state,

            @expected_control_version,
            @new_control_version,

            @event_type,
            CURRENT_TIMESTAMP(),

            @retry_of_attempt_id,

            @last_error_code,
            @last_error_message,

            CASE
                WHEN @metadata_json IS NULL
                THEN NULL
                ELSE PARSE_JSON(@metadata_json)
            END
        );

        COMMIT TRANSACTION;

        EXCEPTION WHEN ERROR THEN

        ROLLBACK TRANSACTION;
        RAISE;

        END;
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "pipeline_name",
                    "STRING",
                    new_state.pipeline_name,
                ),
                bigquery.ScalarQueryParameter(
                    "environment",
                    "STRING",
                    new_state.environment,
                ),
                bigquery.ScalarQueryParameter(
                    "state",
                    "STRING",
                    new_state.state.value,
                ),
                bigquery.ScalarQueryParameter(
                    "last_successful_window_start",
                    "TIMESTAMP",
                    (last_successful_window.start if last_successful_window else None),
                ),
                bigquery.ScalarQueryParameter(
                    "last_successful_window_end",
                    "TIMESTAMP",
                    (last_successful_window.end if last_successful_window else None),
                ),
                bigquery.ScalarQueryParameter(
                    "active_window_start",
                    "TIMESTAMP",
                    (active_attempt.window.start if active_attempt else None),
                ),
                bigquery.ScalarQueryParameter(
                    "active_window_end",
                    "TIMESTAMP",
                    (active_attempt.window.end if active_attempt else None),
                ),
                bigquery.ScalarQueryParameter(
                    "active_attempt_id",
                    "STRING",
                    (active_attempt.attempt_id if active_attempt else None),
                ),
                bigquery.ScalarQueryParameter(
                    "active_attempt_number",
                    "INT64",
                    (active_attempt.attempt_number if active_attempt else None),
                ),
                bigquery.ScalarQueryParameter(
                    "active_retry_of_attempt_id",
                    "STRING",
                    (active_attempt.retry_of_attempt_id if active_attempt else None),
                ),
                bigquery.ScalarQueryParameter(
                    "new_control_version",
                    "INT64",
                    new_state.control_version,
                ),
                bigquery.ScalarQueryParameter(
                    "expected_control_version",
                    "INT64",
                    previous_state.control_version,
                ),
                bigquery.ScalarQueryParameter(
                    "last_error_code",
                    "STRING",
                    new_state.last_error_code,
                ),
                bigquery.ScalarQueryParameter(
                    "last_error_message",
                    "STRING",
                    new_state.last_error_message,
                ),
                # Audit event
                bigquery.ScalarQueryParameter(
                    "event_id",
                    "STRING",
                    event_id,
                ),
                bigquery.ScalarQueryParameter(
                    "event_type",
                    "STRING",
                    event_type,
                ),
                bigquery.ScalarQueryParameter(
                    "event_attempt_id",
                    "STRING",
                    event_attempt.attempt_id,
                ),
                bigquery.ScalarQueryParameter(
                    "event_attempt_number",
                    "INT64",
                    event_attempt.attempt_number,
                ),
                bigquery.ScalarQueryParameter(
                    "event_window_start",
                    "TIMESTAMP",
                    event_attempt.window.start,
                ),
                bigquery.ScalarQueryParameter(
                    "event_window_end",
                    "TIMESTAMP",
                    event_attempt.window.end,
                ),
                bigquery.ScalarQueryParameter(
                    "retry_of_attempt_id",
                    "STRING",
                    event_attempt.retry_of_attempt_id,
                ),
                bigquery.ScalarQueryParameter(
                    "from_state",
                    "STRING",
                    previous_state.state.value,
                ),
                bigquery.ScalarQueryParameter(
                    "to_state",
                    "STRING",
                    new_state.state.value,
                ),
                bigquery.ScalarQueryParameter(
                    "metadata_json",
                    "STRING",
                    metadata_json,
                ),
            ]
        )

        try:
            self._client.query(
                query,
                job_config=job_config,
            ).result()

        except BadRequest as exc:
            message = str(exc)

            if "M10_STALE_CONTROL_VERSION" in message:
                raise ConcurrentStateUpdateError(
                    "control state transition rejected "
                    "because the persisted control_version "
                    "is stale"
                ) from exc

            if "M10_CONTROL_STATE_INTEGRITY_ERROR" in message:
                raise ControlStateIntegrityError(
                    "control state transition affected " "an unexpected number of rows"
                ) from exc

            raise

    @staticmethod
    def _row_to_control_state(
        row: Any,
    ) -> ControlState:
        last_successful_window = _build_optional_window(
            start=row["last_successful_window_start"],
            end=row["last_successful_window_end"],
            field_name="last_successful_window",
        )

        active_attempt = _build_optional_attempt(row)

        return ControlState(
            pipeline_name=row["pipeline_name"],
            environment=row["environment"],
            state=PipelineState(row["state"]),
            last_successful_window=last_successful_window,
            active_attempt=active_attempt,
            control_version=row["control_version"],
            last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"],
        )

    @staticmethod
    def _validate_state_update(
        *,
        previous_state: ControlState,
        new_state: ControlState,
    ) -> None:
        if (
            previous_state.pipeline_name != new_state.pipeline_name
            or previous_state.environment != new_state.environment
        ):
            raise ValueError("state update cannot change pipeline identity")

        expected_new_version = previous_state.control_version + 1

        if new_state.control_version != expected_new_version:
            raise ValueError(
                "new state control_version must be exactly "
                "previous control_version + 1"
            )

    @staticmethod
    def _select_event_attempt(
        *,
        previous_state: ControlState,
        new_state: ControlState,
    ) -> Attempt:
        if new_state.active_attempt is not None:
            return new_state.active_attempt

        if previous_state.active_attempt is not None:
            return previous_state.active_attempt

        raise ValueError(
            "state transition requires an attempt " "for audit-event persistence"
        )


def _build_optional_window(
    *,
    start: Any,
    end: Any,
    field_name: str,
) -> Window | None:
    if start is None and end is None:
        return None

    if start is None or end is None:
        raise ControlStateIntegrityError(f"{field_name} requires both start and end")

    return Window(
        start=start,
        end=end,
    )


def _build_optional_attempt(
    row: Any,
) -> Attempt | None:
    attempt_id = row["active_attempt_id"]
    attempt_number = row["active_attempt_number"]
    window_start = row["active_window_start"]
    window_end = row["active_window_end"]

    required_values = (
        attempt_id,
        attempt_number,
        window_start,
        window_end,
    )

    if all(value is None for value in required_values):
        return None

    if any(value is None for value in required_values):
        raise ControlStateIntegrityError(
            "active attempt requires attempt_id, "
            "attempt_number, window_start, and window_end"
        )

    return Attempt(
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        window=Window(
            start=window_start,
            end=window_end,
        ),
        retry_of_attempt_id=row["active_retry_of_attempt_id"],
    )
