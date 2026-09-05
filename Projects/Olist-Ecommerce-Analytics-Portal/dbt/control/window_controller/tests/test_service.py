import unittest
from datetime import datetime, timedelta, timezone

from window_controller.models import (
    ControlState,
    PipelineState,
    Window,
)
from window_controller.service import (
    complete_current_window,
    derive_next_monthly_window,
    derive_next_window,
    fail_current_window,
    move_to_waiting_retry,
    start_new_window,
    start_retry,
)

SOURCE_START = datetime(
    2016,
    9,
    1,
    tzinfo=timezone.utc,
)

SOURCE_END = datetime(
    2018,
    11,
    1,
    tzinfo=timezone.utc,
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
        self.assertEqual(
            failed.cycle_id,
            self.idle_state.cycle_id,
        )

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
        self.assertEqual(
            retrying.cycle_id,
            failed.cycle_id,
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

    def test_derive_initial_monthly_window(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
        )

        cycle_id, window = derive_next_monthly_window(
            state,
            source_start=SOURCE_START,
            source_end=SOURCE_END,
        )

        self.assertEqual(cycle_id, 1)

        self.assertEqual(
            window.start,
            SOURCE_START,
        )

        self.assertEqual(
            window.end,
            datetime(
                2016,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_derive_next_monthly_window_from_watermark(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
            last_successful_window=Window(
                start=datetime(
                    2016,
                    9,
                    1,
                    tzinfo=timezone.utc,
                ),
                end=datetime(
                    2016,
                    10,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        cycle_id, window = derive_next_monthly_window(
            state,
            source_start=SOURCE_START,
            source_end=SOURCE_END,
        )

        self.assertEqual(cycle_id, 1)

        self.assertEqual(
            window.start,
            datetime(
                2016,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(
            window.end,
            datetime(
                2016,
                11,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_calendar_month_handles_february(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
            last_successful_window=Window(
                start=datetime(
                    2017,
                    1,
                    1,
                    tzinfo=timezone.utc,
                ),
                end=datetime(
                    2017,
                    2,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        _, window = derive_next_monthly_window(
            state,
            source_start=datetime(
                2017,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            source_end=datetime(
                2017,
                4,
                1,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(
            window.start,
            datetime(
                2017,
                2,
                1,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(
            window.end,
            datetime(
                2017,
                3,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_calendar_month_handles_leap_february(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
            last_successful_window=Window(
                start=datetime(
                    2020,
                    1,
                    1,
                    tzinfo=timezone.utc,
                ),
                end=datetime(
                    2020,
                    2,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        _, window = derive_next_monthly_window(
            state,
            source_start=datetime(
                2020,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            source_end=datetime(
                2020,
                4,
                1,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(
            window.end,
            datetime(
                2020,
                3,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_calendar_month_handles_year_rollover(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
            last_successful_window=Window(
                start=datetime(
                    2017,
                    11,
                    1,
                    tzinfo=timezone.utc,
                ),
                end=datetime(
                    2017,
                    12,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        _, window = derive_next_monthly_window(
            state,
            source_start=datetime(
                2017,
                11,
                1,
                tzinfo=timezone.utc,
            ),
            source_end=datetime(
                2018,
                2,
                1,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(
            window.start,
            datetime(
                2017,
                12,
                1,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual(
            window.end,
            datetime(
                2018,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_source_end_starts_next_cycle(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
            last_successful_window=Window(
                start=datetime(
                    2018,
                    10,
                    1,
                    tzinfo=timezone.utc,
                ),
                end=SOURCE_END,
            ),
        )

        cycle_id, window = derive_next_monthly_window(
            state,
            source_start=SOURCE_START,
            source_end=SOURCE_END,
        )

        self.assertEqual(cycle_id, 2)
        self.assertEqual(window.start, SOURCE_START)

        self.assertEqual(
            window.end,
            datetime(
                2016,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_legacy_cycle_zero_starts_monthly_cycle_one(
        self,
    ) -> None:
        legacy_state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=0,
            last_successful_window=Window(
                start=datetime(
                    2016,
                    9,
                    4,
                    tzinfo=timezone.utc,
                ),
                end=datetime(
                    2016,
                    9,
                    5,
                    tzinfo=timezone.utc,
                ),
            ),
        )

        cycle_id, window = derive_next_monthly_window(
            legacy_state,
            source_start=SOURCE_START,
            source_end=SOURCE_END,
        )

        self.assertEqual(cycle_id, 1)
        self.assertEqual(window.start, SOURCE_START)

        self.assertEqual(
            window.end,
            datetime(
                2016,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )

    def test_new_window_allows_next_cycle_reset(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
            last_successful_window=Window(
                start=datetime(
                    2018,
                    10,
                    1,
                    tzinfo=timezone.utc,
                ),
                end=SOURCE_END,
            ),
        )

        new_state = start_new_window(
            state,
            cycle_id=2,
            window=Window(
                start=SOURCE_START,
                end=datetime(
                    2016,
                    10,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            attempt_id="attempt-cycle-2",
        )

        self.assertEqual(new_state.cycle_id, 2)
        self.assertEqual(
            new_state.state,
            PipelineState.RUNNING,
        )

        self.assertEqual(
            new_state.active_attempt.window.start,
            SOURCE_START,
        )

    def test_new_window_cannot_move_to_earlier_cycle(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=2,
        )

        with self.assertRaises(ValueError):
            start_new_window(
                state,
                cycle_id=1,
                window=Window(
                    start=SOURCE_START,
                    end=datetime(
                        2016,
                        10,
                        1,
                        tzinfo=timezone.utc,
                    ),
                ),
                attempt_id="attempt-invalid",
            )

    def test_new_window_cannot_skip_cycles(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
        )

        with self.assertRaises(ValueError):
            start_new_window(
                state,
                cycle_id=3,
                window=Window(
                    start=SOURCE_START,
                    end=datetime(
                        2016,
                        10,
                        1,
                        tzinfo=timezone.utc,
                    ),
                ),
                attempt_id="attempt-invalid",
            )

    def test_monthly_window_rejects_source_end_before_start(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
        )

        with self.assertRaises(ValueError):
            derive_next_monthly_window(
                state,
                source_start=SOURCE_END,
                source_end=SOURCE_START,
            )

    def test_monthly_window_rejects_non_month_boundary(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
        )

        with self.assertRaises(ValueError):
            derive_next_monthly_window(
                state,
                source_start=datetime(
                    2016,
                    9,
                    4,
                    tzinfo=timezone.utc,
                ),
                source_end=SOURCE_END,
            )

    def test_monthly_window_rejects_naive_source_boundary(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
        )

        with self.assertRaises(ValueError):
            derive_next_monthly_window(
                state,
                source_start=datetime(
                    2016,
                    9,
                    1,
                ),
                source_end=SOURCE_END,
            )

    def test_monthly_window_rejects_non_month_source_end(
        self,
    ) -> None:
        state = ControlState(
            pipeline_name="olist-dbt-build-job",
            environment="prod",
            state=PipelineState.IDLE,
            cycle_id=1,
        )

        with self.assertRaises(ValueError):
            derive_next_monthly_window(
                state,
                source_start=SOURCE_START,
                source_end=datetime(
                    2018,
                    10,
                    17,
                    tzinfo=timezone.utc,
                ),
            )


if __name__ == "__main__":
    unittest.main()
