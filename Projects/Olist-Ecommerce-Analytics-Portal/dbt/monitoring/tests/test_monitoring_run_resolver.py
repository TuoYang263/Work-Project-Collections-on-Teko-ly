import unittest

from monitoring.monitoring_run_resolver import (
    BigQueryMonitoringRunResolver,
    MonitoringRunIntegrityError,
    MonitoringRunNotFoundError,
)


class FakeQueryJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self, rows):
        self.project = "test-project"
        self._rows = rows
        self.query_calls = []

    def query(
        self,
        query,
        *,
        job_config,
    ):
        self.query_calls.append(
            {
                "query": query,
                "job_config": job_config,
            }
        )

        return FakeQueryJob(self._rows)


class TestBigQueryMonitoringRunResolver(unittest.TestCase):
    def test_returns_exact_monitoring_run_id(self):
        client = FakeClient(
            [
                {
                    "monitoring_run_id": "monitoring-run-001",
                }
            ]
        )

        resolver = BigQueryMonitoringRunResolver(
            client,
        )

        result = resolver.resolve(
            control_attempt_id="attempt-001",
        )

        self.assertEqual(
            result,
            "monitoring-run-001",
        )

        self.assertEqual(
            len(client.query_calls),
            1,
        )

        query_call = client.query_calls[0]

        self.assertIn(
            "WHERE control_attempt_id = " "@control_attempt_id",
            query_call["query"],
        )

        parameters = query_call["job_config"].query_parameters

        self.assertEqual(
            len(parameters),
            1,
        )

        self.assertEqual(
            parameters[0].name,
            "control_attempt_id",
        )

        self.assertEqual(
            parameters[0].value,
            "attempt-001",
        )

    def test_missing_monitoring_run_is_rejected(self):
        client = FakeClient([])

        resolver = BigQueryMonitoringRunResolver(
            client,
        )

        with self.assertRaises(MonitoringRunNotFoundError):
            resolver.resolve(
                control_attempt_id="attempt-001",
            )

    def test_multiple_monitoring_runs_are_rejected(
        self,
    ):
        client = FakeClient(
            [
                {
                    "monitoring_run_id": "monitoring-run-001",
                },
                {
                    "monitoring_run_id": "monitoring-run-002",
                },
            ]
        )

        resolver = BigQueryMonitoringRunResolver(
            client,
        )

        with self.assertRaises(MonitoringRunIntegrityError):
            resolver.resolve(
                control_attempt_id="attempt-001",
            )

    def test_empty_attempt_id_is_rejected(self):
        client = FakeClient([])

        resolver = BigQueryMonitoringRunResolver(
            client,
        )

        with self.assertRaises(ValueError):
            resolver.resolve(
                control_attempt_id="   ",
            )

        self.assertEqual(
            client.query_calls,
            [],
        )

    def test_empty_monitoring_run_id_is_rejected(
        self,
    ):
        client = FakeClient(
            [
                {
                    "monitoring_run_id": "   ",
                }
            ]
        )

        resolver = BigQueryMonitoringRunResolver(
            client,
        )

        with self.assertRaises(MonitoringRunIntegrityError):
            resolver.resolve(
                control_attempt_id="attempt-001",
            )


if __name__ == "__main__":
    unittest.main()
