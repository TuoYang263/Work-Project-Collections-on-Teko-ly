from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer.review_persistence import (
    ReviewPersistenceError,
    prepare_review_snapshot,
)


def id_factory():
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"id-{counter}"

    return next_id


def valid_payload() -> dict:
    return {
        "monitoring_run_id": "run-1",
        "evaluations": [
            {
                "rule_id": "M9-R001",
                "result": "PASS",
                "severity": None,
                "entity_type": "pipeline_run",
                "entity_id": "run-1",
                "evidence_source": "pipeline_runs",
                "evidence": {"status": "success"},
                "reason": "Pipeline succeeded.",
            },
            {
                "rule_id": "M9-R006",
                "result": "TRIGGERED",
                "severity": "MEDIUM",
                "entity_type": "model",
                "entity_id": "model.example",
                "evidence_source": "model_run_results",
                "evidence": {"runtime": 20.0},
                "reason": "Runtime increased.",
            },
            {
                "rule_id": "M9-R005",
                "result": "NOT_EVALUATED",
                "severity": None,
                "entity_type": "model",
                "entity_id": "model.view_example",
                "evidence_source": "model_metadata_snapshots",
                "evidence": {},
                "reason": "Row count unavailable.",
            },
        ],
        "finding_package": {
            "monitoring_run_id": "run-1",
            "summary": {
                "total_evaluations": 3,
                "pass": 1,
                "triggered": 1,
                "not_evaluated": 1,
            },
            "findings": [
                {
                    "finding_id": ("run-1:M9-R006:model:model.example"),
                    "rule_id": "M9-R006",
                    "result": "TRIGGERED",
                    "severity": "MEDIUM",
                    "entity_type": "model",
                    "entity_id": "model.example",
                    "evidence_source": "model_run_results",
                    "evidence": {"runtime": 20.0},
                    "reason": "Runtime increased.",
                }
            ],
        },
    }


class ReviewPersistenceTests(unittest.TestCase):
    def test_prepares_counts_and_finding_id(self):
        snapshot = prepare_review_snapshot(
            valid_payload(),
            id_factory=id_factory(),
        )

        self.assertEqual(snapshot.review_id, "id-4")
        self.assertEqual(snapshot.total_evaluations, 3)
        self.assertEqual(snapshot.pass_count, 1)
        self.assertEqual(snapshot.triggered_count, 1)
        self.assertEqual(snapshot.not_evaluated_count, 1)

        evaluations = json.loads(snapshot.evaluations_json)

        self.assertIsNone(evaluations[0]["finding_id"])
        self.assertEqual(
            evaluations[1]["finding_id"],
            "run-1:M9-R006:model:model.example",
        )
        self.assertIsNone(evaluations[2]["finding_id"])

    def test_rejects_summary_mismatch(self):
        payload = valid_payload()
        payload["finding_package"]["summary"]["pass"] = 2

        with self.assertRaises(ReviewPersistenceError):
            prepare_review_snapshot(payload)

    def test_rejects_missing_triggered_finding(self):
        payload = valid_payload()
        payload["finding_package"]["findings"] = []

        with self.assertRaises(ReviewPersistenceError):
            prepare_review_snapshot(payload)

    def test_rejects_extra_deterministic_finding(self):
        payload = valid_payload()
        payload["finding_package"]["findings"].append(
            {
                "finding_id": "extra",
                "rule_id": "M9-R004",
                "entity_type": "model",
                "entity_id": "model.extra",
            }
        )

        with self.assertRaises(ReviewPersistenceError):
            prepare_review_snapshot(payload)

    def test_rejects_monitoring_run_mismatch(self):
        payload = valid_payload()
        payload["finding_package"]["monitoring_run_id"] = "other-run"

        with self.assertRaises(ReviewPersistenceError):
            prepare_review_snapshot(payload)

    def test_persistence_supports_wide_float_evidence(self):
        from pipeline_reviewer.review_persistence import (
            persist_review_snapshot,
        )

        payload = valid_payload()
        payload["evaluations"][1]["evidence"] = {
            "runtime_seconds": 2.5535964965820312,
        }

        snapshot = prepare_review_snapshot(
            payload,
            id_factory=id_factory(),
        )

        self.assertIn(
            "2.5535964965820312",
            snapshot.evaluations_json,
        )

        class FakeJob:
            def result(self):
                return None

        class FakeClient:
            def __init__(self):
                self.sql = None
                self.location = None

            def query(
                self,
                sql,
                *,
                job_config,
                location,
            ):
                self.sql = sql
                self.location = location
                return FakeJob()

        client = FakeClient()

        persist_review_snapshot(
            snapshot,
            project_id="test-project",
            dataset_id="olist_monitoring",
            location="EU",
            client=client,
        )

        self.assertIn(
            "wide_number_mode => 'round'",
            client.sql,
        )
        self.assertEqual(client.location, "EU")


if __name__ == "__main__":
    unittest.main()
