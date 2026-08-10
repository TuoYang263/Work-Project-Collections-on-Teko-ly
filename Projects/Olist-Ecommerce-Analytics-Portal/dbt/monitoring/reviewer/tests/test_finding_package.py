from __future__ import annotations

import sys
import unittest
from pathlib import Path

REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer.finding_package import build_finding_package  # noqa: E402
from pipeline_reviewer.models import RuleEvaluation  # noqa: E402


class FindingPackageTests(unittest.TestCase):
    def test_no_triggered_evaluations_produces_empty_findings(self) -> None:
        evaluations = [
            RuleEvaluation(
                rule_id="M9-R001",
                result="PASS",
                severity=None,
                entity_type="pipeline_run",
                entity_id="run-001",
                evidence_source="pipeline_runs",
                evidence={"status": "success"},
                reason="Pipeline run succeeded",
            ),
            RuleEvaluation(
                rule_id="M9-R005",
                result="NOT_EVALUATED",
                severity=None,
                entity_type="model",
                entity_id="model.olist.stg_orders",
                evidence_source="model_metadata_snapshots",
                evidence={"row_count": None},
                reason="Current model row_count is missing",
            ),
        ]

        package = build_finding_package(
            monitoring_run_id="run-001",
            evaluations=evaluations,
        )

        self.assertEqual(package["monitoring_run_id"], "run-001")
        self.assertEqual(package["findings"], [])

    def test_only_triggered_evaluations_are_included_in_findings(self) -> None:
        evaluations = [
            RuleEvaluation(
                rule_id="M9-R001",
                result="PASS",
                severity=None,
                entity_type="pipeline_run",
                entity_id="run-001",
                evidence_source="pipeline_runs",
                evidence={"status": "success"},
                reason="Pipeline run succeeded",
            ),
            RuleEvaluation(
                rule_id="M9-R005",
                result="TRIGGERED",
                severity="MEDIUM",
                entity_type="model",
                entity_id="model.olist.fct_orders",
                evidence_source="model_metadata_snapshots",
                evidence={
                    "current_row_count": 1400,
                    "baseline_row_count": 1000,
                },
                reason="Row count increased materially",
            ),
            RuleEvaluation(
                rule_id="M9-R006",
                result="NOT_EVALUATED",
                severity=None,
                entity_type="model",
                entity_id="model.olist.stg_orders",
                evidence_source="model_run_results",
                evidence={"execution_time_seconds": None},
                reason="Runtime evidence is missing",
            ),
        ]

        package = build_finding_package(
            monitoring_run_id="run-001",
            evaluations=evaluations,
        )

        self.assertEqual(len(package["findings"]), 1)

        finding = package["findings"][0]

        self.assertEqual(finding["rule_id"], "M9-R005")
        self.assertEqual(finding["result"], "TRIGGERED")
        self.assertEqual(finding["severity"], "MEDIUM")
        self.assertEqual(
            finding["entity_id"],
            "model.olist.fct_orders",
        )
        self.assertEqual(
            finding["finding_id"],
            "run-001:M9-R005:model:model.olist.fct_orders",
        )

    def test_summary_counts_evaluation_results(self) -> None:
        evaluations = [
            RuleEvaluation(
                rule_id="M9-R001",
                result="PASS",
                severity=None,
                entity_type="pipeline_run",
                entity_id="run-001",
                evidence_source="pipeline_runs",
                evidence={},
                reason="pass",
            ),
            RuleEvaluation(
                rule_id="M9-R005",
                result="TRIGGERED",
                severity="MEDIUM",
                entity_type="model",
                entity_id="model.olist.fct_orders",
                evidence_source="model_metadata_snapshots",
                evidence={},
                reason="triggered",
            ),
            RuleEvaluation(
                rule_id="M9-R006",
                result="NOT_EVALUATED",
                severity=None,
                entity_type="model",
                entity_id="model.olist.stg_orders",
                evidence_source="model_run_results",
                evidence={},
                reason="not evaluated",
            ),
        ]

        package = build_finding_package(
            monitoring_run_id="run-001",
            evaluations=evaluations,
        )

        self.assertEqual(
            package["summary"],
            {
                "total_evaluations": 3,
                "pass": 1,
                "triggered": 1,
                "not_evaluated": 1,
            },
        )
