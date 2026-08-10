from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer.ai_explainer import (
    _validate_explanation_response,
    build_explanation_report,
)


class AIExplanationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finding_id = "run-001:M9-R005:model:" "model.olist.fct_orders"

        self.finding_package = {
            "monitoring_run_id": "run-001",
            "summary": {
                "total_evaluations": 3,
                "pass": 2,
                "triggered": 1,
                "not_evaluated": 0,
            },
            "findings": [
                {
                    "finding_id": self.finding_id,
                    "rule_id": "M9-R005",
                    "result": "TRIGGERED",
                    "severity": "MEDIUM",
                }
            ],
        }

    def test_valid_explanation_response_is_accepted(self) -> None:
        response_text = json.dumps(
            {
                "pipeline_summary": ("One deterministic finding was detected."),
                "findings": [
                    {
                        "finding_id": self.finding_id,
                        "explanation": ("Row count increased materially."),
                        "impact": ("Downstream metrics may need investigation."),
                        "recommended_actions": ["Inspect recent upstream changes."],
                    }
                ],
            }
        )

        result = _validate_explanation_response(
            self.finding_package,
            response_text,
        )

        self.assertEqual(
            result["findings"][0]["finding_id"],
            self.finding_id,
        )

    def test_invented_finding_id_is_rejected(self) -> None:
        response_text = json.dumps(
            {
                "pipeline_summary": "One finding.",
                "findings": [
                    {
                        "finding_id": "invented-finding",
                        "explanation": "Invented explanation.",
                        "impact": "Unknown.",
                        "recommended_actions": [],
                    }
                ],
            }
        )

        with self.assertRaises(ValueError):
            _validate_explanation_response(
                self.finding_package,
                response_text,
            )

    def test_missing_finding_is_rejected(self) -> None:
        finding_package = {
            **self.finding_package,
            "findings": [
                self.finding_package["findings"][0],
                {
                    "finding_id": (
                        "run-001:M9-R006:model:" "model.olist.fct_order_items"
                    ),
                    "rule_id": "M9-R006",
                    "result": "TRIGGERED",
                    "severity": "MEDIUM",
                },
            ],
        }

        response_text = json.dumps(
            {
                "pipeline_summary": "Two findings exist.",
                "findings": [
                    {
                        "finding_id": self.finding_id,
                        "explanation": "First finding.",
                        "impact": "Potential impact.",
                        "recommended_actions": [],
                    }
                ],
            }
        )

        with self.assertRaises(ValueError):
            _validate_explanation_response(
                finding_package,
                response_text,
            )

    def test_duplicate_finding_id_is_rejected(self) -> None:
        response_text = json.dumps(
            {
                "pipeline_summary": "Duplicate response.",
                "findings": [
                    {
                        "finding_id": self.finding_id,
                        "explanation": "First explanation.",
                        "impact": "Impact.",
                        "recommended_actions": [],
                    },
                    {
                        "finding_id": self.finding_id,
                        "explanation": "Duplicate explanation.",
                        "impact": "Impact.",
                        "recommended_actions": [],
                    },
                ],
            }
        )

        with self.assertRaises(ValueError):
            _validate_explanation_response(
                self.finding_package,
                response_text,
            )

    def test_empty_findings_skip_ai(self) -> None:
        finding_package = {
            "monitoring_run_id": "run-001",
            "summary": {
                "total_evaluations": 3,
                "pass": 3,
                "triggered": 0,
                "not_evaluated": 0,
            },
            "findings": [],
        }

        report = build_explanation_report(
            finding_package=finding_package,
            project_id="test-project",
        )

        self.assertEqual(report["ai_status"], "SKIPPED")
        self.assertEqual(report["findings"], [])

    @patch("pipeline_reviewer.ai_explainer.explain_finding_package")
    def test_ai_success_returns_success_report(
        self,
        mock_explain,
    ) -> None:
        # Replace the real Vertex call with a controlled fake response.
        # This keeps the unit test fast, deterministic, and free of API cost.
        mock_explain.return_value = {
            "pipeline_summary": "One finding explained.",
            "findings": [
                {
                    "finding_id": self.finding_id,
                    "explanation": "Explanation.",
                    "impact": "Impact.",
                    "recommended_actions": ["Investigate upstream changes."],
                }
            ],
        }

        report = build_explanation_report(
            finding_package=self.finding_package,
            project_id="test-project",
        )

        # Assert that the wrapper correctly maps a successful AI call
        # into a SUCCESS report while preserving the deterministic finding ID.
        self.assertEqual(report["ai_status"], "SUCCESS")
        self.assertEqual(
            report["findings"][0]["finding_id"],
            self.finding_id,
        )

    @patch("pipeline_reviewer.ai_explainer.explain_finding_package")
    def test_ai_failure_returns_unavailable_report(
        self,
        mock_explain,
    ) -> None:
        # Simulate a Vertex failure without making a real network request.
        mock_explain.side_effect = RuntimeError("Vertex unavailable")

        report = build_explanation_report(
            finding_package=self.finding_package,
            project_id="test-project",
        )

        # AI failure must degrade gracefully:
        # deterministic review remains valid and AI becomes UNAVAILABLE.
        self.assertEqual(
            report["ai_status"],
            "UNAVAILABLE",
        )
        self.assertEqual(report["findings"], [])
        self.assertIn(
            "Vertex unavailable",
            report["error"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
