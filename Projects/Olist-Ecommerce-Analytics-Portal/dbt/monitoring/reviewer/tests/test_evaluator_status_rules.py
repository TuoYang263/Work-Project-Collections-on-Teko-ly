from __future__ import annotations

import sys
import unittest
from pathlib import Path


REVIEWER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REVIEWER_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from pipeline_reviewer import (  # noqa: E402
    DeterministicEvaluator,
    load_rule_catalog,
)


class StatusRuleEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog_path = (
            REVIEWER_ROOT / "config" / "rule_catalog.yml"
        )
        cls.catalog = load_rule_catalog(catalog_path)
        cls.evaluator = DeterministicEvaluator(cls.catalog)

    @staticmethod
    def _base_evidence() -> dict[str, list[dict[str, object]]]:
        return {
            "pipeline_runs": [
                {
                    "monitoring_run_id": "run-001",
                    "status": "success",
                }
            ],
            "model_run_results": [
                {
                    "monitoring_run_id": "run-001",
                    "unique_id": "model.olist.fct_orders",
                    "model_name": "fct_orders",
                    "status": "success",
                }
            ],
            "test_run_results": [
                {
                    "monitoring_run_id": "run-001",
                    "unique_id": "test.olist.not_null_orders",
                    "test_name": "not_null_orders",
                    "status": "pass",
                }
            ],
        }

    def test_all_successful_evidence_passes(self) -> None:
        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=self._base_evidence(),
        )

        self.assertEqual(len(evaluations), 3)
        self.assertTrue(
            all(item.result == "PASS" for item in evaluations)
        )
        self.assertTrue(
            all(item.severity is None for item in evaluations)
        )

    def test_status_values_are_trimmed_and_lowercased(self) -> None:
        evidence = self._base_evidence()
        evidence["pipeline_runs"][0]["status"] = " SUCCESS "
        evidence["model_run_results"][0]["status"] = " Success "
        evidence["test_run_results"][0]["status"] = " PASS "

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )

        self.assertTrue(
            all(item.result == "PASS" for item in evaluations)
        )

    def test_model_error_and_test_failure_trigger_high(self) -> None:
        evidence = self._base_evidence()
        evidence["model_run_results"][0]["status"] = "error"
        evidence["test_run_results"][0]["status"] = "fail"

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )
        by_rule = {
            item.rule_id: item
            for item in evaluations
        }

        self.assertEqual(
            by_rule["M9-R002"].result,
            "TRIGGERED",
        )
        self.assertEqual(
            by_rule["M9-R002"].severity,
            "HIGH",
        )
        self.assertEqual(
            by_rule["M9-R003"].result,
            "TRIGGERED",
        )
        self.assertEqual(
            by_rule["M9-R003"].severity,
            "HIGH",
        )

    def test_test_warning_triggers_medium(self) -> None:
        evidence = self._base_evidence()
        evidence["test_run_results"][0]["status"] = "warn"

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )
        r003 = next(
            item
            for item in evaluations
            if item.rule_id == "M9-R003"
        )

        self.assertEqual(r003.result, "TRIGGERED")
        self.assertEqual(r003.severity, "MEDIUM")

    def test_missing_required_model_field_is_not_evaluated(
        self,
    ) -> None:
        evidence = self._base_evidence()
        evidence["model_run_results"][0]["model_name"] = ""

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )
        r002 = next(
            item
            for item in evaluations
            if item.rule_id == "M9-R002"
        )

        self.assertEqual(r002.result, "NOT_EVALUATED")
        self.assertIsNone(r002.severity)
        self.assertIn("model_name", r002.reason)

    def test_duplicate_pipeline_records_are_not_evaluated(
        self,
    ) -> None:
        evidence = self._base_evidence()
        evidence["pipeline_runs"].append(
            {
                "monitoring_run_id": "run-001",
                "status": "success",
            }
        )

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )
        r001 = next(
            item
            for item in evaluations
            if item.rule_id == "M9-R001"
        )

        self.assertEqual(r001.result, "NOT_EVALUATED")
        self.assertIn("exactly one", r001.reason)

    def test_records_from_other_runs_are_ignored(self) -> None:
        evidence = self._base_evidence()
        evidence["model_run_results"].append(
            {
                "monitoring_run_id": "run-999",
                "unique_id": "model.olist.other",
                "model_name": "other",
                "status": "error",
            }
        )

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )
        r002_items = [
            item
            for item in evaluations
            if item.rule_id == "M9-R002"
        ]

        self.assertEqual(len(r002_items), 1)
        self.assertEqual(r002_items[0].result, "PASS")

    def test_evaluation_evidence_is_immutable(self) -> None:
        evidence = self._base_evidence()
        evidence["model_run_results"][0]["status"] = "error"

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )

        r002 = next(
            item
            for item in evaluations
            if item.rule_id == "M9-R002"
        )

        with self.assertRaises(TypeError):
            r002.evidence["status"] = "success"

    def test_to_dict_returns_detached_mutable_copy(self) -> None:
        evidence = self._base_evidence()
        evidence["model_run_results"][0]["status"] = "error"

        evaluations = self.evaluator.evaluate_status_rules(
            selected_run_id="run-001",
            evidence=evidence,
        )

        r002 = next(
            item
            for item in evaluations
            if item.rule_id == "M9-R002"
        )

        payload = r002.to_dict()
        payload["evidence"]["status"] = "success"

        self.assertEqual(
            r002.evidence["status"],
            "error",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)