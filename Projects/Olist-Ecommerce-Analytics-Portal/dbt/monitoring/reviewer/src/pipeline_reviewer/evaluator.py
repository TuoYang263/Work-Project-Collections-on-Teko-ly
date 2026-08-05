from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import RuleEvaluation


class EvaluationError(ValueError):
    pass


class DeterministicEvaluator:
    STATUS_RULE_IDS = ("M9-R001", "M9-R002", "M9-R003")

    def __init__(self, catalog: Mapping[str, Any]) -> None:
        rules = catalog.get("rules")

        if not isinstance(rules, list):
            raise EvaluationError(
                "Catalog field 'rules' must be a list"
            )

        self._rules_by_id = {
            rule["rule_id"]: rule
            for rule in rules
            if isinstance(rule, dict) and "rule_id" in rule
        }

        missing_rules = [
            rule_id
            for rule_id in self.STATUS_RULE_IDS
            if rule_id not in self._rules_by_id
        ]

        if missing_rules:
            raise EvaluationError(
                f"Required status rules are missing: {missing_rules}"
            )

    def evaluate_status_rules(
        self,
        selected_run_id: str,
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        if not selected_run_id.strip():
            raise EvaluationError(
                "selected_run_id must be non-empty"
            )

        evaluations: list[RuleEvaluation] = []

        for rule_id in self.STATUS_RULE_IDS:
            rule = self._rules_by_id[rule_id]
            evaluations.extend(
                self._evaluate_rule(
                    rule=rule,
                    selected_run_id=selected_run_id,
                    evidence=evidence,
                )
            )

        return evaluations

    def _evaluate_rule(
        self,
        rule: Mapping[str, Any],
        selected_run_id: str,
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        evidence_definition = rule["required_evidence"][0]
        source = evidence_definition["source"]
        source_rows = evidence.get(source, [])

        if not isinstance(source_rows, Sequence):
            raise EvaluationError(
                f"Evidence source '{source}' must be a sequence"
            )

        selected_rows = [
            dict(row)
            for row in source_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        record_requirement = evidence_definition.get(
            "record_requirement"
        )

        if record_requirement == "exactly_one":
            if len(selected_rows) != 1:
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Expected exactly one selected-run record "
                            f"but found {len(selected_rows)}"
                        ),
                        entity_id=selected_run_id,
                    )
                ]
        elif not selected_rows:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=(
                        "No evidence records were found for the "
                        "selected pipeline run"
                    ),
                    entity_id=None,
                )
            ]

        return [
            self._evaluate_row(
                rule=rule,
                source=source,
                evidence_definition=evidence_definition,
                row=row,
            )
            for row in selected_rows
        ]

    def _evaluate_row(
        self,
        rule: Mapping[str, Any],
        source: str,
        evidence_definition: Mapping[str, Any],
        row: dict[str, Any],
    ) -> RuleEvaluation:
        required_fields = self._required_fields(
            evidence_definition
        )
        missing_fields = [
            field
            for field in required_fields
            if self._is_missing(row.get(field))
        ]

        entity_type = rule["applicability"]["entity_type"]
        entity_id = self._entity_id(
            entity_type=entity_type,
            row=row,
        )

        if missing_fields:
            return self._not_evaluated(
                rule=rule,
                source=source,
                reason=(
                    "Missing required evidence fields: "
                    f"{sorted(missing_fields)}"
                ),
                entity_id=entity_id,
                evidence=row,
            )

        trigger_logic = rule["trigger_logic"]
        operator = trigger_logic["operator"]

        if operator != "not_equals":
            raise EvaluationError(
                f"Unsupported operator for {rule['rule_id']}: "
                f"{operator}"
            )

        metric_field = trigger_logic["field"].split(".")[-1]
        normalization = trigger_logic.get("normalization")
        actual_value = self._normalize(
            row[metric_field],
            normalization,
        )
        comparison_value = self._normalize(
            trigger_logic["comparison_value"],
            normalization,
        )

        triggered = actual_value != comparison_value
        result = "TRIGGERED" if triggered else "PASS"
        severity = (
            self._resolve_severity(rule, actual_value)
            if triggered
            else None
        )

        reason = (
            f"Normalized value '{actual_value}' does not equal "
            f"'{comparison_value}'"
            if triggered
            else (
                f"Normalized value '{actual_value}' equals "
                f"'{comparison_value}'"
            )
        )

        return RuleEvaluation(
            rule_id=rule["rule_id"],
            result=result,
            severity=severity,
            entity_type=entity_type,
            entity_id=entity_id,
            evidence_source=source,
            evidence=row,
            reason=reason,
        )

    @staticmethod
    def _required_fields(
        evidence_definition: Mapping[str, Any],
    ) -> list[str]:
        fields = evidence_definition.get("fields")

        if isinstance(fields, list):
            return list(fields)

        value_requirements = evidence_definition.get(
            "value_requirements"
        )

        if isinstance(value_requirements, dict):
            return list(value_requirements)

        raise EvaluationError(
            "Required evidence must define either 'fields' or "
            "'value_requirements'"
        )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or (
            isinstance(value, str) and not value.strip()
        )

    @staticmethod
    def _normalize(value: Any, normalization: str | None) -> Any:
        if normalization is None:
            return value

        if normalization == "trim_and_lowercase":
            if not isinstance(value, str):
                raise EvaluationError(
                    "trim_and_lowercase requires a string value"
                )
            return value.strip().lower()

        raise EvaluationError(
            f"Unsupported normalization: {normalization}"
        )

    @staticmethod
    def _entity_id(
        entity_type: str,
        row: Mapping[str, Any],
    ) -> str | None:
        if entity_type == "pipeline_run":
            return row.get("monitoring_run_id")

        if entity_type in {"model", "test"}:
            return row.get("unique_id")

        return None

    @staticmethod
    def _resolve_severity(
        rule: Mapping[str, Any],
        normalized_value: Any,
    ) -> str:
        severity_logic = rule.get("severity_logic")

        if not isinstance(severity_logic, dict):
            return rule["default_severity"]

        mapping = severity_logic.get("mapping", {})

        if normalized_value in mapping:
            return mapping[normalized_value]

        return severity_logic.get(
            "fallback",
            rule["default_severity"],
        )

    @staticmethod
    def _not_evaluated(
        rule: Mapping[str, Any],
        source: str,
        reason: str,
        entity_id: str | None,
        evidence: dict[str, Any] | None = None,
    ) -> RuleEvaluation:
        return RuleEvaluation(
            rule_id=rule["rule_id"],
            result="NOT_EVALUATED",
            severity=None,
            entity_type=rule["applicability"]["entity_type"],
            entity_id=entity_id,
            evidence_source=source,
            evidence=evidence or {},
            reason=reason,
        )