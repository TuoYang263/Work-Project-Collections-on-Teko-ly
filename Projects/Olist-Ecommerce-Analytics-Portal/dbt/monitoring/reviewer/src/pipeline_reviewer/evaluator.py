from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .baseline import median_baseline
from .models import RuleEvaluation


class EvaluationError(ValueError):
    pass


class DeterministicEvaluator:
    STATUS_RULE_IDS = ("M9-R001", "M9-R002", "M9-R003")

    HISTORICAL_RULE_IDS = ("M9-R004", "M9-R005", "M9-R006")

    def __init__(self, catalog: Mapping[str, Any]) -> None:
        rules = catalog.get("rules")

        if not isinstance(rules, list):
            raise EvaluationError("Catalog field 'rules' must be a list")

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
            raise EvaluationError(f"Required status rules are missing: {missing_rules}")

    def evaluate_status_rules(
        self,
        selected_run_id: str,
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        if not selected_run_id.strip():
            raise EvaluationError("selected_run_id must be non-empty")

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

    def evaluate_historical_rules(
        self,
        selected_run_id: str,
        comparable_run_ids: Sequence[str],
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        if not selected_run_id.strip():
            raise EvaluationError("selected_run_id must be non-empty")

        missing_rules = [
            rule_id
            for rule_id in self.HISTORICAL_RULE_IDS
            if rule_id not in self._rules_by_id
        ]
        if missing_rules:
            raise EvaluationError(
                f"Required historical rules are missing: {missing_rules}"
            )

        r004_evaluations = self._evaluate_model_missing_rule(
            rule=self._rules_by_id["M9-R004"],
            selected_run_id=selected_run_id,
            comparable_run_ids=comparable_run_ids,
            evidence=evidence,
        )
        r005_evaluations = self._evaluate_row_count_anomaly_rule(
            rule=self._rules_by_id["M9-R005"],
            selected_run_id=selected_run_id,
            comparable_run_ids=comparable_run_ids,
            evidence=evidence,
        )
        r006_evaluations = self._evaluate_runtime_regression_rule(
            rule=self._rules_by_id["M9-R006"],
            selected_run_id=selected_run_id,
            comparable_run_ids=comparable_run_ids,
            evidence=evidence,
        )

        return [
            *r004_evaluations,
            *r005_evaluations,
            *r006_evaluations,
        ]

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
            raise EvaluationError(f"Evidence source '{source}' must be a sequence")

        selected_rows = [
            dict(row)
            for row in source_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        record_requirement = evidence_definition.get("record_requirement")

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

    def _evaluate_model_missing_rule(
        self,
        rule: Mapping[str, Any],
        selected_run_id: str,
        comparable_run_ids: Sequence[str],
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        source = "model_metadata_snapshots"

        if not comparable_run_ids:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=(
                        "No previous comparable successful pipeline run " "is available"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                    },
                )
            ]

        baseline_run_id = comparable_run_ids[0]

        pipeline_rows = evidence.get("pipeline_runs", ())
        metadata_rows = evidence.get(source, ())

        selected_pipeline_rows = [
            dict(row)
            for row in pipeline_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        baseline_pipeline_rows = [
            dict(row)
            for row in pipeline_rows
            if row.get("monitoring_run_id") == baseline_run_id
        ]

        if len(selected_pipeline_rows) != 1:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Expected exactly one selected pipeline-run record "
                        f"but found {len(selected_pipeline_rows)}"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        if len(baseline_pipeline_rows) != 1:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Expected exactly one baseline pipeline-run record "
                        f"but found {len(baseline_pipeline_rows)}"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        selected_pipeline = selected_pipeline_rows[0]
        baseline_pipeline = baseline_pipeline_rows[0]

        selected_status_raw = selected_pipeline.get("status")

        if self._is_missing(selected_status_raw):
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason="Selected pipeline-run record is missing required field: status",
                    entity_id=selected_run_id,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        selected_status = self._normalize(
            selected_status_raw,
            "trim_and_lowercase",
        )

        allowed_selected_statuses = {
            str(status).strip().lower()
            for status in rule["applicability"].get(
                "selected_run_statuses",
                [],
            )
        }

        if selected_status not in allowed_selected_statuses:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "R004 applies only to selected runs with an "
                        f"eligible status; found {selected_status!r}"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                        "selected_status": selected_status,
                    },
                )
            ]

        baseline_status_raw = baseline_pipeline.get("status")

        if self._is_missing(baseline_status_raw):
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason="Baseline pipeline-run record is missing required field: status",
                    entity_id=baseline_run_id,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        baseline_status = self._normalize(
            baseline_status_raw,
            "trim_and_lowercase",
        )

        eligible_baseline_statuses = {
            str(status).strip().lower()
            for status in rule["baseline_policy"].get(
                "eligible_statuses",
                [],
            )
        }

        if baseline_status not in eligible_baseline_statuses:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Baseline run does not have an eligible status; "
                        f"found {baseline_status!r}"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                        "baseline_status": baseline_status,
                    },
                )
            ]

        selected_inventory = [
            dict(row)
            for row in metadata_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        baseline_inventory = [
            dict(row)
            for row in metadata_rows
            if row.get("monitoring_run_id") == baseline_run_id
        ]

        if not selected_inventory:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=("No selected-run model inventory evidence " "is available"),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        if not baseline_inventory:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=("No baseline-run model inventory evidence " "is available"),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        required_inventory_fields = (
            "monitoring_run_id",
            "unique_id",
            "model_name",
            "resource_type",
        )

        for inventory_name, rows in (
            ("selected", selected_inventory),
            ("baseline", baseline_inventory),
        ):
            for row in rows:
                missing_fields = [
                    field
                    for field in required_inventory_fields
                    if self._is_missing(row.get(field))
                ]

                if missing_fields:
                    return [
                        self._not_evaluated(
                            rule=rule,
                            source=source,
                            reason=(
                                f"{inventory_name.capitalize()} inventory "
                                "contains a record with missing required "
                                f"fields: {sorted(missing_fields)}"
                            ),
                            entity_id=row.get("unique_id"),
                            evidence=row,
                        )
                    ]

        allowed_resource_types = {
            str(resource_type).strip().lower()
            for resource_type in (
                rule["trigger_logic"].get("filters", {}).get("resource_type_in", [])
            )
        }

        selected_models = [
            row
            for row in selected_inventory
            if str(row["resource_type"]).strip().lower() in allowed_resource_types
        ]

        baseline_models = [
            row
            for row in baseline_inventory
            if str(row["resource_type"]).strip().lower() in allowed_resource_types
        ]

        if not selected_models:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=(
                        "No eligible selected-run model records remain after "
                        "applying the resource-type filter"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        if not baseline_models:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=(
                        "No eligible baseline model records remain after "
                        "applying the resource-type filter"
                    ),
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                    },
                )
            ]

        selected_by_id: dict[str, dict[str, Any]] = {}
        baseline_by_id: dict[str, dict[str, Any]] = {}

        for row in selected_models:
            unique_id = str(row["unique_id"]).strip()

            if unique_id in selected_by_id:
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Selected model inventory contains duplicate "
                            f"unique_id {unique_id!r}"
                        ),
                        entity_id=unique_id,
                        evidence=row,
                    )
                ]

            selected_by_id[unique_id] = row

        for row in baseline_models:
            unique_id = str(row["unique_id"]).strip()

            if unique_id in baseline_by_id:
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Baseline model inventory contains duplicate "
                            f"unique_id {unique_id!r}"
                        ),
                        entity_id=unique_id,
                        evidence=row,
                    )
                ]

            baseline_by_id[unique_id] = row

        evaluations: list[RuleEvaluation] = []

        for unique_id in sorted(baseline_by_id):
            baseline_model = baseline_by_id[unique_id]
            selected_model = selected_by_id.get(unique_id)

            triggered = selected_model is None

            result = "TRIGGERED" if triggered else "PASS"

            severity = rule["default_severity"] if triggered else None

            if triggered:
                reason = (
                    f"Model {unique_id!r} was present in baseline run "
                    f"{baseline_run_id!r} but is absent from selected run "
                    f"{selected_run_id!r}"
                )
            else:
                reason = (
                    f"Model {unique_id!r} is present in both baseline run "
                    f"{baseline_run_id!r} and selected run "
                    f"{selected_run_id!r}"
                )

            evaluations.append(
                RuleEvaluation(
                    rule_id=rule["rule_id"],
                    result=result,
                    severity=severity,
                    entity_type=rule["applicability"]["entity_type"],
                    entity_id=unique_id,
                    evidence_source=source,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "baseline_run_id": baseline_run_id,
                        "baseline_model": baseline_model,
                        "selected_model": selected_model,
                    },
                    reason=reason,
                )
            )

        return evaluations

    def _evaluate_row_count_anomaly_rule(
        self,
        rule: Mapping[str, Any],
        selected_run_id: str,
        comparable_run_ids: Sequence[str],
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        source = "model_metadata_snapshots"

        if not comparable_run_ids:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=(
                        "No previous comparable successful pipeline runs "
                        "are available for the row-count baseline"
                    ),
                    entity_id=None,
                    evidence={"selected_run_id": selected_run_id},
                )
            ]

        pipeline_rows = evidence.get("pipeline_runs", ())
        metadata_rows = evidence.get(source, ())

        selected_pipeline_rows = [
            dict(row)
            for row in pipeline_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        if len(selected_pipeline_rows) != 1:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Expected exactly one selected pipeline-run record "
                        f"but found {len(selected_pipeline_rows)}"
                    ),
                    entity_id=selected_run_id,
                    evidence={"selected_run_id": selected_run_id},
                )
            ]

        selected_pipeline = selected_pipeline_rows[0]
        selected_status_raw = selected_pipeline.get("status")

        if self._is_missing(selected_status_raw):
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Selected pipeline-run record is missing required "
                        "field: status"
                    ),
                    entity_id=selected_run_id,
                    evidence={"selected_run_id": selected_run_id},
                )
            ]

        selected_status = self._normalize(
            selected_status_raw,
            "trim_and_lowercase",
        )
        allowed_selected_statuses = {
            str(status).strip().lower()
            for status in rule["applicability"].get(
                "selected_run_statuses",
                [],
            )
        }

        if selected_status not in allowed_selected_statuses:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "R005 applies only to selected runs with an eligible "
                        f"status; found {selected_status!r}"
                    ),
                    entity_id=selected_run_id,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "selected_status": selected_status,
                    },
                )
            ]

        window_size = int(rule["baseline_policy"]["window_size"])
        historical_run_ids = tuple(comparable_run_ids[:window_size])
        historical_run_id_set = set(historical_run_ids)

        selected_models = [
            dict(row)
            for row in metadata_rows
            if row.get("monitoring_run_id") == selected_run_id
            and str(row.get("resource_type", "")).strip().lower() == "model"
        ]

        if not selected_models:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason="No selected-run model row-count evidence is available",
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "historical_run_ids": historical_run_ids,
                    },
                )
            ]

        selected_by_id: dict[str, dict[str, Any]] = {}

        for row in selected_models:
            unique_id_raw = row.get("unique_id")

            if self._is_missing(unique_id_raw):
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Selected model row-count evidence contains a "
                            "record with missing required field: unique_id"
                        ),
                        entity_id=None,
                        evidence=row,
                    )
                ]

            unique_id = str(unique_id_raw).strip()

            if unique_id in selected_by_id:
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Selected model row-count evidence contains "
                            f"duplicate unique_id {unique_id!r}"
                        ),
                        entity_id=unique_id,
                        evidence=row,
                    )
                ]

            selected_by_id[unique_id] = row

        historical_by_model: dict[str, list[dict[str, Any]]] = {}

        for raw_row in metadata_rows:
            if raw_row.get("monitoring_run_id") not in historical_run_id_set:
                continue

            if str(raw_row.get("resource_type", "")).strip().lower() != "model":
                continue

            row = dict(raw_row)
            unique_id_raw = row.get("unique_id")

            if self._is_missing(unique_id_raw):
                continue

            unique_id = str(unique_id_raw).strip()
            historical_by_model.setdefault(unique_id, []).append(row)

        relative_threshold = float(rule["trigger_logic"]["relative_change_threshold"])
        minimum_absolute_change = float(
            rule["trigger_logic"]["minimum_absolute_change"]
        )
        minimum_observations = int(
            rule["baseline_policy"].get(
                "minimum_observations_per_model",
                1,
            )
        )

        evaluations: list[RuleEvaluation] = []

        for unique_id in sorted(selected_by_id):
            current_model = selected_by_id[unique_id]
            current_row_count_raw = current_model.get("row_count")

            if not self._is_number(current_row_count_raw):
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=("Current model row_count is missing or non-numeric"),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model": current_model,
                        },
                    )
                )
                continue

            historical_observations: list[dict[str, Any]] = []
            seen_run_ids: set[str] = set()
            duplicate_historical_evidence = False

            for row in historical_by_model.get(unique_id, []):
                run_id_raw = row.get("monitoring_run_id")

                if self._is_missing(run_id_raw):
                    continue

                run_id = str(run_id_raw).strip()

                if run_id in seen_run_ids:
                    duplicate_historical_evidence = True
                    break

                row_count_raw = row.get("row_count")

                if not self._is_number(row_count_raw):
                    continue

                seen_run_ids.add(run_id)
                historical_observations.append(
                    {
                        "monitoring_run_id": run_id,
                        "row_count": float(row_count_raw),
                    }
                )

            if duplicate_historical_evidence:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Historical row-count evidence contains duplicate "
                            f"records for model {unique_id!r} within the same run"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model": current_model,
                        },
                    )
                )
                continue

            if len(historical_observations) < minimum_observations:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Insufficient historical row-count observations "
                            f"for model {unique_id!r}; found "
                            f"{len(historical_observations)}, require "
                            f"{minimum_observations}"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model": current_model,
                            "historical_observations": historical_observations,
                        },
                    )
                )
                continue

            baseline = median_baseline(
                observation["row_count"] for observation in historical_observations
            )

            if baseline is None:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Historical row-count baseline could not be "
                            f"calculated for model {unique_id!r}"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model": current_model,
                            "historical_observations": historical_observations,
                        },
                    )
                )
                continue

            current_row_count = float(current_row_count_raw)
            absolute_change = abs(current_row_count - baseline)

            if baseline == 0:
                relative_change = None
                triggered = absolute_change >= minimum_absolute_change
            else:
                relative_change = absolute_change / abs(baseline)
                triggered = (
                    relative_change >= relative_threshold
                    and absolute_change >= minimum_absolute_change
                )

            result = "TRIGGERED" if triggered else "PASS"
            severity = (
                self._resolve_row_count_severity(
                    rule=rule,
                    relative_change=relative_change,
                )
                if triggered
                else None
            )

            if baseline == 0:
                reason = (
                    f"Current row_count {current_row_count:g} differs from "
                    f"zero baseline by {absolute_change:g} rows; "
                    f"minimum absolute change is "
                    f"{minimum_absolute_change:g}"
                )
            else:
                reason = (
                    f"Current row_count {current_row_count:g} vs median "
                    f"baseline {baseline:g}: absolute change "
                    f"{absolute_change:g}, relative change "
                    f"{relative_change:.2%}; thresholds are "
                    f"{minimum_absolute_change:g} rows and "
                    f"{relative_threshold:.0%}"
                )

            evaluations.append(
                RuleEvaluation(
                    rule_id=rule["rule_id"],
                    result=result,
                    severity=severity,
                    entity_type=rule["applicability"]["entity_type"],
                    entity_id=unique_id,
                    evidence_source=source,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "historical_run_ids": historical_run_ids,
                        "current_model": current_model,
                        "historical_observations": historical_observations,
                        "baseline_median_row_count": baseline,
                        "baseline_sample_size": len(historical_observations),
                        "absolute_change": absolute_change,
                        "relative_change": relative_change,
                        "relative_change_threshold": relative_threshold,
                        "minimum_absolute_change": minimum_absolute_change,
                    },
                    reason=reason,
                )
            )

        return evaluations

    @staticmethod
    def _resolve_row_count_severity(
        rule: Mapping[str, Any],
        relative_change: float | None,
    ) -> str:
        severity_logic = rule.get("severity_logic")

        if not isinstance(severity_logic, Mapping):
            return rule["default_severity"]

        fallback = severity_logic.get(
            "fallback",
            rule["default_severity"],
        )

        if relative_change is None:
            return fallback

        bands = severity_logic.get("bands", [])

        if not isinstance(bands, list):
            return fallback

        sorted_bands = sorted(
            (
                band
                for band in bands
                if isinstance(band, Mapping)
                and "minimum_value" in band
                and "severity" in band
            ),
            key=lambda band: float(band["minimum_value"]),
            reverse=True,
        )

        for band in sorted_bands:
            if relative_change >= float(band["minimum_value"]):
                return str(band["severity"])

        return fallback

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _evaluate_runtime_regression_rule(
        self,
        rule: Mapping[str, Any],
        selected_run_id: str,
        comparable_run_ids: Sequence[str],
        evidence: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> list[RuleEvaluation]:
        source = "model_run_results"

        if not comparable_run_ids:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason=(
                        "No previous comparable successful pipeline runs "
                        "are available for the runtime baseline"
                    ),
                    entity_id=None,
                    evidence={"selected_run_id": selected_run_id},
                )
            ]

        pipeline_rows = evidence.get("pipeline_runs", ())
        model_rows = evidence.get(source, ())

        selected_pipeline_rows = [
            dict(row)
            for row in pipeline_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        if len(selected_pipeline_rows) != 1:
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Expected exactly one selected pipeline-run record "
                        f"but found {len(selected_pipeline_rows)}"
                    ),
                    entity_id=selected_run_id,
                    evidence={"selected_run_id": selected_run_id},
                )
            ]

        selected_pipeline_status = selected_pipeline_rows[0].get("status")

        if self._is_missing(selected_pipeline_status):
            return [
                self._not_evaluated(
                    rule=rule,
                    source="pipeline_runs",
                    reason=(
                        "Selected pipeline-run record is missing required "
                        "field: status"
                    ),
                    entity_id=selected_run_id,
                    evidence={"selected_run_id": selected_run_id},
                )
            ]

        window_size = int(rule["baseline_policy"]["window_size"])
        historical_run_ids = tuple(comparable_run_ids[:window_size])
        historical_run_id_set = set(historical_run_ids)
        historical_run_order = {
            run_id: index for index, run_id in enumerate(historical_run_ids)
        }

        current_rows = [
            dict(row)
            for row in model_rows
            if row.get("monitoring_run_id") == selected_run_id
        ]

        if not current_rows:
            return [
                self._not_evaluated(
                    rule=rule,
                    source=source,
                    reason="No current model execution evidence is available",
                    entity_id=None,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "historical_run_ids": historical_run_ids,
                    },
                )
            ]

        current_by_id: dict[str, dict[str, Any]] = {}

        for row in current_rows:
            unique_id_raw = row.get("unique_id")

            if self._is_missing(unique_id_raw):
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Current model execution evidence contains a "
                            "record with missing required field: unique_id"
                        ),
                        entity_id=None,
                        evidence=row,
                    )
                ]

            unique_id = str(unique_id_raw).strip()

            if unique_id in current_by_id:
                return [
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Current model execution evidence contains "
                            f"duplicate unique_id {unique_id!r}"
                        ),
                        entity_id=unique_id,
                        evidence=row,
                    )
                ]

            current_by_id[unique_id] = row

        historical_by_model: dict[str, list[dict[str, Any]]] = {}

        for raw_row in model_rows:
            run_id_raw = raw_row.get("monitoring_run_id")

            if run_id_raw not in historical_run_id_set:
                continue

            unique_id_raw = raw_row.get("unique_id")

            if self._is_missing(unique_id_raw):
                continue

            unique_id = str(unique_id_raw).strip()
            historical_by_model.setdefault(unique_id, []).append(dict(raw_row))

        current_allowed_statuses = {
            str(status).strip().lower()
            for status in rule["applicability"].get(
                "current_model_statuses",
                [],
            )
        }
        relative_threshold = float(rule["trigger_logic"]["relative_increase_threshold"])
        minimum_absolute_increase = float(
            rule["trigger_logic"]["minimum_absolute_increase_seconds"]
        )
        minimum_observations = int(
            rule["baseline_policy"].get(
                "minimum_observations_per_model",
                1,
            )
        )

        evaluations: list[RuleEvaluation] = []

        for unique_id in sorted(current_by_id):
            current_model = current_by_id[unique_id]

            model_name_raw = current_model.get("model_name")
            status_raw = current_model.get("status")
            runtime_raw = current_model.get("execution_time_seconds")

            if self._is_missing(model_name_raw) or self._is_missing(status_raw):
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Current model execution is missing required "
                            "model_name or status evidence"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "current_model_execution": current_model,
                        },
                    )
                )
                continue

            current_status = self._normalize(
                status_raw,
                "trim_and_lowercase",
            )

            if current_status not in current_allowed_statuses:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "R006 applies only to successful current model "
                            f"executions; found {current_status!r}"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "current_model_execution": current_model,
                        },
                    )
                )
                continue

            if not self._is_number(runtime_raw):
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Current model execution_time_seconds is missing "
                            "or non-numeric"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "current_model_execution": current_model,
                        },
                    )
                )
                continue

            historical_observations: list[dict[str, Any]] = []
            seen_run_ids: set[str] = set()
            duplicate_historical_evidence = False

            candidate_rows = sorted(
                historical_by_model.get(unique_id, []),
                key=lambda row: historical_run_order.get(
                    str(row.get("monitoring_run_id")),
                    len(historical_run_order),
                ),
            )

            for row in candidate_rows:
                run_id_raw = row.get("monitoring_run_id")
                historical_status_raw = row.get("status")
                historical_runtime_raw = row.get("execution_time_seconds")

                if self._is_missing(run_id_raw) or self._is_missing(
                    historical_status_raw
                ):
                    continue

                run_id = str(run_id_raw).strip()

                if run_id in seen_run_ids:
                    duplicate_historical_evidence = True
                    break

                historical_status = self._normalize(
                    historical_status_raw,
                    "trim_and_lowercase",
                )

                if historical_status != "success":
                    continue

                if not self._is_number(historical_runtime_raw):
                    continue

                seen_run_ids.add(run_id)
                historical_observations.append(
                    {
                        "monitoring_run_id": run_id,
                        "execution_time_seconds": float(historical_runtime_raw),
                    }
                )

            if duplicate_historical_evidence:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Historical runtime evidence contains duplicate "
                            f"records for model {unique_id!r} within the same run"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model_execution": current_model,
                        },
                    )
                )
                continue

            if len(historical_observations) < minimum_observations:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Insufficient successful historical runtime "
                            f"observations for model {unique_id!r}; found "
                            f"{len(historical_observations)}, require "
                            f"{minimum_observations}"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model_execution": current_model,
                            "historical_observations": historical_observations,
                        },
                    )
                )
                continue

            baseline = median_baseline(
                observation["execution_time_seconds"]
                for observation in historical_observations
            )

            if baseline is None or baseline <= 0:
                evaluations.append(
                    self._not_evaluated(
                        rule=rule,
                        source=source,
                        reason=(
                            "Historical median runtime baseline is zero, "
                            "negative, or unavailable"
                        ),
                        entity_id=unique_id,
                        evidence={
                            "selected_run_id": selected_run_id,
                            "historical_run_ids": historical_run_ids,
                            "current_model_execution": current_model,
                            "historical_observations": historical_observations,
                            "baseline_median_execution_time_seconds": baseline,
                        },
                    )
                )
                continue

            current_runtime = float(runtime_raw)
            absolute_increase = current_runtime - baseline
            relative_increase = absolute_increase / baseline

            triggered = (
                absolute_increase >= minimum_absolute_increase
                and relative_increase >= relative_threshold
            )

            result = "TRIGGERED" if triggered else "PASS"
            severity = (
                self._resolve_runtime_severity(
                    rule=rule,
                    relative_increase=relative_increase,
                    absolute_increase=absolute_increase,
                )
                if triggered
                else None
            )

            reason = (
                f"Current runtime {current_runtime:.3f}s vs median baseline "
                f"{baseline:.3f}s: absolute increase "
                f"{absolute_increase:.3f}s, relative increase "
                f"{relative_increase:.2%}; thresholds are "
                f"{minimum_absolute_increase:.3f}s and "
                f"{relative_threshold:.0%}"
            )

            evaluations.append(
                RuleEvaluation(
                    rule_id=rule["rule_id"],
                    result=result,
                    severity=severity,
                    entity_type=rule["applicability"]["entity_type"],
                    entity_id=unique_id,
                    evidence_source=source,
                    evidence={
                        "selected_run_id": selected_run_id,
                        "historical_run_ids": historical_run_ids,
                        "current_model_execution": current_model,
                        "historical_observations": historical_observations,
                        "baseline_median_execution_time_seconds": baseline,
                        "baseline_sample_size": len(historical_observations),
                        "absolute_increase_seconds": absolute_increase,
                        "relative_increase": relative_increase,
                        "relative_increase_threshold": relative_threshold,
                        "minimum_absolute_increase_seconds": (
                            minimum_absolute_increase
                        ),
                    },
                    reason=reason,
                )
            )

        return evaluations

    @staticmethod
    def _resolve_runtime_severity(
        rule: Mapping[str, Any],
        relative_increase: float,
        absolute_increase: float,
    ) -> str:
        severity_logic = rule.get("severity_logic")

        if not isinstance(severity_logic, Mapping):
            return rule["default_severity"]

        fallback = severity_logic.get(
            "fallback",
            rule["default_severity"],
        )
        bands = severity_logic.get("bands", [])

        if not isinstance(bands, list):
            return fallback

        sorted_bands = sorted(
            (
                band
                for band in bands
                if isinstance(band, Mapping)
                and "minimum_relative_increase" in band
                and "minimum_absolute_increase_seconds" in band
                and "severity" in band
            ),
            key=lambda band: (
                float(band["minimum_relative_increase"]),
                float(band["minimum_absolute_increase_seconds"]),
            ),
            reverse=True,
        )

        for band in sorted_bands:
            if relative_increase >= float(
                band["minimum_relative_increase"]
            ) and absolute_increase >= float(band["minimum_absolute_increase_seconds"]):
                return str(band["severity"])

        return str(fallback)

    def _evaluate_row(
        self,
        rule: Mapping[str, Any],
        source: str,
        evidence_definition: Mapping[str, Any],
        row: dict[str, Any],
    ) -> RuleEvaluation:
        required_fields = self._required_fields(evidence_definition)
        missing_fields = [
            field for field in required_fields if self._is_missing(row.get(field))
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
                    "Missing required evidence fields: " f"{sorted(missing_fields)}"
                ),
                entity_id=entity_id,
                evidence=row,
            )

        trigger_logic = rule["trigger_logic"]
        operator = trigger_logic["operator"]

        if operator != "not_equals":
            raise EvaluationError(
                f"Unsupported operator for {rule['rule_id']}: " f"{operator}"
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
        severity = self._resolve_severity(rule, actual_value) if triggered else None

        reason = (
            f"Normalized value '{actual_value}' does not equal " f"'{comparison_value}'"
            if triggered
            else (f"Normalized value '{actual_value}' equals " f"'{comparison_value}'")
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

        value_requirements = evidence_definition.get("value_requirements")

        if isinstance(value_requirements, dict):
            return list(value_requirements)

        raise EvaluationError(
            "Required evidence must define either 'fields' or " "'value_requirements'"
        )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    @staticmethod
    def _normalize(value: Any, normalization: str | None) -> Any:
        if normalization is None:
            return value

        if normalization == "trim_and_lowercase":
            if not isinstance(value, str):
                raise EvaluationError("trim_and_lowercase requires a string value")
            return value.strip().lower()

        raise EvaluationError(f"Unsupported normalization: {normalization}")

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
