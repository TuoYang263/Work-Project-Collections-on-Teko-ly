from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

VALID_RESULTS = {
    "PASS",
    "TRIGGERED",
    "NOT_EVALUATED",
}

_DATASET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ReviewPersistenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedReviewSnapshot:
    review_id: str
    monitoring_run_id: str
    total_evaluations: int
    pass_count: int
    triggered_count: int
    not_evaluated_count: int
    evaluations_json: str


def prepare_review_snapshot(
    payload: Mapping[str, Any],
    *,
    id_factory: Callable[[], str] | None = None,
) -> PreparedReviewSnapshot:
    id_factory = id_factory or (lambda: str(uuid.uuid4()))

    monitoring_run_id = _require_non_empty_string(
        payload.get("monitoring_run_id"),
        "monitoring_run_id",
    )

    evaluations_raw = payload.get("evaluations")
    if not isinstance(evaluations_raw, list):
        raise ReviewPersistenceError("Review payload must contain an evaluations list.")

    finding_package = payload.get("finding_package")
    if not isinstance(finding_package, Mapping):
        raise ReviewPersistenceError(
            "Review payload must contain a finding_package object."
        )

    package_run_id = _require_non_empty_string(
        finding_package.get("monitoring_run_id"),
        "finding_package.monitoring_run_id",
    )

    if package_run_id != monitoring_run_id:
        raise ReviewPersistenceError(
            "finding_package monitoring_run_id does not match review payload."
        )

    summary = finding_package.get("summary")
    if not isinstance(summary, Mapping):
        raise ReviewPersistenceError("finding_package must contain a summary object.")

    finding_ids = _build_finding_id_lookup(
        finding_package.get("findings"),
    )

    prepared_evaluations: list[dict[str, Any]] = []
    actual_counts = {
        "PASS": 0,
        "TRIGGERED": 0,
        "NOT_EVALUATED": 0,
    }
    triggered_keys: set[tuple[str, str, str | None]] = set()

    for raw in evaluations_raw:
        if not isinstance(raw, Mapping):
            raise ReviewPersistenceError("Each evaluation must be a JSON object.")

        rule_id = _require_non_empty_string(
            raw.get("rule_id"),
            "evaluation.rule_id",
        )
        result = _require_non_empty_string(
            raw.get("result"),
            "evaluation.result",
        )
        entity_type = _require_non_empty_string(
            raw.get("entity_type"),
            "evaluation.entity_type",
        )
        evidence_source = _require_non_empty_string(
            raw.get("evidence_source"),
            "evaluation.evidence_source",
        )
        reason = _require_non_empty_string(
            raw.get("reason"),
            "evaluation.reason",
        )

        if result not in VALID_RESULTS:
            raise ReviewPersistenceError(f"Unsupported evaluation result: {result!r}")

        entity_id = raw.get("entity_id")
        if entity_id is not None and not isinstance(entity_id, str):
            raise ReviewPersistenceError(
                "evaluation.entity_id must be a string or null."
            )

        severity = raw.get("severity")
        if severity is not None and not isinstance(severity, str):
            raise ReviewPersistenceError(
                "evaluation.severity must be a string or null."
            )

        evidence = raw.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ReviewPersistenceError("evaluation.evidence must be a JSON object.")

        key = (rule_id, entity_type, entity_id)
        finding_id: str | None = None

        if result == "TRIGGERED":
            if key in triggered_keys:
                raise ReviewPersistenceError(
                    "Duplicate triggered evaluation identity: " f"{key!r}"
                )

            triggered_keys.add(key)

            finding_id = finding_ids.get(key)
            if finding_id is None:
                raise ReviewPersistenceError(
                    "Triggered evaluation has no matching deterministic "
                    f"finding_id: {key!r}"
                )

        actual_counts[result] += 1

        prepared_evaluations.append(
            {
                "evaluation_id": id_factory(),
                "finding_id": finding_id,
                "rule_id": rule_id,
                "result": result,
                "severity": severity,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "evidence_source": evidence_source,
                "evidence": dict(evidence),
                "reason": reason,
            }
        )

    if triggered_keys != set(finding_ids):
        raise ReviewPersistenceError(
            "Deterministic finding identities do not exactly match "
            "triggered evaluations."
        )

    expected_counts = {
        "total_evaluations": _require_non_negative_int(
            summary.get("total_evaluations"),
            "summary.total_evaluations",
        ),
        "PASS": _require_non_negative_int(
            summary.get("pass"),
            "summary.pass",
        ),
        "TRIGGERED": _require_non_negative_int(
            summary.get("triggered"),
            "summary.triggered",
        ),
        "NOT_EVALUATED": _require_non_negative_int(
            summary.get("not_evaluated"),
            "summary.not_evaluated",
        ),
    }

    if expected_counts["total_evaluations"] != len(prepared_evaluations):
        raise ReviewPersistenceError(
            "Summary total_evaluations does not match evaluation count."
        )

    for result in VALID_RESULTS:
        if expected_counts[result] != actual_counts[result]:
            raise ReviewPersistenceError(
                f"Summary count for {result} does not match evaluations."
            )

    return PreparedReviewSnapshot(
        review_id=id_factory(),
        monitoring_run_id=monitoring_run_id,
        total_evaluations=len(prepared_evaluations),
        pass_count=actual_counts["PASS"],
        triggered_count=actual_counts["TRIGGERED"],
        not_evaluated_count=actual_counts["NOT_EVALUATED"],
        evaluations_json=json.dumps(
            prepared_evaluations,
            ensure_ascii=False,
            default=str,
        ),
    )


def persist_review_snapshot(
    snapshot: PreparedReviewSnapshot,
    *,
    project_id: str,
    dataset_id: str,
    location: str = "EU",
    client: Any | None = None,
) -> None:
    _validate_project_id(project_id)
    _validate_dataset_id(dataset_id)

    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ReviewPersistenceError(
            "google-cloud-bigquery is required for review persistence."
        ) from exc

    client = client or bigquery.Client(
        project=project_id,
        location=location,
    )

    pipeline_runs = f"`{project_id}.{dataset_id}.pipeline_runs`"
    review_runs = f"`{project_id}.{dataset_id}.pipeline_review_runs`"
    review_evaluations = f"`{project_id}.{dataset_id}.pipeline_review_evaluations`"

    sql = f"""
    DECLARE source_row_count INT64;
    DECLARE review_job_name STRING;
    DECLARE review_environment STRING;
    DECLARE reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP();

    SET source_row_count = (
      SELECT COUNT(*)
      FROM {pipeline_runs}
      WHERE monitoring_run_id = @monitoring_run_id
    );

    ASSERT source_row_count = 1
      AS 'Expected exactly one pipeline_runs row for monitoring_run_id';

    SET (review_job_name, review_environment) = (
      SELECT AS STRUCT
        job_name,
        environment
      FROM {pipeline_runs}
      WHERE monitoring_run_id = @monitoring_run_id
      LIMIT 1
    );

    BEGIN TRANSACTION;

    INSERT INTO {review_runs}
    (
      review_id,
      monitoring_run_id,
      job_name,
      environment,
      total_evaluations,
      pass_count,
      triggered_count,
      not_evaluated_count,
      reviewed_at
    )
    VALUES
    (
      @review_id,
      @monitoring_run_id,
      review_job_name,
      review_environment,
      @total_evaluations,
      @pass_count,
      @triggered_count,
      @not_evaluated_count,
      reviewed_at
    );

    INSERT INTO {review_evaluations}
    (
      review_id,
      monitoring_run_id,
      evaluation_id,
      finding_id,
      rule_id,
      result,
      severity,
      entity_type,
      entity_id,
      evidence_source,
      evidence_json,
      reason,
      reviewed_at
    )
    SELECT
      @review_id,
      @monitoring_run_id,
      JSON_VALUE(item, '$.evaluation_id'),
      JSON_VALUE(item, '$.finding_id'),
      JSON_VALUE(item, '$.rule_id'),
      JSON_VALUE(item, '$.result'),
      JSON_VALUE(item, '$.severity'),
      JSON_VALUE(item, '$.entity_type'),
      JSON_VALUE(item, '$.entity_id'),
      JSON_VALUE(item, '$.evidence_source'),
      JSON_QUERY(item, '$.evidence'),
      JSON_VALUE(item, '$.reason'),
      reviewed_at
    FROM UNNEST(
      JSON_QUERY_ARRAY(
        PARSE_JSON(
          @evaluations_json,
          wide_number_mode => 'round'
        )
      )
    ) AS item;

    COMMIT TRANSACTION;
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "review_id",
                "STRING",
                snapshot.review_id,
            ),
            bigquery.ScalarQueryParameter(
                "monitoring_run_id",
                "STRING",
                snapshot.monitoring_run_id,
            ),
            bigquery.ScalarQueryParameter(
                "total_evaluations",
                "INT64",
                snapshot.total_evaluations,
            ),
            bigquery.ScalarQueryParameter(
                "pass_count",
                "INT64",
                snapshot.pass_count,
            ),
            bigquery.ScalarQueryParameter(
                "triggered_count",
                "INT64",
                snapshot.triggered_count,
            ),
            bigquery.ScalarQueryParameter(
                "not_evaluated_count",
                "INT64",
                snapshot.not_evaluated_count,
            ),
            bigquery.ScalarQueryParameter(
                "evaluations_json",
                "STRING",
                snapshot.evaluations_json,
            ),
        ]
    )

    try:
        client.query(
            sql,
            job_config=job_config,
            location=location,
        ).result()
    except Exception as exc:
        raise ReviewPersistenceError(
            "Failed to persist deterministic review snapshot: {exc}"
        ) from exc


def _build_finding_id_lookup(
    raw_findings: Any,
) -> dict[tuple[str, str, str | None], str]:
    if not isinstance(raw_findings, list):
        raise ReviewPersistenceError("finding_package.findings must be a list.")

    result: dict[tuple[str, str, str | None], str] = {}

    for finding in raw_findings:
        if not isinstance(finding, Mapping):
            raise ReviewPersistenceError(
                "Each deterministic finding must be an object."
            )

        finding_id = _require_non_empty_string(
            finding.get("finding_id"),
            "finding.finding_id",
        )
        rule_id = _require_non_empty_string(
            finding.get("rule_id"),
            "finding.rule_id",
        )
        entity_type = _require_non_empty_string(
            finding.get("entity_type"),
            "finding.entity_type",
        )

        entity_id = finding.get("entity_id")
        if entity_id is not None and not isinstance(entity_id, str):
            raise ReviewPersistenceError("finding.entity_id must be a string or null.")

        key = (rule_id, entity_type, entity_id)

        if key in result:
            raise ReviewPersistenceError(
                f"Duplicate deterministic finding identity: {key!r}"
            )

        result[key] = finding_id

    return result


def _require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewPersistenceError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_non_negative_int(
    value: Any,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReviewPersistenceError(f"{field_name} must be a non-negative integer.")
    return value


def _validate_project_id(value: str) -> None:
    if not _PROJECT_RE.fullmatch(value):
        raise ReviewPersistenceError(f"Invalid BigQuery project ID: {value!r}")


def _validate_dataset_id(value: str) -> None:
    if not _DATASET_RE.fullmatch(value):
        raise ReviewPersistenceError(f"Invalid BigQuery dataset ID: {value!r}")
