"""Airflow-authoritative inspection and explicitly authorized evidence repair.

`python -m scripts.reconciliation_airflow --dag-id ... --run-id ...` is read-only.
Add `--apply` only after reviewing the dry-run report. Legacy V1 repairs
additionally require `--allow-legacy` because V1 did not store dag_id.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from airflow.models import DagRun, TaskInstance
from airflow.utils.session import create_session

from scripts.run_evidence import (
    DEFAULT_RUN_EVIDENCE_DIR,
    _finalize_locked,
    evidence_write_lock,
    load_run_evidence,
)
from scripts.run_reconciliation import decide_reconciliation


def inspect_reconciliation(
    dag_id: str,
    run_id: str,
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
) -> dict[str, Any]:
    if (not isinstance(dag_id, str) or not dag_id.strip()
            or not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("dag_id and run_id are required")

    with create_session() as session:
        dag_run = (
            session.query(DagRun)
            .filter(DagRun.dag_id == dag_id, DagRun.run_id == run_id)
            .one_or_none()
        )
        if dag_run is None:
            raise LookupError(f"Airflow DAG Run not found: {dag_id}/{run_id}")
        dag_state = dag_run.state
        failed_task_rows = (
            session.query(TaskInstance.task_id)
            .filter(
                TaskInstance.dag_id == dag_id,
                TaskInstance.run_id == run_id,
                TaskInstance.state == "failed",
            )
            .all()
        )
        failed_task_ids = sorted({row[0] for row in failed_task_rows})

        # Read V2 strictly by DAG identity. V1 is only a legacy fallback.
        try:
            evidence = load_run_evidence(run_id, dag_id=dag_id, directory=directory)
            evidence_format = "v2"
        except FileNotFoundError:
            try:
                evidence = load_run_evidence(run_id, directory=directory)
                evidence_format = "v1"
            except FileNotFoundError:
                evidence = None
                evidence_format = None

        if evidence_format == "v1":
            # The V1 record contains no DAG identity; fail closed when ambiguous.
            matching_dags = (
                session.query(DagRun.dag_id)
                .filter(DagRun.run_id == run_id)
                .all()
            )
            if len(matching_dags) != 1:
                raise RuntimeError(f"Ambiguous legacy run_id across DAGs: {run_id}")

    evidence_status = evidence["status"] if evidence is not None else None
    decision = decide_reconciliation(dag_state, evidence_status)
    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "dag_state": dag_state,
        "evidence_format": evidence_format,
        "evidence_status": evidence_status,
        "failed_task_ids": failed_task_ids,
        "decision": decision,
        "applied": False,
    }


def apply_reconciliation(
    dag_id: str,
    run_id: str,
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Read metadata and conditionally finalize evidence under the shared file lock.

    We re-inspect INSIDE the lock that all our evidence writers use. This
    guarantees callback/reconciler serialization on cooperating POSIX writers.
    Does not lock out external Airflow UI clears or direct DB modification.
    """
    with evidence_write_lock(run_id, directory=directory):
        report = inspect_reconciliation(dag_id, run_id, directory=directory)
        decision = report["decision"]
        if decision not in {"FINALIZE_SUCCESS", "FINALIZE_FAILURE"}:
            return report  # WAIT, consistent, or missing: no writes

        if report["evidence_format"] == "v1" and not allow_legacy:
            raise PermissionError(
                "Legacy V1 repair needs explicit --allow-legacy; "
                "legacy JSON has no dag_id"
            )
        if report["evidence_format"] not in {"v1", "v2"}:
            raise RuntimeError("Cannot apply repair to missing evidence")

        failure_stage = None
        if decision == "FINALIZE_FAILURE":
            if not report["failed_task_ids"]:
                raise RuntimeError("Cannot reconcile FAILED run without a failed task")
            failure_stage = report["failed_task_ids"][0]

        target_status = "SUCCEEDED" if decision == "FINALIZE_SUCCESS" else "FAILED"
        dag_identity = dag_id if report["evidence_format"] == "v2" else None
        result = _finalize_locked(
            run_id,
            target_status,
            failure_stage=failure_stage,
            directory=directory,
            finished_at=None,
            dag_id=dag_identity,
        )
        return {**report, "applied": True, "status": result["status"],
                "failure_stage": result["failure_stage"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or repair NYC run evidence")
    parser.add_argument("--dag-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--apply", action="store_true", help="Perform a guarded repair")
    parser.add_argument("--allow-legacy", action="store_true", help="Explicitly permit V1 repair")
    args = parser.parse_args()
    if args.allow_legacy and not args.apply:
        parser.error("--allow-legacy only makes sense with --apply")
    report = (
        apply_reconciliation(args.dag_id, args.run_id, allow_legacy=args.allow_legacy)
        if args.apply
        else inspect_reconciliation(args.dag_id, args.run_id)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
