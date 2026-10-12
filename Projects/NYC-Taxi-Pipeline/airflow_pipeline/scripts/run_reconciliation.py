
from __future__ import annotations


def decide_reconciliation(
    dag_state: str,
    evidence_status: str | None,
) -> str:
    valid_dag_states = {
        "queued",
        "running",
        "success",
        "failed",
    }

    if dag_state not in valid_dag_states:
        raise ValueError(
            f"Unsupported DAG state: {dag_state}"
        )

    if evidence_status is None:
        return "MISSING_EVIDENCE"

    valid_evidence_statuses = {
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
    }

    if evidence_status not in valid_evidence_statuses:
        raise ValueError(
            f"Unsupported evidence status: {evidence_status}"
        )

    if dag_state in {"queued", "running"}:
        if evidence_status != "RUNNING":
            raise RuntimeError(
                "Non-terminal DAG has terminal evidence"
            )

        return "WAIT"

    expected_status = (
        "SUCCEEDED"
        if dag_state == "success"
        else "FAILED"
    )

    if evidence_status == expected_status:
        return "ALREADY_CONSISTENT"

    if evidence_status == "RUNNING":
        return (
            "FINALIZE_SUCCESS"
            if dag_state == "success"
            else "FINALIZE_FAILURE"
        )

    raise RuntimeError(
        "Conflicting terminal states: "
        f"DAG={dag_state}, evidence={evidence_status}"
    )
