
import pytest

from scripts.run_reconciliation import (
    decide_reconciliation,
)


def test_running_dag_must_wait():
    assert decide_reconciliation(
        "running", "RUNNING"
    ) == "WAIT"


def test_success_can_finalize_running_evidence():
    assert decide_reconciliation(
        "success", "RUNNING"
    ) == "FINALIZE_SUCCESS"


def test_failure_can_finalize_running_evidence():
    assert decide_reconciliation(
        "failed", "RUNNING"
    ) == "FINALIZE_FAILURE"


def test_consistent_terminal_state_is_noop():
    assert decide_reconciliation(
        "success", "SUCCEEDED"
    ) == "ALREADY_CONSISTENT"


def test_conflicting_terminal_state_rejected():
    with pytest.raises(
        RuntimeError,
        match="Conflicting terminal states",
    ):
        decide_reconciliation(
            "failed", "SUCCEEDED"
        )


def test_missing_evidence_requires_review():
    assert decide_reconciliation(
        "success", None
    ) == "MISSING_EVIDENCE"
