import pytest
from scripts.run_reconciliation import decide_reconciliation

@pytest.mark.parametrize("dag_state, evidence, expected", [
    ("queued", "RUNNING", "WAIT"),
    ("running", "RUNNING", "WAIT"),
    ("success", "RUNNING", "FINALIZE_SUCCESS"),
    ("failed", "RUNNING", "FINALIZE_FAILURE"),
    ("success", "SUCCEEDED", "ALREADY_CONSISTENT"),
    ("failed", "FAILED", "ALREADY_CONSISTENT"),
    ("failed", None, "MISSING_EVIDENCE"),
    ("success", None, "MISSING_EVIDENCE"),
])
def test_decisions(dag_state, evidence, expected):
    assert decide_reconciliation(dag_state, evidence) == expected

@pytest.mark.parametrize("dag_state, evidence", [
    ("success", "FAILED"),
    ("failed", "SUCCEEDED"),
    ("running", "SUCCEEDED"),
])
def test_conflicting_states_fail_closed(dag_state, evidence):
    with pytest.raises(RuntimeError):
        decide_reconciliation(dag_state, evidence)

@pytest.mark.parametrize("dag_state, evidence", [
    ("unknown", "RUNNING"),
    ("success", "CORRUPT"),
])
def test_invalid_states_rejected(dag_state, evidence):
    with pytest.raises(ValueError):
        decide_reconciliation(dag_state, evidence)
