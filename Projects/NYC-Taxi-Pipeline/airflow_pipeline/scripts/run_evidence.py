"""Atomic, per-run-locked lifecycle evidence store (legacy V1 and namespaced V2).

Linux / WSL only: fcntl advisory locks require cooperating writers and a shared
filesystem with functioning flock support (verify Docker volume semantics).
Existing V1 JSON remains byte-for-byte untouched until explicitly reconciled.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

DEFAULT_RUN_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "control" / "run_evidence"
)
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_id(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _record_path(run_id: str, directory: Path, *, dag_id: str | None = None) -> Path:
    _require_id("run_id", run_id)
    directory = Path(directory)
    if dag_id is None:
        return directory / f"{hashlib.sha256(run_id.encode('utf-8')).hexdigest()}.json"
    _require_id("dag_id", dag_id)
    identity = json.dumps([dag_id, run_id], ensure_ascii=False, separators=(",", ":"))
    return directory / "v2" / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.json"


@contextmanager
def evidence_write_lock(run_id: str, *, directory: Path = DEFAULT_RUN_EVIDENCE_DIR) -> Iterator[None]:
    """Serialize cooperating writers for a run_id, across V1 and V2 namespaces.

    Never delete .lock files: unlinking allows concurrent writers to lock different inodes.
    """
    _require_id("run_id", run_id)
    directory = Path(directory)
    lock_dir = directory / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{hashlib.sha256(run_id.encode('utf-8')).hexdigest()}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(record), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def start_run_evidence(
    run_id: str,
    execution_contract: Mapping[str, Any],
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
    started_at: str | None = None,
    dag_id: str | None = None,
) -> dict[str, Any]:
    required = {"mode", "logical_window", "read_scope", "write_scope", "rerun_semantics"}
    missing = sorted(required - set(execution_contract.keys()))
    if missing:
        raise ValueError(f"Execution contract missing fields: {missing}")
    path = _record_path(run_id, directory, dag_id=dag_id)
    record = {
        "schema_version": 2 if dag_id is not None else 1,
        "run_id": run_id,
        "mode": execution_contract["mode"],
        "execution_scope": execution_contract.get("execution_scope"),
        "logical_window": dict(execution_contract["logical_window"]),
        "read_scope": dict(execution_contract["read_scope"]),
        "write_scope": execution_contract["write_scope"],
        "rerun_semantics": execution_contract["rerun_semantics"],
        "status": "RUNNING",
        "started_at": started_at or _utc_now_iso(),
        "finished_at": None,
        "failure_stage": None,
    }
    if dag_id is not None:
        record["dag_id"] = dag_id
    with evidence_write_lock(run_id, directory=directory):
        if path.exists():
            raise RuntimeError(f"Run evidence already exists: {run_id}")
        # An old record cannot be proven to belong to a particular DAG.
        # Refuse a silent cross-version overwrite/restart for the same run_id.
        if dag_id is not None and _record_path(run_id, directory).exists():
            raise RuntimeError(f"Legacy run evidence already exists: {run_id}")
        _write_atomic(path, record)
    return record


def load_run_evidence(
    run_id: str,
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
    dag_id: str | None = None,
) -> dict[str, Any]:
    path = _record_path(run_id, directory, dag_id=dag_id)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Run evidence not found: {run_id}") from None
    if not isinstance(record, dict) or record.get("run_id") != run_id:
        raise RuntimeError("Run evidence identity mismatch")
    if dag_id is not None:
        if record.get("schema_version") != 2 or record.get("dag_id") != dag_id:
            raise RuntimeError("Run evidence identity mismatch")
    else:
        if record.get("schema_version") != 1 or "dag_id" in record:
            raise RuntimeError("Legacy run evidence identity mismatch")
    return record


def _finalize_locked(
    run_id: str,
    status: str,
    *,
    failure_stage: str | None,
    directory: Path,
    finished_at: str | None,
    dag_id: str | None,
) -> dict[str, Any]:
    """Caller MUST hold evidence_write_lock(run_id, directory=...)."""
    record = load_run_evidence(run_id, directory=directory, dag_id=dag_id)
    if record.get("status") != "RUNNING":
        raise RuntimeError(f"Run evidence is already terminal: {record.get('status')}")
    record["status"] = status
    record["finished_at"] = finished_at or _utc_now_iso()
    record["failure_stage"] = failure_stage if status == "FAILED" else None
    _write_atomic(_record_path(run_id, directory, dag_id=dag_id), record)
    return record


def finalize_run_evidence(
    run_id: str,
    status: str,
    *,
    failure_stage: str | None = None,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
    finished_at: str | None = None,
    dag_id: str | None = None,
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Unsupported terminal status: {status}")
    if status == "FAILED" and not failure_stage:
        raise ValueError("FAILED run requires failure_stage")
    with evidence_write_lock(run_id, directory=directory):
        return _finalize_locked(
            run_id, status, failure_stage=failure_stage,
            directory=directory, finished_at=finished_at, dag_id=dag_id,
        )
