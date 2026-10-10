from __future__ import annotations

import hashlib
import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RUN_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "control"
    / "run_evidence"
)

TERMINAL_STATUSES = {
    "SUCCEEDED",
    "FAILED",
}


def _utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _record_path(
    run_id: str,
    directory: Path,
) -> Path:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(
            "run_id must be a non-empty string"
        )

    digest = hashlib.sha256(
        run_id.encode("utf-8")
    ).hexdigest()

    return directory / f"{digest}.json"


def _write_atomic(
    path: Path,
    record: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                dict(record),
                handle,
                sort_keys=True,
                indent=2,
            )

            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temp_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass

        raise


def start_run_evidence(
    run_id: str,
    execution_contract: Mapping[str, Any],
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
    started_at: str | None = None,
) -> dict[str, Any]:
    required_contract_fields = {
        "mode",
        "logical_window",
        "read_scope",
        "write_scope",
        "rerun_semantics",
    }

    missing_fields = sorted(
        required_contract_fields
        - set(execution_contract.keys())
    )

    if missing_fields:
        raise ValueError(
            "Execution contract missing fields: "
            f"{missing_fields}"
        )

    path = _record_path(
        run_id,
        directory,
    )

    if path.exists():
        raise RuntimeError(
            f"Run evidence already exists: {run_id}"
        )

    record = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": execution_contract["mode"],
        "execution_scope":
            execution_contract.get(
                "execution_scope"
            ),
        "logical_window":
            dict(
                execution_contract[
                    "logical_window"
                ]
            ),
        "read_scope":
            dict(
                execution_contract[
                    "read_scope"
                ]
            ),
        "write_scope":
            execution_contract["write_scope"],
        "rerun_semantics":
            execution_contract[
                "rerun_semantics"
            ],
        "status": "RUNNING",
        "started_at":
            started_at or _utc_now_iso(),
        "finished_at": None,
        "failure_stage": None,
    }

    _write_atomic(
        path,
        record,
    )

    return record


def load_run_evidence(
    run_id: str,
    *,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
) -> dict[str, Any]:
    path = _record_path(
        run_id,
        directory,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Run evidence not found: {run_id}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def finalize_run_evidence(
    run_id: str,
    status: str,
    *,
    failure_stage: str | None = None,
    directory: Path = DEFAULT_RUN_EVIDENCE_DIR,
    finished_at: str | None = None,
) -> dict[str, Any]:
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            f"Unsupported terminal status: {status}"
        )

    if (
        status == "FAILED"
        and not failure_stage
    ):
        raise ValueError(
            "FAILED run requires failure_stage"
        )

    record = load_run_evidence(
        run_id,
        directory=directory,
    )

    if record["status"] != "RUNNING":
        raise RuntimeError(
            "Run evidence is already terminal: "
            f"{record['status']}"
        )

    record["status"] = status
    record["finished_at"] = (
        finished_at or _utc_now_iso()
    )

    record["failure_stage"] = (
        failure_stage
        if status == "FAILED"
        else None
    )

    path = _record_path(
        run_id,
        directory,
    )

    _write_atomic(
        path,
        record,
    )

    return record