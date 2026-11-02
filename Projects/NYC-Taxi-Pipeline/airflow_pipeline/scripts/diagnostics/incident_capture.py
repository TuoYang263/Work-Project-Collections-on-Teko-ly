
"""Prepare bounded incident evidence artifacts."""

import os
import json
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Mapping


RUN_METADATA_FIELDS = (
    "schema_version",
    "dag_id",
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "failure_stage",
)


def prepare_run_metadata(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize an allowlisted projection of Run Evidence V2."""

    if (
        record.get("schema_version") != 2
        or record.get("dag_id") != "nyc_lifecycle_probe"
        or not isinstance(record.get("run_id"), str)
        or not record["run_id"]
        or record.get("status") not in {"SUCCEEDED", "FAILED"}
    ):
        raise ValueError("Invalid historical run evidence")

    metadata = {
        field: record.get(field)
        for field in RUN_METADATA_FIELDS
    }

    payload = {
        "artifact_type": "RUN_EVIDENCE_METADATA",
        "capture_scope": "METADATA_ONLY",
        "redaction_policy": "ALLOWLIST_V1",
        "source": metadata,
    }

    content = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    ).encode("utf-8")

    return {
        "filename": "run_evidence_metadata.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def stage_metadata_artifact(
    artifact: Mapping[str, Any],
    staging_root: Path,
) -> Path:
    """Write a verified metadata artifact into private staging."""

    content = artifact.get("content")

    if (
        artifact.get("filename") != "run_evidence_metadata.json"
        or not isinstance(content, bytes)
        or not content
        or len(content) > 1024 * 1024
        or artifact.get("size_bytes") != len(content)
        or artifact.get("sha256")
        != hashlib.sha256(content).hexdigest()
    ):
        raise ValueError("Invalid prepared metadata artifact")

    root = Path(staging_root)

    if not root.is_dir() or root.is_symlink():
        raise ValueError("Invalid staging directory")

    if root.stat().st_mode & 0o077:
        raise PermissionError("Staging directory must be private")

    stage_dir = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=root)
    )

    try:
        path = stage_dir / "run_evidence_metadata.json"

        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("Staged artifact integrity mismatch")

        return stage_dir

    except Exception:
        shutil.rmtree(stage_dir)
        raise