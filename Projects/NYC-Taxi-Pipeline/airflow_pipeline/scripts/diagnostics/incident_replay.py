"""Read-only, pinned evidence replay for locally archived lifecycle incidents.

This module never contacts Airflow, PostgreSQL, AWS, or an AI model.
It does not promote locally validated evidence to SNAPSHOT_VERIFIED.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.diagnostics.incident_publication import verify_published_snapshot

MAX_INDEX_BYTES = 64 * 1024
MAX_INDEX_ENTRIES = 100
CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
DAG_ID = "nyc_lifecycle_probe"


class ReplayError(Exception):
    """A typed, safe-to-display replay failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _private_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise ReplayError("EVIDENCE_MISSING", f"{description} does not exist")
    try:
        if (path.is_symlink() or not path.is_dir()
                or path.stat().st_mode & 0o077):
            raise ReplayError("UNAVAILABLE", f"{description} is not private")
    except OSError as exc:
        raise ReplayError("UNAVAILABLE", f"Cannot inspect {description}") from exc


def read_trusted_case_index(index_path: Path) -> list[dict[str, Any]]:
    """Load an operator-protected digest registry, never from inside a ZIP.

    The index must reside in a private directory and itself be mode 0600
    (or equivalently no group/other permissions). Trust still depends on
    protecting this file from the actor who can alter the archived ZIP.
    """
    path = Path(index_path)
    _private_directory(path.parent, "Trusted index directory")
    try:
        if path.is_symlink() or not path.is_file():
            raise ReplayError("INDEX_INVALID", "Trusted index is not a regular file")
        if path.stat().st_mode & 0o077:
            raise ReplayError(
                "INDEX_INVALID", "Trusted index permissions are too broad"
            )
        with path.open("rb") as handle:
            raw = handle.read(MAX_INDEX_BYTES + 1)
    except FileNotFoundError as exc:
        raise ReplayError("EVIDENCE_MISSING", "Trusted case index is missing") from exc
    except OSError as exc:
        raise ReplayError("UNAVAILABLE", "Cannot read trusted case index") from exc

    if len(raw) > MAX_INDEX_BYTES:
        raise ReplayError("INDEX_INVALID", "Trusted case index is too large")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReplayError(
            "INDEX_INVALID", "Trusted case index is not valid JSON"
        ) from exc

    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise ReplayError("INDEX_INVALID", "Unsupported trusted index schema")
    records = data.get("cases")
    if not isinstance(records, list) or len(records) > MAX_INDEX_ENTRIES:
        raise ReplayError("INDEX_INVALID", "Invalid trusted index cases")

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    labels: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReplayError("INDEX_INVALID", "Invalid case record")
        case_id = record.get("case_id")
        revision = record.get("revision")
        dag_id = record.get("dag_id")
        label = record.get("label", case_id)
        if (not isinstance(case_id, str) or CASE_ID_PATTERN.fullmatch(case_id) is None
                or type(revision) is not int or revision < 1
                or dag_id != DAG_ID
                or not isinstance(label, str) or not label.strip() or len(label) > 80):
            raise ReplayError("INDEX_INVALID", "Invalid case identity or label")
        for key in ("snapshot_sha256", "manifest_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise ReplayError("INDEX_INVALID", "Invalid external SHA256 record")
        identity = (case_id, revision)
        if identity in seen:
            raise ReplayError("INDEX_INVALID", "Duplicate case revision")
        if case_id in labels and labels[case_id] != label:
            raise ReplayError("INDEX_INVALID", "Conflicting labels for one case")
        seen.add(identity)
        labels[case_id] = label
        result.append({
            "case_id": case_id,
            "revision": revision,
            "dag_id": dag_id,
            "label": label,
            "snapshot_sha256": record["snapshot_sha256"],
            "manifest_sha256": record["manifest_sha256"],
        })
    return result


def available_cases(index_path: Path) -> list[dict[str, Any]]:
    """List pinned case/revision choices without touching archived evidence."""
    grouped: dict[str, dict[str, Any]] = {}
    for entry in read_trusted_case_index(index_path):
        item = grouped.setdefault(entry["case_id"], {
            "case_id": entry["case_id"],
            "label": entry["label"],
            "revisions": [],
        })
        item["revisions"].append(entry["revision"])
    for item in grouped.values():
        item["revisions"].sort(reverse=True)
    return sorted(grouped.values(), key=lambda item: item["case_id"])


def load_verified_replay(
    *,
    archive_root: Path,
    trusted_index_path: Path,
    case_id: str,
    revision: int,
) -> dict[str, Any]:
    """Return a view built only from bytes validated in a single ZIP read.

    The existing verifier is extended with a backwards-compatible optional
    `include_verified_contents` argument. It returns the same in-memory bytes
    that were checked, preventing a verify-then-reopen race in this reader.
    """
    entries = read_trusted_case_index(trusted_index_path)
    match = next(
        (entry for entry in entries if entry["case_id"] == case_id
         and entry["revision"] == revision),
        None,
    )
    if match is None:
        raise ReplayError(
            "EVIDENCE_MISSING", "Case revision is not pinned in the trusted index"
        )

    root = Path(archive_root)
    _private_directory(root, "Incident archive root")
    index = Path(trusted_index_path)
    if index.resolve().is_relative_to(root.resolve()):
        raise ReplayError(
            "INDEX_INVALID", "Trusted digest index must be outside the archive root"
        )
    case_dir = root / case_id
    _private_directory(case_dir, "Case archive directory")
    archive_path = case_dir / f"revision-{revision:04d}.zip"
    if not archive_path.exists() and not archive_path.is_symlink():
        raise ReplayError("EVIDENCE_MISSING", "Pinned archive is missing")

    try:
        verified = verify_published_snapshot(
            archive_path,
            expected_snapshot_sha256=match["snapshot_sha256"],
            expected_manifest_sha256=match["manifest_sha256"],
            dag_id=match["dag_id"],
            run_id=match["case_id"],
            revision=match["revision"],
            include_verified_contents=True,
        )
    except FileNotFoundError as exc:
        raise ReplayError("EVIDENCE_MISSING", "Pinned archive is missing") from exc
    except ValueError as exc:
        code = (
            "EVIDENCE_CONFLICT"
            if "identity" in str(exc).lower()
            else "INTEGRITY_FAILED"
        )
        raise ReplayError(
            code, "Pinned archive failed integrity or identity verification"
        ) from exc
    except (PermissionError, OSError) as exc:
        raise ReplayError("UNAVAILABLE", "Cannot open pinned archive") from exc

    contents = verified.get("verified_contents")
    if not isinstance(contents, dict):
        raise ReplayError(
            "UNAVAILABLE", "Verifier did not return checked evidence bytes"
        )
    try:
        manifest = json.loads(contents["manifest.json"])
        metadata = json.loads(contents["run_evidence_metadata.json"])
        database = json.loads(contents["airflow_task_diagnostics.json"])
        logs = json.loads(contents["log_excerpts.json"])
        source = metadata["source"]
        evidence = database["evidence"]
        task_instances = evidence["task_instances"]
        task_failures = evidence["task_failures"]
        attempts = logs["logs"]
        if (not isinstance(manifest, dict) or not isinstance(source, dict)
                or not isinstance(task_instances, list)
                or not isinstance(task_failures, list)
                or not isinstance(attempts, list)
        ):
            raise TypeError("Invalid verified evidence shape")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ReplayError(
            "EVIDENCE_CONFLICT", "Checked evidence cannot be projected safely"
        ) from exc

    # Pass captured facts through. Do not reconstruct unavailable attempt history
    # or derive statuses from TaskFail ordering.
    return {
        "case_id": case_id,
        "label": match["label"],
        "revision": revision,
        "dag_id": match["dag_id"],
        "verification_status": verified["verification_status"],
        "capture_status": verified["capture_status"],
        "snapshot_sha256": verified["snapshot_sha256"],
        "manifest_sha256": verified["manifest_sha256"],
        "provenance": manifest.get("provenance"),
        "limitations": manifest.get("limitations", []),
        "source": source,
        "task_instances": task_instances,
        "task_failures": task_failures,
        "log_attempts": attempts,
        "manifest": manifest,
        "capture_scope": {
            "run_metadata": metadata.get("capture_scope"),
            "database": database.get("capture_scope"),
            "logs": logs.get("capture_scope"),
            "attempt_history_complete": database.get("attempt_history_complete"),
            "full_log_archived": logs.get("full_log_archived"),
            "redaction_policy": logs.get("redaction_policy"),
        },
    }
