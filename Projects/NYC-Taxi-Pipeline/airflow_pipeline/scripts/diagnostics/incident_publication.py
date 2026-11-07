import os
import io
import re
import json
import hashlib
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime, timedelta

REQUIRED_ARTIFACTS = {
    "run_evidence_metadata.json",
    "airflow_task_diagnostics.json",
    "log_excerpts.json",
}

MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024

MAX_UNCOMPRESSED_BYTES = (
    MAX_EVIDENCE_BYTES + MAX_MANIFEST_BYTES
)

def validate_prepared_artifact(
    artifact: Mapping[str, Any],
) -> None:
    filename = artifact.get("filename")
    content = artifact.get("content")
    size_bytes = artifact.get("size_bytes")
    sha256 = artifact.get("sha256")

    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise ValueError("Invalid artifact filename")

    if not isinstance(content, bytes) or not content:
        raise ValueError("Invalid artifact content")

    if type(size_bytes) is not int or size_bytes != len(content):
        raise ValueError("Artifact size mismatch")

    actual_sha256 = hashlib.sha256(content).hexdigest()

    if sha256 != actual_sha256:
        raise ValueError("Artifact checksum mismatch")


def build_attachment_record(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a manifest record for a validated artifact."""

    validate_prepared_artifact(artifact)

    return {
        "relative_path": artifact["filename"],
        "size_bytes": artifact["size_bytes"],
        "sha256": artifact["sha256"],
    }


def build_attachment_records(
    artifacts: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build unique attachment records."""

    if not artifacts:
        raise ValueError("Snapshot has no artifacts")

    records = []
    seen = set()

    for artifact in artifacts:
        record = build_attachment_record(artifact)
        filename = record["relative_path"]

        if filename in seen:
            raise ValueError(
                f"Duplicate artifact filename: {filename}"
            )

        seen.add(filename)
        records.append(record)

    return sorted(
        records,
        key=lambda record: record["relative_path"],
    )


REQUIRED_ARTIFACTS = {
    "run_evidence_metadata.json",
    "airflow_task_diagnostics.json",
    "log_excerpts.json",
}


def validate_snapshot_identity(
    artifacts: list[Mapping[str, Any]],
    dag_id: str,
    run_id: str,
) -> None:
    """Ensure all evidence belongs to the same Airflow run."""

    records = build_attachment_records(artifacts)

    filenames = {
        record["relative_path"]
        for record in records
    }

    if filenames != REQUIRED_ARTIFACTS:
        raise ValueError("Snapshot artifact set mismatch")

    documents = {}

    for artifact in artifacts:
        try:
            document = json.loads(artifact["content"])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid snapshot JSON") from exc

        if not isinstance(document, dict):
            raise ValueError("Invalid snapshot structure")

        documents[artifact["filename"]] = document

    # 1. Run Evidence identity
    metadata = documents["run_evidence_metadata.json"]

    source = metadata.get("source")

    if (
        metadata.get("artifact_type") != "RUN_EVIDENCE_METADATA"
        or not isinstance(source, dict)
        or source.get("schema_version") != 2
        or source.get("dag_id") != dag_id
        or source.get("run_id") != run_id
    ):
        raise ValueError("Run evidence identity mismatch")

    # 2. PostgreSQL evidence identity
    database = documents["airflow_task_diagnostics.json"]
    evidence = database.get("evidence")

    if (
        database.get("artifact_type") != "AIRFLOW_TASK_DIAGNOSTICS"
        or database.get("source_run_id") != run_id
        or not isinstance(evidence, dict)
        or evidence.get("dag_id") != dag_id
        or evidence.get("run_id") != run_id
    ):
        raise ValueError("Database evidence identity mismatch")

    for key in ("task_instances", "task_failures"):
        rows = evidence.get(key)

        if not isinstance(rows, list):
            raise ValueError("Invalid database evidence rows")

        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("dag_id") != dag_id
                or row.get("run_id") != run_id
            ):
                raise ValueError("Database evidence identity mismatch")

    # 3. Airflow log identity
    logs = documents["log_excerpts.json"]

    if (
        logs.get("artifact_type") != "AIRFLOW_LOG_EXCERPTS"
        or logs.get("dag_id") != dag_id
        or logs.get("run_id") != run_id
        or logs.get("task_id") != "harmless_task"
    ):
        raise ValueError("Log evidence identity mismatch")

    entries = logs.get("logs")

    if not isinstance(entries, list) or not entries:
        raise ValueError("Invalid log evidence entries")

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid log evidence entries")

        attempt = entry.get("attempt_number")

        expected_path = (
            f"dag_id={dag_id}/run_id={run_id}/"
            f"task_id=harmless_task/attempt={attempt}.log"
        )

        if (
            type(attempt) is not int
            or attempt <= 0
            or entry.get("source_path") != expected_path
        ):
            raise ValueError("Log evidence identity mismatch")


def build_snapshot_manifest(
    artifacts: list[Mapping[str, Any]],
    *,
    dag_id: str,
    run_id: str,
    revision: int,
    prepared_at_utc: str,
) -> dict[str, Any]:
    """Prepare a manifest without claiming publication."""

    if (
        dag_id != "nyc_lifecycle_probe"
        or not isinstance(run_id, str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            run_id,
        ) is None
    ):
        raise ValueError("Invalid snapshot case identity")

    if type(revision) is not int or revision < 1:
        raise ValueError("Invalid snapshot revision")

    try:
        prepared_time = datetime.fromisoformat(
            prepared_at_utc.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid UTC preparation timestamp"
        ) from exc

    if (
        prepared_time.tzinfo is None
        or prepared_time.utcoffset() != timedelta(0)
    ):
        raise ValueError(
            "Invalid UTC preparation timestamp"
        )

    prepared_at = prepared_time.isoformat().replace(
        "+00:00", "Z"
    )

    validate_snapshot_identity(
        artifacts,
        dag_id=dag_id,
        run_id=run_id,
    )

    attachments = build_attachment_records(artifacts)

    metadata_artifact = next(
        artifact
        for artifact in artifacts
        if artifact["filename"] == "run_evidence_metadata.json"
    )

    metadata = json.loads(metadata_artifact["content"])
    source = metadata["source"]

    status = source.get("status")
    failure_stage = source.get("failure_stage")

    if status not in {"SUCCEEDED", "FAILED"}:
        raise ValueError("Invalid source run status")

    if status == "SUCCEEDED" and failure_stage is not None:
        raise ValueError("Inconsistent source run outcome")

    if status == "FAILED" and (
        not isinstance(failure_stage, str)
        or not failure_stage
    ):
        raise ValueError("Inconsistent source run outcome")

    return {
        "schema_version": 1,
        "case_id": run_id,
        "revision": revision,
        "provenance": "OBSERVED_TEST_RUN",
        "source_dag_id": dag_id,
        "source_run_id": run_id,
        "source_status": status,
        "source_started_at": source.get("started_at"),
        "source_finished_at": source.get("finished_at"),
        "source_failure_stage": failure_stage,
        "prepared_at_utc": prepared_at,
        "captured_at_utc": None,
        "source_commit": None,
        "capture_status": "PREPARED_NOT_PUBLISHED",
        "snapshot_sha256": None,
        "attachments": attachments,
        "limitations": [
            "Not published, independently backed up, "
            "or verified for offline replay."
        ],
    }


def manifest_bytes_and_digest(
    manifest: Mapping[str, Any],
) -> tuple[bytes, str]:
    """Encode a manifest deterministically and hash its bytes."""

    content = (
        json.dumps(
            dict(manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")

    digest = hashlib.sha256(content).hexdigest()

    return content, digest


def publish_snapshot(
    artifacts: list[Mapping[str, Any]],
    *,
    dag_id: str,
    run_id: str,
    revision: int,
    prepared_at_utc: str,
    archive_root: Path,
) -> dict[str, Any]:
    """Publish a validated ZIP without overwriting an existing revision."""

    # 1. Build and validate the snapshot manifest.
    manifest = build_snapshot_manifest(
        artifacts,
        dag_id=dag_id,
        run_id=run_id,
        revision=revision,
        prepared_at_utc=prepared_at_utc,
    )

    # 2. Keep the snapshot bounded.
    total_size = sum(
        len(artifact["content"])
        for artifact in artifacts
    )

    if total_size > 2 * 1024 * 1024:
        raise ValueError("Snapshot size limit exceeded")

    manifest["capture_status"] = "PUBLISHED_PENDING_BACKUP"
    manifest["limitations"] = [
        "Local publication only; independent backup "
        "and offline replay not verified."
    ]

    manifest_bytes, manifest_sha256 = manifest_bytes_and_digest(
        manifest
    )

    # 3. Require a private archive root.
    root = Path(archive_root)

    if (
        root.is_symlink()
        or not root.is_dir()
        or root.stat().st_mode & 0o077
    ):
        raise PermissionError(
            "Archive root must be a private directory"
        )

    # 4. Prepare the case-specific directory.
    case_dir = root / run_id
    case_dir.mkdir(mode=0o700, exist_ok=True)

    if (
        case_dir.is_symlink()
        or not case_dir.is_dir()
        or case_dir.stat().st_mode & 0o077
    ):
        raise PermissionError(
            "Case archive directory must be private"
        )

    final_path = case_dir / f"revision-{revision:04d}.zip"

    # 5. Create a temporary file in the same directory.
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".staging-",
        suffix=".zip",
        dir=case_dir,
    )

    try:
        # 6. Write all evidence and the manifest.
        with os.fdopen(descriptor, "wb") as handle:
            with zipfile.ZipFile(handle, mode="w") as bundle:
                entries = [
                    (artifact["filename"], artifact["content"])
                    for artifact in artifacts
                ]

                entries.append(
                    ("manifest.json", manifest_bytes)
                )

                for filename, content in sorted(entries):
                    info = zipfile.ZipInfo(
                        filename=filename,
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )

                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100600 << 16

                    bundle.writestr(info, content)

            # Flush while the underlying file is still open.
            handle.flush()
            os.fsync(handle.fileno())

        # 7. Verify the staged ZIP before publication.
        with zipfile.ZipFile(temporary_path) as bundle:
            expected_names = {
                filename
                for filename, _ in entries
            }

            if set(bundle.namelist()) != expected_names:
                raise ValueError(
                    "Snapshot ZIP contents mismatch"
                )

            for filename, content in entries:
                if bundle.read(filename) != content:
                    raise ValueError(
                        "Snapshot ZIP integrity mismatch"
                    )

        # 8. Calculate the final archive fingerprint.
        snapshot_sha256 = hashlib.sha256(
            Path(temporary_path).read_bytes()
        ).hexdigest()

        # 9. Publish atomically without replacing an existing file.
        os.link(temporary_path, final_path)

        return {
            "archive_path": str(final_path),
            "snapshot_sha256": snapshot_sha256,
            "manifest_sha256": manifest_sha256,
            "capture_status": "PUBLISHED_PENDING_BACKUP",
        }

    finally:
        # 10. Clean up temporary files, including on failure.
        Path(temporary_path).unlink(missing_ok=True)



def verify_published_snapshot(
    archive_path: Path,
    *,
    expected_snapshot_sha256: str,
    expected_manifest_sha256: str,
    dag_id: str,
    run_id: str,
    revision: int,
    include_verified_contents: bool = False,
) -> dict[str, Any]:
    """Verify a local archive against trusted external digests."""

    # 1. Validate the requested snapshot identity.
    if (
        dag_id != "nyc_lifecycle_probe"
        or not isinstance(run_id, str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            run_id,
        ) is None
        or type(revision) is not int
        or revision < 1
    ):
        raise ValueError("Invalid snapshot identity")

    for digest in (
        expected_snapshot_sha256,
        expected_manifest_sha256,
    ):
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Invalid expected SHA256")

    # 2. Validate the local archive path.
    path = Path(archive_path)

    if (
        path.is_symlink()
        or not path.is_file()
        or path.parent.name != run_id
        or path.name != f"revision-{revision:04d}.zip"
    ):
        raise ValueError("Invalid or missing snapshot path")

    # 3. Read bounded bytes, then verify the whole archive.
    max_archive_bytes = 3 * 1024 * 1024

    with path.open("rb") as handle:
        raw = handle.read(max_archive_bytes + 1)

    if len(raw) > max_archive_bytes:
        raise ValueError(
            "Snapshot archive size limit exceeded"
        )

    snapshot_digest = hashlib.sha256(raw).hexdigest()

    if snapshot_digest != expected_snapshot_sha256:
        raise ValueError("Snapshot checksum mismatch")

    # 4. Check ZIP structure and read its contents.
    expected_names = REQUIRED_ARTIFACTS | {"manifest.json"}

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]

            if (
                len(names) != len(expected_names)
                or set(names) != expected_names
            ):
                raise ValueError(
                    "Snapshot ZIP entries mismatch"
                )

            max_uncompressed = 2 * 1024 * 1024 + 64 * 1024

            if sum(info.file_size for info in infos) > max_uncompressed:
                raise ValueError(
                    "Snapshot uncompressed size limit exceeded"
                )

            if any(
                info.is_dir() or info.flag_bits & 1
                for info in infos
            ):
                raise ValueError("Unexpected ZIP entry type")

            contents = {
                info.filename: bundle.read(info)
                for info in infos
            }

    except (
        zipfile.BadZipFile,
        RuntimeError,
        EOFError,
        NotImplementedError,
    ) as exc:
        raise ValueError("Invalid snapshot ZIP") from exc

    # 5. Verify manifest digest and canonical encoding.
    manifest_bytes = contents.pop("manifest.json")

    manifest_digest = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    if manifest_digest != expected_manifest_sha256:
        raise ValueError("Manifest checksum mismatch")

    try:
        manifest = json.loads(manifest_bytes)
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError("Invalid manifest JSON") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Invalid manifest structure")

    canonical_bytes, _ = manifest_bytes_and_digest(
        manifest
    )

    if canonical_bytes != manifest_bytes:
        raise ValueError("Non-canonical manifest")

    # 6. Rebuild attachment metadata from actual ZIP bytes.
    artifacts = [
        {
            "filename": filename,
            "content": content,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for filename, content in contents.items()
    ]

    if manifest.get("attachments") != build_attachment_records(
        artifacts
    ):
        raise ValueError("Manifest attachments mismatch")

    # 7. Revalidate the evidence identity.
    validate_snapshot_identity(
        artifacts,
        dag_id=dag_id,
        run_id=run_id,
    )

    source = json.loads(
        contents["run_evidence_metadata.json"]
    )["source"]

    if (
        manifest.get("schema_version") != 1
        or manifest.get("case_id") != run_id
        or manifest.get("revision") != revision
        or manifest.get("provenance") != "OBSERVED_TEST_RUN"
        or manifest.get("source_dag_id") != dag_id
        or manifest.get("source_run_id") != run_id
        or manifest.get("source_status") != source.get("status")
        or manifest.get("source_started_at") != source.get("started_at")
        or manifest.get("source_finished_at") != source.get("finished_at")
        or manifest.get("source_failure_stage")
            != source.get("failure_stage")
        or manifest.get("capture_status")
            != "PUBLISHED_PENDING_BACKUP"
        or manifest.get("snapshot_sha256") is not None
    ):
        raise ValueError(
            "Manifest identity or status mismatch"
        )

    # 8. Reuse the very same validated bytes for offline replay.
    result = {
        "archive_path": str(path),
        "case_id": run_id,
        "revision": revision,
        "verification_status": "LOCAL_INTEGRITY_VERIFIED",
        "capture_status": manifest["capture_status"],
        "snapshot_sha256": snapshot_digest,
        "manifest_sha256": manifest_digest,
    }
    if include_verified_contents:
        result["verified_contents"] = {
            **contents,
            "manifest.json": manifest_bytes,
        }
    return result
