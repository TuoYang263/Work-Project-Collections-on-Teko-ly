
"""Verified local backup and recovery of incident snapshots."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts.diagnostics.incident_publication import (
    verify_published_snapshot,
)


MAX_ARCHIVE_BYTES = 3 * 1024 * 1024
COPY_CHUNK_BYTES = 64 * 1024


def _private_root(root: Path) -> None:
    """Require an existing private directory."""

    if (
        root.is_symlink()
        or not root.is_dir()
        or root.stat().st_mode & 0o077
    ):
        raise PermissionError(
            "Destination root must be an existing private directory"
        )


def _copy_verified_revision(
    source_archive: Path,
    destination_root: Path,
    *,
    expected_snapshot_sha256: str,
    expected_manifest_sha256: str,
    dag_id: str,
    run_id: str,
    revision: int,
) -> dict[str, Any]:
    """Copy verified bytes without overwriting a revision."""

    source = Path(source_archive)
    destination_root = Path(destination_root)

    # 1. Verify the original archive first.
    verify_args = {
        "expected_snapshot_sha256": expected_snapshot_sha256,
        "expected_manifest_sha256": expected_manifest_sha256,
        "dag_id": dag_id,
        "run_id": run_id,
        "revision": revision,
    }

    verify_published_snapshot(
        source,
        **verify_args,
    )

    # 2. Prepare a private destination.
    _private_root(destination_root)

    case_dir = destination_root / run_id
    case_dir.mkdir(mode=0o700, exist_ok=True)

    _private_root(case_dir)

    final_path = (
        case_dir / f"revision-{revision:04d}.zip"
    )

    # 3. Create an independent destination file.
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".backup-staging-",
        suffix=".zip",
        dir=case_dir,
    )

    staged = Path(temporary_path)
    published = False

    try:
        digest = hashlib.sha256()
        copied = 0

        # 4. Copy in bounded chunks.
        with os.fdopen(descriptor, "wb") as output:
            with source.open("rb") as input_file:
                while chunk := input_file.read(
                    COPY_CHUNK_BYTES
                ):
                    copied += len(chunk)

                    if copied > MAX_ARCHIVE_BYTES:
                        raise ValueError(
                            "Snapshot copy exceeds size limit"
                        )

                    digest.update(chunk)
                    output.write(chunk)

            output.flush()
            os.fsync(output.fileno())

        # 5. Detect changes during copying.
        if digest.hexdigest() != expected_snapshot_sha256:
            raise ValueError(
                "Copied snapshot checksum mismatch"
            )

        # 6. Publish without replacing an existing revision.
        os.link(staged, final_path)
        published = True

        # 7. Re-verify the destination.
        verification = verify_published_snapshot(
            final_path,
            **verify_args,
        )

        return {
            "archive_path": str(final_path),
            "snapshot_sha256": expected_snapshot_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "verification_status": verification[
                "verification_status"
            ],
        }

    except Exception:
        if published:
            final_path.unlink(missing_ok=True)
        raise

    finally:
        staged.unlink(missing_ok=True)


def backup_verified_snapshot(
    archive_path: Path,
    *,
    backup_root: Path,
    expected_snapshot_sha256: str,
    expected_manifest_sha256: str,
    dag_id: str,
    run_id: str,
    revision: int,
) -> dict[str, Any]:
    """Create a verified local backup copy."""

    result = _copy_verified_revision(
        archive_path,
        backup_root,
        expected_snapshot_sha256=expected_snapshot_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        dag_id=dag_id,
        run_id=run_id,
        revision=revision,
    )

    return {
        "backup_path": result["archive_path"],
        "snapshot_sha256": result["snapshot_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "backup_status": "LOCAL_COPY_VERIFIED",
    }


def restore_verified_snapshot(
    backup_path: Path,
    *,
    restore_root: Path,
    expected_snapshot_sha256: str,
    expected_manifest_sha256: str,
    dag_id: str,
    run_id: str,
    revision: int,
) -> dict[str, Any]:
    """Restore a verified backup without overwriting."""

    result = _copy_verified_revision(
        backup_path,
        restore_root,
        expected_snapshot_sha256=expected_snapshot_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        dag_id=dag_id,
        run_id=run_id,
        revision=revision,
    )

    return {
        "restored_path": result["archive_path"],
        "snapshot_sha256": result["snapshot_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "restore_status": "LOCAL_RESTORE_VERIFIED",
    }