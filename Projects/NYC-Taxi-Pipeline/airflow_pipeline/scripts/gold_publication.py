
"""Local Gold release activation prototype."""

import json
import os
import re
import tempfile
from pathlib import Path
from delta.tables import DeltaTable

from scripts.gold_quality_gate import (
    GOLD_TABLE_CONTRACTS,
    validate_all_gold_tables,
)


SUPPORTED_MANIFEST_SCHEMA_VERSION = 1


def activate_gold_release(
    spark,
    *,
    release_id: str,
    candidate_root: str,
    active_manifest: str,
    execution_contract: dict,
) -> dict:
    """Validate candidate tables before switching active manifest."""

    if not isinstance(release_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
        release_id,
    ):
        raise ValueError("Invalid release_id")

    if execution_contract.get("write_scope") != "FULL_TABLE_OVERWRITE":
        raise ValueError(
            "G4 local prototype supports FULL_TABLE_OVERWRITE only"
        )

    root = Path(candidate_root).resolve(strict=True)
    manifest_path = Path(active_manifest).resolve()

    table_paths = {
        name: str(root / name)
        for name in GOLD_TABLE_CONTRACTS
    }

    for name, path in table_paths.items():
        if not (Path(path) / "_delta_log").is_dir():
            raise ValueError(
                f"Candidate Delta table missing: {name}"
            )

    # Never reuse the currently published table paths.
    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        previous_paths = previous.get("table_paths", {})

        if set(previous_paths.values()) & set(table_paths.values()):
            raise ValueError(
                "Candidate reuses active Gold table paths"
            )

    # Capture candidate versions before validation.
    versions_before = _read_delta_versions(
        spark, table_paths
    )

    # Any Quality Gate failure prevents activation.
    validate_all_gold_tables(
        spark=spark,
        table_paths=table_paths,
        execution_contract=execution_contract,
        table_versions=versions_before,
    )

    # Detect candidate changes during validation.
    versions_after = _read_delta_versions(
        spark, table_paths
    )

    if versions_before != versions_after:
        raise RuntimeError(
            "Gold candidate changed during validation"
        )

    manifest = {
        "schema_version": SUPPORTED_MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "logical_window": execution_contract["logical_window"],
        "table_paths": table_paths,
        "table_versions": versions_before,
    }

    # Write a temporary file on the same filesystem,
    # then atomically replace the active pointer.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=".gold-active-",
        suffix=".tmp",
        dir=str(manifest_path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_name, manifest_path)

    finally:
        Path(temp_name).unlink(missing_ok=True)

    return manifest


def _read_delta_versions(spark, table_paths: dict) -> dict:
    """Read the latest committed version of each Delta table."""
    versions = {}

    for name, path in table_paths.items():
        row = (
            DeltaTable.forPath(spark, path)
            .history(1)
            .select("version")
            .first()
        )

        if row is None:
            raise ValueError(
                f"Missing Delta version: {name}"
            )

        versions[name] = int(row["version"])

    return versions


def read_active_gold_table(
    spark,
    *,
    active_manifest: str,
    table_name: str,
):
    """Read the published Delta version from the active manifest."""

    # 1. Load the active release manifest.
    manifest_path = Path(active_manifest)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if not isinstance(manifest, dict):
        raise ValueError("Invalid Gold active manifest")

    if (
        manifest.get("schema_version")
        != SUPPORTED_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported manifest schema version")

    # 2. Validate the requested Gold table.
    if table_name not in GOLD_TABLE_CONTRACTS:
        raise ValueError(f"Unknown Gold table: {table_name}")

    paths = manifest.get("table_paths")
    versions = manifest.get("table_versions")

    if not isinstance(paths, dict) or not isinstance(
        versions, dict
    ):
        raise ValueError("Manifest missing table paths or versions")

    path = paths.get(table_name)
    version = versions.get(table_name)

    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Missing published path: {table_name}")

    if type(version) is not int or version < 0:
        raise ValueError(f"Invalid published version: {table_name}")

    # 3. Read the certified version, not the latest version.
    return (
        spark.read.format("delta")
        .option("versionAsOf", version)
        .load(path)
    )
