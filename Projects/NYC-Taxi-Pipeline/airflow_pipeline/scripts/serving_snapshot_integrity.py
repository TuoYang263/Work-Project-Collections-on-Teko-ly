"""Seal an R1 export with exact, streamed Parquet-file evidence.

This module has no AWS dependency and imports Spark only for legacy sealing.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

TABLES = (
    "trip_summary_hourly",
    "zone_summary_pickup_daily",
    "zone_summary_dropoff_daily",
    "flow_imbalance_daily",
    "airport_economics_daily",
    "od_corridor_daily",
)
RELEASE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
PART_PATTERN = re.compile(r"part-[A-Za-z0-9_.-]+\.parquet\Z")
BUFFER_SIZE = 1024 * 1024
SEAL_NAME = "file_inventory.json"


def digest_file(path: Path) -> tuple[int, str]:
    """Stream a local file; memory usage independent of file size."""
    total = 0
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(BUFFER_SIZE):
            total += len(chunk)
            sha.update(chunk)
    return total, sha.hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def read_plan(release_dir: Path) -> tuple[dict, bytes]:
    path = release_dir / "export_plan.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or unsafe R1 export_plan.json")
    raw = path.read_bytes()
    plan = json.loads(raw)
    if not isinstance(plan, dict):
        raise ValueError("Invalid R1 export plan")
    release_id = release_dir.name
    if not RELEASE_PATTERN.fullmatch(release_id):
        raise ValueError("Unsafe release ID")
    if plan.get("release_id") != release_id:
        raise ValueError("R1 plan release ID mismatch")
    if not isinstance(plan.get("tables"), dict) or set(plan["tables"]) != set(TABLES):
        raise ValueError("R1 plan must contain exactly six tables")
    versions = plan.get("source_versions")
    if not isinstance(versions, dict) or set(versions) != set(TABLES):
        raise ValueError("Invalid pinned versions")
    if any(type(versions[name]) is not int or versions[name] < 0 for name in TABLES):
        raise ValueError("Invalid pinned Delta version")
    window = plan.get("logical_window")
    if not isinstance(window, dict) or not all(
        isinstance(window.get(key), str) and window[key]
        for key in ("start", "end_exclusive")
    ):
        raise ValueError("Invalid logical window")
    for name in TABLES:
        info = plan["tables"][name]
        if not isinstance(info, dict) or type(info.get("row_count")) is not int:
            raise ValueError(f"Invalid row count metadata: {name}")
        if info["row_count"] <= 0:
            raise ValueError(f"Empty R1 export: {name}")
    return plan, raw


def scan_parts(release_dir: Path, name: str) -> list[dict]:
    """Discover current files; the seal later turns these into an expected set."""
    directory = release_dir / name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Missing or unsafe Parquet directory: {name}")
    expected_dir = directory.resolve(strict=True)
    if expected_dir != (release_dir.resolve(strict=True) / name):
        raise ValueError(f"Parquet directory escaped release: {name}")
    # Search *all* .parquet files, rather than silently ignoring an odd filename.
    all_parquet = sorted(directory.glob("*.parquet"), key=lambda p: p.name)
    if not all_parquet:
        raise ValueError(f"Missing Parquet files: {name}")
    evidence = []
    for path in all_parquet:
        if (
            path.is_symlink()
            or not path.is_file()
            or not PART_PATTERN.fullmatch(path.name)
        ):
            raise ValueError(f"Unexpected Parquet file: {path}")
        size, sha = digest_file(path)
        if size <= 0:
            raise ValueError(f"Empty Parquet file: {path}")
        evidence.append({"name": path.name, "size_bytes": size, "sha256": sha})
    if len({e["name"] for e in evidence}) != len(evidence):
        raise ValueError(f"Duplicate Parquet names: {name}")
    return evidence


def build_seal(release_dir: Path) -> dict:
    plan, raw = read_plan(release_dir)
    tables = {}
    for name in TABLES:
        info = plan["tables"][name]
        directory = (release_dir / name).resolve(strict=True)
        if Path(info.get("parquet_path", "")).resolve() != directory:
            raise ValueError(f"Unexpected R1 export directory: {name}")
        tables[name] = scan_parts(release_dir, name)
    return {
        "schema_version": 1,
        "release_id": release_dir.name,
        "export_plan_sha256": hashlib.sha256(raw).hexdigest(),
        "tables": tables,
    }


def publish_seal(release_dir: Path) -> dict:
    """Write the seal last, atomically, with no replacement of prior evidence."""
    seal = build_seal(release_dir)
    payload = json_bytes(seal)
    target = release_dir / SEAL_NAME
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.read_bytes() != payload:
            raise ValueError("Existing file inventory differs: refusing overwrite")
        return seal
    fd, temp_name = tempfile.mkstemp(prefix=".inventory-", dir=str(release_dir))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link create is exclusive: prevents two sealers from overwriting.
        try:
            os.link(temp_name, target)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != payload:
                raise ValueError("Concurrent inventory publication conflict")
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return seal


def verify_seal(release_dir: Path) -> tuple[dict, dict]:
    """Reject any added, missing, truncated or modified Parquet part."""
    seal_path = release_dir / SEAL_NAME
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("R1 file inventory missing; seal before R2 upload")
    seal = json.loads(seal_path.read_bytes())
    plan, raw = read_plan(release_dir)
    if (
        not isinstance(seal, dict)
        or seal.get("schema_version") != 1
        or seal.get("release_id") != release_dir.name
        or seal.get("export_plan_sha256") != hashlib.sha256(raw).hexdigest()
        or not isinstance(seal.get("tables"), dict)
        or set(seal["tables"]) != set(TABLES)
    ):
        raise ValueError("R1 inventory header mismatch")
    for name in TABLES:
        expected = seal["tables"][name]
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Invalid sealed file list: {name}")
        actual = scan_parts(release_dir, name)
        if expected != actual:
            raise ValueError(f"Parquet inventory mismatch: {name}")
        if Path(plan["tables"][name]["parquet_path"]).resolve() != (
            release_dir / name
        ).resolve():
            raise ValueError(f"Unexpected export directory: {name}")
    return plan, seal


def seal_existing_export(spark, manifest_path: Path, release_dir: Path) -> dict:
    """For OLD R1 exports only: prove exact Spark row-multiset equality first."""
    original = manifest_path.read_bytes()
    active = json.loads(original)
    plan, _ = read_plan(release_dir)
    if (
        active.get("release_id") != release_dir.name
        or active.get("table_versions") != plan["source_versions"]
        or active.get("logical_window") != plan["logical_window"]
        or not isinstance(active.get("table_paths"), dict)
        or set(active["table_paths"]) != set(TABLES)
    ):
        raise ValueError("Active Gold publication does not match R1 export")
    for name in TABLES:
        source = (
            spark.read.format("delta")
            .option("versionAsOf", plan["source_versions"][name])
            .load(active["table_paths"][name])
        )
        exported = spark.read.parquet(plan["tables"][name]["parquet_path"])
        signature = lambda frame: [
            (f.name, f.dataType.simpleString()) for f in frame.schema.fields
        ]
        if signature(source) != signature(exported):
            raise ValueError(f"Source/export schema mismatch: {name}")
        expected_rows = plan["tables"][name]["row_count"]
        if source.count() != expected_rows or exported.count() != expected_rows:
            raise ValueError(f"Source/export row-count mismatch: {name}")
        # exceptAll preserves duplicates: equality is stronger than row-count alone.
        if source.exceptAll(exported).limit(1).count():
            raise ValueError(f"R1 data mismatch (missing rows): {name}")
        if exported.exceptAll(source).limit(1).count():
            raise ValueError(f"R1 data mismatch (extra rows): {name}")
        print(f"[R1 seal] {name}: exact data MATCH")
    if manifest_path.read_bytes() != original:
        raise RuntimeError("Active Gold Manifest changed during verification")
    return publish_seal(release_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal an existing R1 snapshot")
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--output-root", default="/opt/airflow/data/delta/serving_exports"
    )
    parser.add_argument(
        "--manifest", default="/opt/airflow/data/delta/gold/active.json"
    )
    args = parser.parse_args()
    if not RELEASE_PATTERN.fullmatch(args.release_id):
        raise ValueError("Unsafe release ID")
    release_dir = Path(args.output_root).resolve(strict=True) / args.release_id
    if (release_dir / SEAL_NAME).exists():
        verify_seal(release_dir)
        print("R1 INVENTORY ALREADY SEALED: PASS")
        return
    from scripts.delta_medallion_pipeline import get_spark

    spark = get_spark("R1 Legacy Snapshot Integrity Seal")
    try:
        seal_existing_export(spark, Path(args.manifest), release_dir)
    finally:
        spark.stop()
    print("R1 LEGACY SNAPSHOT SEAL: PASS")


if __name__ == "__main__":
    main()
