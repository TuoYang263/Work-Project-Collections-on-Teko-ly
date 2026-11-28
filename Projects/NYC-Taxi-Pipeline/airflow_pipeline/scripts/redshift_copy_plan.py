"""R4.1B: generate a release-bound Redshift COPY plan; never execute SQL."""

import argparse
import hashlib
import json
import re

from datetime import date
from pathlib import Path

from scripts.serving_snapshot_integrity import (
    PART_PATTERN,
    RELEASE_PATTERN,
    TABLES,
)

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{2,62}\Z")
IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,126}\Z")

SIMPLE_TYPES = {
    "TIMESTAMP",
    "DATE",
    "BIGINT",
    "INTEGER",
    "SMALLINT",
    "DOUBLE PRECISION",
    "REAL",
    "VARCHAR(65535)",
    "BOOLEAN",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def unique_object(pairs):
    result = {}

    for key, value in pairs:
        require(
            key not in result,
            f"Duplicate JSON key: {key}",
        )
        result[key] = value

    return result


def valid_type(value):
    if value in SIMPLE_TYPES:
        return True

    match = re.fullmatch(
        r"DECIMAL\(([1-9][0-9]?),(0|[1-9][0-9]?)\)",
        value,
    )

    return bool(match and 1 <= int(match[1]) <= 38 and int(match[2]) <= int(match[1]))


def parse_ddl(info, staging):
    """Accept only the canonical R1 CREATE TABLE grammar."""

    columns = info.get("columns")

    require(
        isinstance(columns, list) and bool(columns),
        "Missing columns",
    )

    require(
        all(matches(IDENTIFIER, c) for c in columns),
        "Unsafe column identifier",
    )

    require(
        len(columns) == len(set(columns)),
        "Duplicate columns",
    )

    target = f'nyc_stage."{staging}"'

    prefix = f"CREATE TABLE {target} (\n"
    suffix = "\n);"

    ddl = info.get("create_table_sql")

    require(
        isinstance(ddl, str) and ddl.startswith(prefix) and ddl.endswith(suffix),
        "Unexpected DDL target or syntax",
    )

    body = ddl[len(prefix) : -len(suffix)]

    fields = []

    for line in body.split(",\n"):

        match = re.fullmatch(
            r'  "([a-z_][a-z0-9_]*)" (.+)',
            line,
        )

        require(
            match is not None,
            "Invalid DDL column definition",
        )

        name, data_type = match.groups()

        require(
            valid_type(data_type),
            f"Unsupported Redshift type: {data_type}",
        )

        fields.append(
            {
                "name": name,
                "type": data_type,
            }
        )

    require(
        [f["name"] for f in fields] == columns,
        "DDL column order mismatch",
    )

    rebuilt = (
        prefix + ",\n".join(f'  "{f["name"]}" {f["type"]}' for f in fields) + suffix
    )

    require(
        rebuilt == ddl,
        "Non-canonical DDL",
    )

    return fields, rebuilt


def build_copy_plan(
    manifest_bytes,
    *,
    approved_sha256,
    bucket,
    release_id,
):
    """Pure function; a valid plan is not execution approval."""

    # 1. Validate the approved release identity.

    require(
        matches(SHA256, approved_sha256),
        "Invalid approved SHA256",
    )

    require(
        hashlib.sha256(manifest_bytes).hexdigest() == approved_sha256,
        "Manifest SHA256 mismatch",
    )

    require(
        matches(BUCKET, bucket),
        "Unsafe bucket",
    )

    require(
        matches(RELEASE_PATTERN, release_id),
        "Unsafe release ID",
    )

    manifest = json.loads(
        manifest_bytes,
        object_pairs_hook=unique_object,
    )

    require(
        isinstance(manifest, dict),
        "Invalid manifest",
    )

    require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 2,
        "Unsupported manifest version",
    )

    require(
        manifest.get("gold_release_id") == release_id,
        "Release ID mismatch",
    )

    # 2. Validate table inventory and source lineage.

    tables = manifest.get("tables")
    versions = manifest.get("gold_delta_versions")

    require(
        isinstance(tables, dict) and set(tables) == set(TABLES),
        "Table inventory mismatch",
    )

    require(
        isinstance(versions, dict) and set(versions) == set(TABLES),
        "Delta inventory mismatch",
    )

    require(
        all(type(v) is int and v >= 0 for v in versions.values()),
        "Invalid Delta version",
    )

    seal = manifest.get("r1_inventory_sha256")

    require(
        matches(SHA256, seal),
        "Invalid R1 inventory SHA256",
    )

    # 3. Validate the half-open logical window.

    window = manifest.get("logical_window")

    require(
        isinstance(window, dict) and window.get("anchor") == "PICKUP_DATETIME",
        "Invalid logical window anchor",
    )

    for key in ("start", "end_exclusive"):

        require(
            isinstance(window.get(key), str)
            and re.fullmatch(
                r"\d{4}-\d{2}-\d{2}",
                window[key],
            ),
            "Invalid logical window date",
        )

    require(
        date.fromisoformat(window["start"])
        < date.fromisoformat(window["end_exclusive"]),
        "Invalid logical window order",
    )

    # 4. Prepare deterministic release-specific names.

    prefix = f"s3://{bucket}/serving/releases/{release_id}"

    digest = hashlib.sha256(release_id.encode("utf-8")).hexdigest()[:12]

    result = []

    # 5. Build the loading plan for each Gold table.

    for name in TABLES:

        info = tables[name]

        require(
            isinstance(info, dict),
            f"Invalid table: {name}",
        )

        staging = f"{name}_r{digest}"

        require(
            info.get("staging_table") == staging,
            "Staging table mismatch",
        )

        rows = info.get("row_count")

        require(
            type(rows) is int and rows > 0,
            "Invalid row count",
        )

        fields, ddl = parse_ddl(info, staging)

        uri = f"{prefix}/copy_manifests/{name}.json"

        require(
            info.get("copy_manifest") == uri,
            "COPY manifest path mismatch",
        )

        files = info.get("files")

        require(
            isinstance(files, list) and bool(files),
            "Missing Parquet evidence",
        )

        seen = set()
        entries = []
        file_evidence = []

        # 6. Validate every referenced Parquet file.

        for file in files:

            require(
                isinstance(file, dict),
                "Invalid file evidence",
            )

            url = file.get("url")
            size = file.get("size_bytes")
            sha = file.get("sha256")

            base = f"{prefix}/{name}/"

            require(
                isinstance(url, str)
                and url.startswith(base)
                and matches(
                    PART_PATTERN,
                    url[len(base) :],
                ),
                "Parquet path mismatch",
            )

            require(
                url not in seen,
                "Duplicate Parquet file",
            )

            require(
                type(size) is int and size > 0 and matches(SHA256, sha),
                "Invalid file evidence",
            )

            seen.add(url)

            entries.append(
                {
                    "url": url,
                    "mandatory": True,
                    "meta": {
                        "content_length": size,
                    },
                }
            )

            file_evidence.append(
                {
                    "url": url,
                    "size_bytes": size,
                    "sha256": sha,
                }
            )

        # 7. Generate SQL without executing it.

        target = f'nyc_stage."{staging}"'

        result.append(
            {
                "table": name,
                "staging_table": staging,
                "expected_rows": rows,
                "columns": fields,
                "create_table_sql": ddl,
                "copy_manifest": uri,
                "expected_copy_manifest": {
                    "entries": entries,
                },
                "files": file_evidence,
                "copy_sql": (
                    f"COPY {target}\n"
                    f"FROM '{uri}'\n"
                    "IAM_ROLE default\n"
                    "FORMAT AS PARQUET\n"
                    "MANIFEST;"
                ),
                "count_sql": (f"SELECT COUNT(*) AS row_count " f"FROM {target};"),
            }
        )

    # 8. Return the complete execution-independent plan.

    return {
        "schema_version": 1,
        "gold_release_id": release_id,
        "bucket": bucket,
        "manifest_sha256": approved_sha256,
        "r1_inventory_sha256": seal,
        "logical_window": window,
        "gold_delta_versions": versions,
        "create_schema_sql": ("CREATE SCHEMA IF NOT EXISTS nyc_stage;"),
        "auto_execute": False,
        "tables": result,
    }


def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--approved-sha256",
        required=True,
    )

    parser.add_argument(
        "--bucket",
        required=True,
    )

    parser.add_argument(
        "--release-id",
        required=True,
    )

    args = parser.parse_args()

    plan = build_copy_plan(
        args.manifest.read_bytes(),
        approved_sha256=args.approved_sha256,
        bucket=args.bucket,
        release_id=args.release_id,
    )

    print(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
