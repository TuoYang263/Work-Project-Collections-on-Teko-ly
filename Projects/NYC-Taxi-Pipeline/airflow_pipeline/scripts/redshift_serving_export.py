"""Export certified Gold Delta versions for Redshift staging.

Local-only R1: no S3 upload, Redshift connection, or cloud writes.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampNTZType,
    TimestampType,
)

from scripts import config
from scripts.delta_medallion_pipeline import get_spark
from scripts.serving_snapshot_integrity import publish_seal
from scripts.gold_quality_gate import (
    GOLD_TABLE_CONTRACTS,
    validate_gold_schema,
)


def sql_identifier(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def redshift_type(spark_type) -> str:
    """Explicit Spark -> Redshift type contract."""

    if isinstance(spark_type, TimestampNTZType):
        return "TIMESTAMP"
    if isinstance(spark_type, TimestampType):
        raise ValueError("Timezone-adjusted Spark TIMESTAMP is not allowed")
    if isinstance(spark_type, DateType):
        return "DATE"
    if isinstance(spark_type, LongType):
        return "BIGINT"
    if isinstance(spark_type, IntegerType):
        return "INTEGER"
    if isinstance(spark_type, ShortType):
        return "SMALLINT"
    if isinstance(spark_type, ByteType):
        return "SMALLINT"
    if isinstance(spark_type, DoubleType):
        return "DOUBLE PRECISION"
    if isinstance(spark_type, FloatType):
        return "REAL"
    if isinstance(spark_type, DecimalType):
        if not (1 <= spark_type.precision <= 38):
            raise ValueError("Unsupported decimal precision")
        return f"DECIMAL({spark_type.precision}," f"{spark_type.scale})"
    if isinstance(spark_type, StringType):
        return "VARCHAR(65535)"
    if isinstance(spark_type, BooleanType):
        return "BOOLEAN"

    raise ValueError(f"Unsupported serving type: {spark_type.simpleString()}")


def staging_table_name(table: str, release_id: str) -> str:
    sql_identifier(table)
    digest = hashlib.sha256(release_id.encode("utf-8")).hexdigest()[:12]
    return f"{table}_r{digest}"


def staging_ddl(table: str, release_id: str, schema) -> str:
    """Preserve Parquet field order exactly in Redshift DDL."""

    physical_name = staging_table_name(table, release_id)

    columns = [
        f"  {sql_identifier(field.name)} " f"{redshift_type(field.dataType)}"
        for field in schema.fields
    ]

    if not columns:
        raise ValueError("Serving schema must not be empty")

    return (
        f"CREATE TABLE nyc_stage."
        f"{sql_identifier(physical_name)} (\n" + ",\n".join(columns) + "\n);"
    )


def validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid Active Manifest")

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest version")

    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
        release_id,
    ):
        raise ValueError("Invalid release ID")

    expected = set(GOLD_TABLE_CONTRACTS)

    paths = manifest.get("table_paths")
    versions = manifest.get("table_versions")

    if not isinstance(paths, dict) or set(paths) != expected:
        raise ValueError("Gold table path mapping mismatch")

    if not isinstance(versions, dict) or set(versions) != expected:
        raise ValueError("Gold version mapping mismatch")

    for name in expected:
        if (
            not isinstance(paths[name], str)
            or not Path(paths[name]).is_absolute()
            or type(versions[name]) is not int
            or versions[name] < 0
        ):
            raise ValueError(f"Invalid pinned Gold reference: {name}")


def export_gold_snapshot(
    spark,
    *,
    active_manifest: str,
    output_root: str,
) -> dict:
    """Export six pinned tables; keep incomplete exports on failure."""

    manifest_path = Path(active_manifest)
    original = manifest_path.read_bytes()
    manifest = json.loads(original)

    validate_manifest(manifest)
    release_id = manifest["release_id"]

    destination = Path(output_root).resolve() / release_id

    if destination.exists():
        raise ValueError(f"Serving export already exists: {destination}")

    # Preflight all table schemas before writing anything.
    prepared = {}

    for name in GOLD_TABLE_CONTRACTS:
        df = (
            spark.read.format("delta")
            .option("versionAsOf", manifest["table_versions"][name])
            .load(manifest["table_paths"][name])
        )

        validate_gold_schema(df, name)

        business_time = GOLD_TABLE_CONTRACTS[name]["keys"][0]
        expected_type = TimestampNTZType if name == "trip_summary_hourly" else DateType

        if not isinstance(
            df.schema[business_time].dataType,
            expected_type,
        ):
            raise ValueError(f"Incorrect business-time type: {name}")

        ddl = staging_ddl(name, release_id, df.schema)
        prepared[name] = (df, ddl)

    destination.mkdir(parents=True, exist_ok=False)

    # Preserve microsecond precision for timestamp_ntz export.
    spark.conf.set(
        "spark.sql.parquet.outputTimestampType",
        "TIMESTAMP_MICROS",
    )

    plan = {
        "release_id": release_id,
        "source_manifest": str(manifest_path),
        "source_versions": manifest["table_versions"],
        "logical_window": manifest["logical_window"],
        "tables": {},
    }

    for name, (df, ddl) in prepared.items():
        output_path = destination / name

        df.write.mode("errorifexists").parquet(str(output_path))

        exported = spark.read.parquet(str(output_path))
        row_count = exported.count()

        # Count the pinned source as well; a partial Spark write must not pass.
        if df.count() != row_count:
            raise ValueError(f"Gold/export row count mismatch: {name}")

        if row_count == 0:
            raise ValueError(f"Empty serving snapshot: {name}")

        plan["tables"][name] = {
            "parquet_path": str(output_path),
            "row_count": row_count,
            "columns": df.columns,
            "staging_table": staging_table_name(name, release_id),
            "create_table_sql": ddl,
        }

        print(f"[Redshift R1] {name}: {row_count} rows")

    # Detect publication changes during export.
    if manifest_path.read_bytes() != original:
        raise RuntimeError("Active Gold Manifest changed during export")

    temp_plan = destination / ".export_plan.json.tmp"
    temp_plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    temp_plan.replace(destination / "export_plan.json")

    # Exact file inventory is the final local publication marker.
    publish_seal(destination)

    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(Path(config.BASE_DIR) / "data/delta/gold/active.json"),
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(config.BASE_DIR) / "data/delta/serving_exports"),
    )
    args = parser.parse_args()

    spark = get_spark("NYC Redshift Serving Export")

    try:
        result = export_gold_snapshot(
            spark,
            active_manifest=args.manifest,
            output_root=args.output_root,
        )
        print(
            "REDSHIFT R1 EXPORT: PASS",
            result["release_id"],
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
