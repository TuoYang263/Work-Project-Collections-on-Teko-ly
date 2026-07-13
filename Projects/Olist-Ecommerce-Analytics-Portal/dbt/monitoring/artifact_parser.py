from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

ARTIFACT_DIR = Path("dbt/target")
DEFAULT_JOB_NAME = "local-dbt-artifact-inspection"
DEFAULT_ENVIRONMENT = "dev"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_node_display_name(node: dict) -> str | None:
    if not node:
        return None

    resource_type = node.get("resource_type")

    if resource_type == "source":
        source_name = node.get("source_name")
        name = node.get("name")

        if source_name and name:
            return f"{source_name}.{name}"

    return node.get("name")


def get_node_by_unique_id(manifest: dict, unique_id: str) -> dict:
    nodes = manifest.get("nodes", {})
    sources = manifest.get("sources", {})

    if unique_id in nodes:
        return nodes[unique_id]

    if unique_id in sources:
        return sources[unique_id]

    return {}


def get_run_timing_window(run_results: dict) -> tuple[str | None, str | None]:
    started_at_values = []
    completed_at_values = []

    for result in run_results.get("results", []):
        for timing in result.get("timing", []):
            started_at = timing.get("started_at")
            completed_at = timing.get("completed_at")

            if started_at:
                started_at_values.append(started_at)

            if completed_at:
                completed_at_values.append(completed_at)

    run_started_at = min(started_at_values) if started_at_values else None
    run_completed_at = max(completed_at_values) if completed_at_values else None

    return run_started_at, run_completed_at


def build_pipeline_run_record(
    manifest: dict,
    run_results: dict,
    catalog: dict,
) -> dict:
    ingested_at = datetime.now(timezone.utc).isoformat()

    run_metadata = run_results.get("metadata", {})
    manifest_metadata = manifest.get("metadata", {})

    dbt_invocation_id = run_metadata.get("invocation_id")
    dbt_version = run_metadata.get("dbt_version") or manifest_metadata.get(
        "dbt_version"
    )

    monitoring_run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    run_started_at, run_completed_at = get_run_timing_window(run_results)

    results = run_results.get("results", [])

    model_results = []
    test_results = []

    for result in results:
        unique_id = result.get("unique_id")
        node = get_node_by_unique_id(manifest, unique_id)
        resource_type = node.get("resource_type")

        if resource_type in {"model", "seed", "snapshot"}:
            model_results.append(result)

        if resource_type == "test":
            test_results.append(result)

    models_total = len(model_results)
    models_success = sum(
        1 for result in model_results if result.get("status") == "success"
    )
    models_error = sum(1 for result in model_results if result.get("status") == "error")
    models_skipped = sum(
        1 for result in model_results if result.get("status") == "skipped"
    )

    tests_total = len(test_results)
    tests_passed = sum(
        1 for result in test_results if result.get("status") in {"success", "pass"}
    )
    tests_failed = sum(1 for result in test_results if result.get("status") == "fail")
    tests_warned = sum(1 for result in test_results if result.get("status") == "warn")
    tests_error = sum(1 for result in test_results if result.get("status") == "error")

    if models_error > 0 or tests_error > 0:
        pipeline_status = "error"
    elif tests_failed > 0 or tests_warned > 0 or models_skipped > 0:
        pipeline_status = "partial_failure"
    else:
        pipeline_status = "success"

    return {
        "monitoring_run_id": monitoring_run_id,
        "dbt_invocation_id": dbt_invocation_id,
        "job_name": DEFAULT_JOB_NAME,
        "environment": DEFAULT_ENVIRONMENT,
        "dbt_version": dbt_version,
        "generated_at": run_metadata.get("generated_at"),
        "ingested_at": ingested_at,
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at or ingested_at,
        "total_elapsed_time_seconds": run_results.get("elapsed_time"),
        "status": pipeline_status,
        "models_total": models_total,
        "models_success": models_success,
        "models_error": models_error,
        "models_skipped": models_skipped,
        "tests_total": tests_total,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_warned": tests_warned,
        "tests_error": tests_error,
        "artifact_manifest_path": str(ARTIFACT_DIR / "manifest.json"),
        "artifact_run_results_path": str(ARTIFACT_DIR / "run_results.json"),
        "artifact_catalog_path": str(ARTIFACT_DIR / "catalog.json"),
    }


def build_model_run_result_records(
    manifest: dict,
    run_results: dict,
    pipeline_run_record: dict,
) -> list[dict]:
    model_run_records = []

    for result in run_results.get("results", []):
        unique_id = result.get("unique_id")
        node = get_node_by_unique_id(manifest, unique_id)
        resource_type = node.get("resource_type")

        if resource_type not in {"model", "seed", "snapshot"}:
            continue

        config = node.get("config", {})

        model_run_records.append(
            {
                "monitoring_run_id": pipeline_run_record["monitoring_run_id"],
                "dbt_invocation_id": pipeline_run_record["dbt_invocation_id"],
                "unique_id": unique_id,
                "model_name": node.get("name"),
                "resource_type": resource_type,
                "package_name": node.get("package_name"),
                "database_name": node.get("database"),
                "schema_name": node.get("schema"),
                "alias": node.get("alias"),
                "materialized": config.get("materialized"),
                "status": result.get("status"),
                "execution_time_seconds": result.get("execution_time"),
                "thread_id": result.get("thread_id"),
                "message": result.get("message"),
                "adapter_response_json": json.dumps(
                    result.get("adapter_response", {}),
                    ensure_ascii=False,
                ),
                "ingested_at": pipeline_run_record["ingested_at"],
            }
        )

    return model_run_records


def normalize_test_status(status: str | None) -> str | None:
    if status == "success":
        return "pass"

    return status


def build_test_run_result_records(
    manifest: dict,
    run_results: dict,
    pipeline_run_record: dict,
) -> list[dict]:
    test_run_records = []

    for result in run_results.get("results", []):
        unique_id = result.get("unique_id")
        node = get_node_by_unique_id(manifest, unique_id)
        resource_type = node.get("resource_type")

        if resource_type != "test":
            continue

        config = node.get("config", {})
        test_metadata = node.get("test_metadata", {})
        attached_node_unique_id = node.get("attached_node")

        attached_node = get_node_by_unique_id(
            manifest=manifest,
            unique_id=attached_node_unique_id,
        )

        test_type = "generic" if test_metadata else "singular"

        test_run_records.append(
            {
                "monitoring_run_id": pipeline_run_record["monitoring_run_id"],
                "dbt_invocation_id": pipeline_run_record["dbt_invocation_id"],
                "unique_id": unique_id,
                "test_name": node.get("name"),
                "test_type": test_type,
                "test_metadata_name": test_metadata.get("name"),
                "model_unique_id": attached_node_unique_id,
                "model_name": attached_node.get("name"),
                "column_name": node.get("column_name"),
                "status": normalize_test_status(result.get("status")),
                "severity": (
                    str(config.get("severity")).lower()
                    if config.get("severity")
                    else None
                ),
                "failures": result.get("failures"),
                "execution_time_seconds": result.get("execution_time"),
                "thread_id": result.get("thread_id"),
                "message": result.get("message"),
                "adapter_response_json": json.dumps(
                    result.get("adapter_response", {}),
                    ensure_ascii=False,
                ),
                "ingested_at": pipeline_run_record["ingested_at"],
            }
        )

    return test_run_records


def get_catalog_node_by_unique_id(catalog: dict, unique_id: str) -> dict:
    nodes = catalog.get("nodes", {})
    sources = catalog.get("sources", {})

    if unique_id in nodes:
        return nodes[unique_id]

    if unique_id in sources:
        return sources[unique_id]

    return {}


def build_column_tests_lookup(manifest: dict) -> dict[tuple[str, str], list[dict]]:
    column_tests_lookup = {}

    for test_unique_id, test_node in manifest.get("nodes", {}).items():
        if test_node.get("resource_type") != "test":
            continue

        attached_node_unique_id = test_node.get("attached_node")
        column_name = test_node.get("column_name")

        if not attached_node_unique_id or not column_name:
            continue

        config = test_node.get("config", {})
        test_metadata = test_node.get("test_metadata", {})

        test_summary = {
            "unique_id": test_unique_id,
            "test_name": test_node.get("name"),
            "test_metadata_name": test_metadata.get("name"),
            "severity": (
                str(config.get("severity")).lower() if config.get("severity") else None
            ),
        }

        lookup_key = (attached_node_unique_id, column_name)
        column_tests_lookup.setdefault(lookup_key, []).append(test_summary)

    return column_tests_lookup


def build_model_column_snapshot_records(
    manifest: dict,
    catalog: dict,
    pipeline_run_record: dict,
) -> list[dict]:
    column_records = []
    column_tests_lookup = build_column_tests_lookup(manifest)

    parent_nodes = {}

    for unique_id, node in manifest.get("nodes", {}).items():
        if node.get("resource_type") in {"model", "seed", "snapshot"}:
            parent_nodes[unique_id] = node

    for unique_id, source in manifest.get("sources", {}).items():
        parent_nodes[unique_id] = source

    for parent_unique_id, parent_node in parent_nodes.items():
        catalog_node = get_catalog_node_by_unique_id(
            catalog=catalog,
            unique_id=parent_unique_id,
        )

        manifest_columns = parent_node.get("columns", {}) or {}
        catalog_columns = catalog_node.get("columns", {}) or {}

        column_names = []

        sorted_catalog_columns = sorted(
            catalog_columns.items(),
            key=lambda item: (
                item[1].get("index") if item[1].get("index") is not None else 999999
            ),
        )

        for column_name, _ in sorted_catalog_columns:
            column_names.append(column_name)

        for column_name in manifest_columns:
            if column_name not in column_names:
                column_names.append(column_name)

        for column_name in column_names:
            manifest_column = manifest_columns.get(column_name, {})
            catalog_column = catalog_columns.get(column_name, {})

            related_tests = column_tests_lookup.get((parent_unique_id, column_name), [])

            column_records.append(
                {
                    "monitoring_run_id": pipeline_run_record["monitoring_run_id"],
                    "dbt_invocation_id": pipeline_run_record["dbt_invocation_id"],
                    "model_unique_id": parent_unique_id,
                    "model_name": parent_node.get("name"),
                    "resource_type": parent_node.get("resource_type"),
                    "column_name": column_name,
                    "data_type": catalog_column.get("type")
                    or manifest_column.get("data_type"),
                    "column_index": catalog_column.get("index"),
                    "description": manifest_column.get("description")
                    or catalog_column.get("comment"),
                    "tests_json": json.dumps(
                        related_tests,
                        ensure_ascii=False,
                    ),
                    "catalog_column_metadata_json": json.dumps(
                        catalog_column,
                        ensure_ascii=False,
                    ),
                    "ingested_at": pipeline_run_record["ingested_at"],
                }
            )

    return column_records


def build_model_lineage_edge_records(
    manifest: dict,
    pipeline_run_record: dict,
) -> list[dict]:
    lineage_records = []

    allowed_child_resource_types = {"model", "seed", "snapshot", "test"}
    allowed_parent_resource_types = {"source", "model", "seed", "snapshot"}

    for child_unique_id, child_node in manifest.get("nodes", {}).items():
        child_resource_type = child_node.get("resource_type")

        if child_resource_type not in allowed_child_resource_types:
            continue

        parent_unique_ids = child_node.get("depends_on", {}).get("nodes", [])

        for parent_unique_id in parent_unique_ids:
            parent_node = get_node_by_unique_id(
                manifest=manifest,
                unique_id=parent_unique_id,
            )
            parent_resource_type = parent_node.get("resource_type")

            if parent_resource_type not in allowed_parent_resource_types:
                continue

            lineage_records.append(
                {
                    "monitoring_run_id": pipeline_run_record["monitoring_run_id"],
                    "dbt_invocation_id": pipeline_run_record["dbt_invocation_id"],
                    "parent_unique_id": parent_unique_id,
                    "parent_name": get_node_display_name(parent_node),
                    "parent_resource_type": parent_resource_type,
                    "child_unique_id": child_unique_id,
                    "child_name": get_node_display_name(child_node),
                    "child_resource_type": child_resource_type,
                    "dependency_type": "depends_on_node",
                    "ingested_at": pipeline_run_record["ingested_at"],
                }
            )

    lineage_records.sort(
        key=lambda record: (
            record["parent_unique_id"] or "",
            record["child_unique_id"] or "",
        )
    )

    return lineage_records


def get_catalog_stat_value(catalog_node: dict, stat_name: str):
    return catalog_node.get("stats", {}).get(stat_name, {}).get("value")


def build_model_metadata_snapshot_records(
    manifest: dict,
    catalog: dict,
    pipeline_run_record: dict,
) -> list[dict]:
    model_metadata_records = []

    for unique_id, node in manifest.get("nodes", {}).items():
        resource_type = node.get("resource_type")

        if resource_type not in {"model", "seed", "snapshot"}:
            continue

        config = node.get("config", {})
        catalog_node = get_catalog_node_by_unique_id(
            catalog=catalog,
            unique_id=unique_id,
        )

        model_metadata_records.append(
            {
                "monitoring_run_id": pipeline_run_record["monitoring_run_id"],
                "dbt_invocation_id": pipeline_run_record["dbt_invocation_id"],
                "unique_id": unique_id,
                "model_name": node.get("name"),
                "resource_type": resource_type,
                "package_name": node.get("package_name"),
                "database_name": node.get("database"),
                "schema_name": node.get("schema"),
                "alias": node.get("alias"),
                "relation_name": node.get("relation_name"),
                "materialized": config.get("materialized"),
                "path": node.get("path"),
                "original_file_path": node.get("original_file_path"),
                "description": node.get("description"),
                "tags_json": json.dumps(node.get("tags", []), ensure_ascii=False),
                "meta_json": json.dumps(node.get("meta", {}), ensure_ascii=False),
                "row_count": get_catalog_stat_value(catalog_node, "num_rows"),
                "bytes": get_catalog_stat_value(catalog_node, "num_bytes"),
                "catalog_metadata_json": json.dumps(
                    catalog_node.get("metadata", {}),
                    ensure_ascii=False,
                ),
                "ingested_at": pipeline_run_record["ingested_at"],
            }
        )

    return model_metadata_records


def main() -> None:
    manifest = load_json(ARTIFACT_DIR / "manifest.json")
    run_results = load_json(ARTIFACT_DIR / "run_results.json")
    catalog = load_json(ARTIFACT_DIR / "catalog.json")

    pipeline_run_record = build_pipeline_run_record(
        manifest=manifest,
        run_results=run_results,
        catalog=catalog,
    )

    model_run_records = build_model_run_result_records(
        manifest=manifest,
        run_results=run_results,
        pipeline_run_record=pipeline_run_record,
    )

    test_run_records = build_test_run_result_records(
        manifest=manifest,
        run_results=run_results,
        pipeline_run_record=pipeline_run_record,
    )

    model_metadata_records = build_model_metadata_snapshot_records(
        manifest=manifest,
        catalog=catalog,
        pipeline_run_record=pipeline_run_record,
    )

    model_column_records = build_model_column_snapshot_records(
        manifest=manifest,
        catalog=catalog,
        pipeline_run_record=pipeline_run_record,
    )

    model_lineage_records = build_model_lineage_edge_records(
        manifest=manifest,
        pipeline_run_record=pipeline_run_record,
    )

    print("pipeline_run_record")
    print("===================")
    print(json.dumps(pipeline_run_record, indent=2, ensure_ascii=False))
    print()

    print("model_run_records sample")
    print("========================")
    print(f"total model run records: {len(model_run_records)}")
    print(json.dumps(model_run_records[:3], indent=2, ensure_ascii=False))
    print()

    print("test_run_records sample")
    print("=======================")
    print(f"total test run records: {len(test_run_records)}")
    print(json.dumps(test_run_records[:3], indent=2, ensure_ascii=False))
    print()

    print("model_metadata_records sample")
    print("=============================")
    print(f"total model metadata records: {len(model_metadata_records)}")
    print(json.dumps(model_metadata_records[:3], indent=2, ensure_ascii=False))
    print()

    print("model_column_records sample")
    print("===========================")
    print(f"total model column records: {len(model_column_records)}")
    print(json.dumps(model_column_records[:5], indent=2, ensure_ascii=False))
    print()

    print("model_lineage_records sample")
    print("============================")
    print(f"total model lineage records: {len(model_lineage_records)}")
    print(json.dumps(model_lineage_records[:10], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
