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


def main() -> None:
    manifest = load_json(ARTIFACT_DIR / "manifest.json")
    run_results = load_json(ARTIFACT_DIR / "run_results.json")
    catalog = load_json(ARTIFACT_DIR / "catalog.json")

    pipeline_run_record = build_pipeline_run_record(
        manifest=manifest,
        run_results=run_results,
        catalog=catalog,
    )

    print(json.dumps(pipeline_run_record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
