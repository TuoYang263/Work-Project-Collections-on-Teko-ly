from pathlib import Path
import json

ARTIFACT_DIR = Path("dbt/target")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
    

def main() -> None:
    manifest_path = ARTIFACT_DIR / "manifest.json"
    run_results_path = ARTIFACT_DIR / "run_results.json"
    catalog_path = ARTIFACT_DIR / "catalog.json"

    manifest = load_json(manifest_path)
    run_results = load_json(run_results_path)
    catalog = load_json(catalog_path)

    manifest_nodes = manifest.get("nodes", {})
    manifest_sources = manifest.get("sources", {})
    run_results_items = run_results.get("results", {})
    catalog_nodes = catalog.get("nodes", {})
    catalog_sources = catalog.get("sources", {})

    print("dbt artifact inspection")
    print("=======================")

    print(f"manifest.json path: {manifest_path}")
    print(f"run_results.json path: {run_results_path}")
    print(f"catalog.json path: {catalog_path}")
    print()

    print("manifest.json")
    print(f"- nodes: {len(manifest_nodes)}")
    print(f"- sources: {len(manifest_sources)}")
    print(f"- invocation_id: {manifest.get('metadata', {}).get('invocation_id')}")
    print(f"- dbt_version: {manifest.get('metadata', {}).get('dbt_version')}")
    print()

    print("run_results.json")
    print(f"- results: {len(run_results_items)}")
    print(f"- invocation_id: {run_results.get('metadata', {}).get('invocation_id')}")
    print(f"- dbt_version: {run_results.get('metadata', {}).get('dbt_version')}")
    print(f"- elapsed_time: {run_results.get('elapsed_time')}")
    print()

    print("catalog.json")
    print(f"- nodes: {len(catalog_nodes)}")
    print(f"- sources: {len(catalog_sources)}")
    print(f"- invocation_id: {catalog.get('metadata', {}).get('invocation_id')}")
    print(f"- dbt_version: {catalog.get('metadata', {}).get('dbt_version')}")

if __name__ == "__main__":
    main()
