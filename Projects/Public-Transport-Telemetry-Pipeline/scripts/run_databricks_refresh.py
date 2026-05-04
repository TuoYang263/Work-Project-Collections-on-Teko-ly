"""
Databricks refresh wrapper for the Public Transport Telemetry Pipeline.

This script is intended to be used by an Azure Databricks Job.

It reuses the existing pipeline scripts:

1. Run Bronze -> Silver -> Gold pipeline
2. Export Gold tables to parquet
3. Upload exported parquet outputs to Azure Blob

The Streamlit dashboard remains unchanged and continues to read
the latest dashboard-ready parquet files from Azure Blob.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_env() -> dict[str, str]:
    """Build a stable runtime environment for local and Databricks execution."""
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
    )

    env.setdefault("SPARK_LOCAL_DIR", "/tmp/spark-tmp")
    env.setdefault("SPARK_LOCAL_DIRS", "/tmp/spark-tmp")
    env.setdefault("SPARK_WAREHOUSE_DIR", "file:/tmp/spark-warehouse/telemetry")

    # Ensure local Spark temp directories exist.
    for key in ["SPARK_LOCAL_DIR", "SPARK_LOCAL_DIRS"]:
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    # Ensure file-based Spark warehouse directory exists.
    warehouse_uri = env["SPARK_WAREHOUSE_DIR"]
    if warehouse_uri.startswith("file:"):
        warehouse_path = Path(warehouse_uri.replace("file:", "", 1))
        warehouse_path.mkdir(parents=True, exist_ok=True)

    return env


def run_command(step_name: str, command: list[str], env: dict[str, str]) -> None:
    """Run one refresh step and fail fast if it fails."""
    print(f"\n=== {step_name} ===")
    print("Command:", " ".join(command))

    start = time.time()

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    elapsed = time.time() - start
    print(f"=== {step_name} completed in {elapsed:.2f}s ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run telemetry refresh through a Databricks-compatible wrapper."
    )

    parser.add_argument(
        "--layer",
        default="full",
        choices=["bronze", "silver", "gold", "full"],
        help="Pipeline layer to run. Default: full.",
    )

    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run pipeline and export only. Useful for local validation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_env()

    print("Project root:", PROJECT_ROOT)
    print("Databricks runtime:", env.get("DATABRICKS_RUNTIME_VERSION", "not detected"))
    print("Selected layer:", args.layer)
    print("Skip upload:", args.skip_upload)

    run_command(
        step_name="Run pipeline",
        command=[
            sys.executable,
            "scripts/run_pipeline.py",
            "--layer",
            args.layer,
        ],
        env=env,
    )

    run_command(
        step_name="Export Gold outputs",
        command=[
            sys.executable,
            "scripts/export_gold.py",
        ],
        env=env,
    )

    if args.skip_upload:
        print("Skipping Azure Blob upload because --skip-upload was provided.")
        return

    run_command(
        step_name="Upload parquet outputs to Azure Blob",
        command=[
            sys.executable,
            "scripts/upload_outputs_to_blob.py",
        ],
        env=env,
    )

    print("\nDatabricks telemetry refresh completed successfully.")


if __name__ == "__main__":
    main()
