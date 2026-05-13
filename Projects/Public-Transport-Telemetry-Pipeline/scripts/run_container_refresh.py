from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print(f"\n[container-refresh] Running: {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_runtime_dirs() -> None:
    runtime_dirs = [
        "/tmp/telemetry-pipeline",
        "/tmp/telemetry-pipeline/data",
        "/tmp/telemetry-pipeline/logs",
        "/tmp/spark-tmp",
        "/tmp/spark-warehouse/telemetry",
    ]

    for directory in runtime_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SPARK_LOCAL_DIRS", "/tmp/spark-tmp")
    os.environ.setdefault("SPARK_WAREHOUSE_DIR", "file:/tmp/spark-warehouse/telemetry")


def main() -> None:
    print("[container-refresh] Starting Azure Container Apps compatible refresh job")
    print(f"[container-refresh] Project root: {PROJECT_ROOT}")

    ensure_runtime_dirs()

    run_command([sys.executable, "scripts/run_pipeline.py", "--layer", "full"])
    run_command([sys.executable, "scripts/export_gold.py"])
    run_command([sys.executable, "scripts/upload_outputs_to_blob.py"])

    print("\n[container-refresh] Refresh job completed successfully")


if __name__ == "__main__":
    main()
