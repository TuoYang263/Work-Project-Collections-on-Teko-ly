
from __future__ import annotations

import argparse
import importlib
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger(__name__)

STAGE_FUNCTIONS = {
    "bronze": "bronze_task",
    "silver": "silver_task",
    "gold": "gold_task",
    "export": "export_bq_all_task",
}


def validate_stage(stage: str) -> None:
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"Unsupported pipeline stage: {stage}")


def submit_stage(stage: str) -> None:
    """Airflow-facing entry point. No Spark imports here."""
    validate_stage(stage)

    command = [
        sys.executable,
        "-m",
        "scripts.pipeline_runner",
        stage,
    ]

    LOGGER.info(
        "Submitting pipeline stage=%s, command=%s",
        stage,
        command,
    )

    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def execute_stage(stage: str) -> None:
    """Child-process entry point. Import compute code only here."""
    validate_stage(stage)

    pipeline = importlib.import_module(
        "scripts.delta_medallion_pipeline"
    )

    task = getattr(
        pipeline,
        STAGE_FUNCTIONS[stage],
    )

    task()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYC Taxi pipeline stage runner"
    )

    parser.add_argument(
        "stage",
        choices=tuple(STAGE_FUNCTIONS),
    )

    args = parser.parse_args()

    execute_stage(args.stage)


if __name__ == "__main__":
    main()