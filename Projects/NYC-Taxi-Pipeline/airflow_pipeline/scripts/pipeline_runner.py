
from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys

from pathlib import Path
from typing import Any, Mapping


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


def submit_stage(
    stage: str,
    execution_contract: Mapping[str, Any],
) -> None:
    """Airflow-facing entry point. No Spark imports here."""
    validate_stage(stage)

    contract_json = json.dumps(
        dict(execution_contract),
        sort_keys=True,
        separators=(",", ":"),
    )

    command = [
        sys.executable,
        "-m",
        "scripts.pipeline_runner",
        stage,
        "--execution-contract-json",
        contract_json,
    ]

    LOGGER.info(
        "Submitting pipeline stage=%s",
        stage,
    )

    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def execute_stage(
    stage: str,
    execution_contract: Mapping[str, Any],
) -> None:
    """Child-process entry point. Import compute code only here."""
    validate_stage(stage)

    pipeline = importlib.import_module(
        "scripts.delta_medallion_pipeline"
    )

    task = getattr(
        pipeline,
        STAGE_FUNCTIONS[stage],
    )

    task(
        execution_contract=dict(execution_contract),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYC Taxi pipeline stage runner"
    )

    parser.add_argument(
        "stage",
        choices=tuple(STAGE_FUNCTIONS),
    )

    parser.add_argument(
        "--execution-contract-json",
        required=True,
    )

    args = parser.parse_args()

    execution_contract = json.loads(
        args.execution_contract_json
    )

    if not isinstance(execution_contract, dict):
        raise ValueError(
            "Execution contract must be a JSON object"
        )

    execute_stage(
        args.stage,
        execution_contract,
    )


if __name__ == "__main__":
    main()