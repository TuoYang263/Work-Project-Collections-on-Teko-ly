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
    "quality": "quality_task",
    "export": "export_bq_all_task",
}


def validate_stage(stage: str) -> None:
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"Unsupported pipeline stage: {stage}")


def submit_stage(
    stage: str,
    execution_contract: Mapping[str, Any],
    raw_references: list[dict] | None = None,
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

    if raw_references is not None:
        if stage != "bronze":
            raise ValueError("Raw references are supported only for Bronze")

        command.extend(
            [
                "--raw-references-json",
                json.dumps(
                    raw_references,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )

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
    raw_references: list[dict] | None = None,
) -> None:
    """Child-process entry point. Import compute code only here."""
    validate_stage(stage)

    pipeline = importlib.import_module("scripts.delta_medallion_pipeline")

    task = getattr(
        pipeline,
        STAGE_FUNCTIONS[stage],
    )

    task_kwargs = {
        "execution_contract": dict(execution_contract),
    }

    if raw_references is not None:
        if stage != "bronze":
            raise ValueError("Raw references are supported only for Bronze")

        task_kwargs["raw_references"] = raw_references

    task(**task_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC Taxi pipeline stage runner")

    parser.add_argument(
        "stage",
        choices=tuple(STAGE_FUNCTIONS),
    )

    parser.add_argument(
        "--execution-contract-json",
        required=True,
    )

    parser.add_argument(
        "--raw-references-json",
        default=None,
    )

    args = parser.parse_args()

    execution_contract = json.loads(args.execution_contract_json)

    if not isinstance(execution_contract, dict):
        raise ValueError("Execution contract must be a JSON object")

    raw_references = None

    if args.raw_references_json is not None:
        raw_references = json.loads(args.raw_references_json)

        if (
            not isinstance(raw_references, list)
            or not raw_references
            or any(not isinstance(ref, dict) for ref in raw_references)
        ):
            raise ValueError("Raw references must be a non-empty list of objects")

        if args.stage != "bronze":
            raise ValueError("Raw references are supported only for Bronze")

    execute_stage(
        args.stage,
        execution_contract,
        raw_references=raw_references,
    )


if __name__ == "__main__":
    main()
