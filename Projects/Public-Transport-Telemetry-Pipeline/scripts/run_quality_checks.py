from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.pipeline_quality import (
    DEFAULT_REPORT_PATH,
    DEFAULT_SUMMARY_PATH,
    run_pipeline_quality_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run portfolio-scale data quality checks against existing "
            "pipeline output datasets."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Path for the full quality report JSON. " f"Default: {DEFAULT_REPORT_PATH}"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=(
            "Path for the lightweight quality summary JSON. "
            f"Default: {DEFAULT_SUMMARY_PATH}"
        ),
    )

    return parser.parse_args()


def print_summary(summary: dict) -> None:
    print("\n[data-quality] Pipeline output quality check completed")
    print(f"[data-quality] Status: {summary['status']}")
    print(f"[data-quality] Total checks: {summary['total_checks']}")
    print(f"[data-quality] Passed: {summary['passed']}")
    print(f"[data-quality] Warnings: {summary['warnings']}")
    print(f"[data-quality] Failed: {summary['failed']}")

    if summary.get("record_count"):
        print("[data-quality] Record counts:")
        for dataset_name, count in summary["record_count"].items():
            print(f"  - {dataset_name}: {count}")


def main() -> int:
    args = parse_args()

    report = run_pipeline_quality_checks(
        report_path=args.output,
        summary_path=args.summary_output,
    )

    summary = report.summary()
    print_summary(summary)

    print(f"\n[data-quality] Full report written to: {args.output}")
    print(f"[data-quality] Summary written to: {args.summary_output}")

    if report.errors:
        print("\n[data-quality] Errors:")
        for error in report.errors:
            print(f"  - {error}")

    if report.warnings:
        print("\n[data-quality] Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
