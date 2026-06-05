from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.validation_report import QualityReport

DEFAULT_REPORT_PATH = Path("data/quality/reports/hsl_source_validation_report.json")
DEFAULT_SUMMARY_PATH = Path("data/quality/reports/latest_hsl_source_summary.json")


def run_hsl_source_validation(
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> QualityReport:
    """
    Run optional compatibility checks for HSL source snapshots.

    This script does not feed the dashboard or modify pipeline outputs.
    It validates whether a source snapshot can be interpreted by the
    project's telemetry event model.
    """
    report = QualityReport(source="hsl_source_snapshot")

    report.add_metadata("validation_scope", "real_source_compatibility")
    report.add_metadata("source_system", "HSL")
    report.add_metadata("connected_to_dashboard", False)
    report.add_metadata("runs_pipeline", False)
    report.add_metadata("modifies_pipeline_outputs", False)

    report.add_check(
        name="hsl_source_validation_skeleton",
        status="passed",
        severity="info",
        details=(
            "HSL source validation script is available. "
            "Dataset-specific compatibility checks will be added in the next step."
        ),
    )

    report.save(report_path)
    report.save_summary(summary_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate optional HSL source snapshot compatibility."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the full HSL source validation report JSON.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the compact HSL source validation summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_hsl_source_validation(
        report_path=args.output,
        summary_path=args.summary_output,
    )

    summary = report.summary()

    print("[data-quality] HSL source validation completed")
    print(f"[data-quality] Status: {summary['status']}")
    print(f"[data-quality] Total checks: {summary['total_checks']}")
    print(f"[data-quality] Passed: {summary['passed']}")
    print(f"[data-quality] Warnings: {summary['warnings']}")
    print(f"[data-quality] Failed: {summary['failed']}")
    print(f"[data-quality] Full report written to: {args.output}")
    print(f"[data-quality] Summary written to: {args.summary_output}")

    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
