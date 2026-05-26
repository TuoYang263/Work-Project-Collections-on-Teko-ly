from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Allowed status values for individual validation checks.
CheckStatus = Literal["passed", "warning", "failed"]

# Allowed overall report status values.
ReportStatus = Literal["passed", "passed_with_warnings", "failed"]

# Allowed severity levels for validation checks.
CheckSeverity = Literal["critical", "warning", "info"]

# Dataclass is used here as a lightweight container for structured check results
@dataclass
class QualityCheck:
    """
    Result of one data quality or validation check.

    status:
    - passed: the check passed normally
    - warning: the check found a non-critical issue
    - failed: the check found a critical issue

    severity:
    - critical: failure should fail the report
    - warning: issue should be visible but should not fail the report
    - info: useful context only
    """
    name: str
    status: CheckStatus
    severity: CheckSeverity
    details: str


@dataclass
class QualityReport:
    """
    Unified JSON-friendly quality report.

    The report is designed for portfolio-scale data engineering validation.
    It can be written as an artifact, reviewed manually, or summarized later
    in the dashboard without making the dashboard execute quality checks.
    """
    source: str

    # Unique identifier for this validation report run.
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Report creation time in UTC ISO format.
    ingest_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    status: ReportStatus = "passed"

    # Each report starts with its own empty list of check results.
    # default_factory avoids using a shared mutable default.
    checks: list[QualityCheck] = field(default_factory=list)

    record_count: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_check(
        self,
        name: str,
        status: CheckStatus,
        details: str,
        severity: CheckSeverity = "critical",
    ) -> None:
        check = QualityCheck(
            name=name,
            status=status,
            severity=severity,
            details=details,
        )
        self.checks.append(check)

        if status == "warning":
            self.warnings.append(f"{name}: {details}")

        if status == "failed":
            self.errors.append(f"{name}: {details}")

        self._refresh_status()

    def set_record_count(self, dataset_name: str, count: int) -> None:
        self.record_count[dataset_name] = int(count)

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def summary(self) -> dict[str, Any]:
        total = len(self.checks)
        passed = sum(1 for check in self.checks if check.status == "passed")
        warnings = sum(1 for check in self.checks if check.status == "warning")
        failed = sum(1 for check in self.checks if check.status == "failed")

        return {
            "source": self.source,
            "status": self.status,
            "run_id": self.run_id,
            "ingest_time": self.ingest_time,
            "total_checks": total,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "record_count": self.record_count,
        }

    def _refresh_status(self) -> None:
        if self.errors:
            self.status = "failed"
        elif self.warnings:
            self.status = "passed_with_warnings"
        else:
            self.status = "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    def save_summary(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output
    
    def exit_code(self) -> int:
        """
        Exit code policy:
        - failed -> 1
        - passed / passed_with_warnings -> 0
        """
        return 1 if self.status == "failed" else 0
    
    

