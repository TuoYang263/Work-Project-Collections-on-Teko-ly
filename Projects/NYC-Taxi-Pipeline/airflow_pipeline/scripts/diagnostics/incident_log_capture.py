
"""Bounded, allowlisted log excerpts for observed lifecycle probes."""

import hashlib
import json
import os
import re
from pathlib import Path

from scripts.diagnostics.incident_db_capture import RUN_IDS
from scripts.diagnostics.log_references import (
    DEFAULT_LOG_ROOT,
    get_task_log_references,
)

DAG_ID = "nyc_lifecycle_probe"
TASK_ID = "harmless_task"
MAX_FILE_BYTES = 16 * 1024
MAX_CASE_BYTES = 64 * 1024

EXPECTED_ATTEMPTS = {
    RUN_IDS[0]: [1],
    RUN_IDS[1]: [1, 2],
    RUN_IDS[2]: [1, 2],
}

MARKERS = {
    "ATTEMPT": re.compile(r"\[PROBE\] Attempt:\s*\d+"),
    "FIRST_FAILURE": re.compile(r"Intentional first-attempt failure"),
    "TERMINAL_FAILURE": re.compile(r"Intentional lifecycle probe failure"),
    "SUCCESS": re.compile(r"Lifecycle probe executed successfully"),
}


def prepare_log_excerpt_artifact(
    run_id: str,
    *,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> dict:
    """Extract only approved observed diagnostic markers."""

    if run_id not in EXPECTED_ATTEMPTS:
        raise ValueError("Unsupported incident run")

    root = Path(log_root).resolve()
    result = get_task_log_references(
        DAG_ID, run_id, TASK_ID, log_root=root
    )

    refs = result["references"]

    if (
        not result["available"]
        or result["truncated"]
        or [r["attempt_number"] for r in refs]
        != EXPECTED_ATTEMPTS[run_id]
    ):
        raise ValueError("Incomplete attempt log references")

    collected = []
    total_bytes = 0

    for ref in refs:
        attempt = ref["attempt_number"]
        expected_path = (
            f"dag_id={DAG_ID}/run_id={run_id}/"
            f"task_id={TASK_ID}/attempt={attempt}.log"
        )

        if ref["relative_path"] != expected_path:
            raise ValueError("Log identity mismatch")

        path = root / expected_path

        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Unsafe log path")

        # Linux/WSL: reject a symlink at the final open boundary.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                raw = handle.read(MAX_FILE_BYTES + 1)
        finally:
            if fd >= 0:
                os.close(fd)

        total_bytes += len(raw)

        if (
            len(raw) > MAX_FILE_BYTES
            or total_bytes > MAX_CASE_BYTES
        ):
            raise ValueError("Log capture size limit exceeded")

        lines = []
        for line_number, line in enumerate(
            raw.decode("utf-8").splitlines(), start=1
        ):
            for label, pattern in MARKERS.items():
                match = pattern.search(line)
                if match:
                    lines.append({
                        "line_number": line_number,
                        "event": label,
                        "excerpt": match.group(0),
                    })

        if not lines:
            raise ValueError("No approved diagnostic excerpts")

        collected.append({
            "attempt_number": attempt,
            "source_path": expected_path,
            "source_size_bytes": len(raw),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "excerpts": lines,
        })

    document = {
        "artifact_type": "AIRFLOW_LOG_EXCERPTS",
        "dag_id": DAG_ID,
        "run_id": run_id,
        "task_id": TASK_ID,
        "capture_scope": "BOUNDED_EXCERPTS_ONLY",
        "redaction_policy": "STRICT_MARKER_ALLOWLIST_V1",
        "full_log_archived": False,
        "logs": collected,
    }

    content = (
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    return {
        "filename": "log_excerpts.json",
        "content": content,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }