"""Read-only references to existing local Airflow task logs."""

import re
from pathlib import Path
from typing import Any

DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[2] / "logs"
MAX_LOG_REFS = 20


def _validate_component(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError("Invalid log identity component")


def get_task_log_references(
    dag_id: str,
    run_id: str,
    task_id: str,
    *,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> dict[str, Any]:
    """Discover existing attempt logs without reading their contents."""
    for value in (dag_id, run_id, task_id):
        _validate_component(value)

    root = Path(log_root).resolve()

    task_dir = (
        root
        / f"dag_id={dag_id}"
        / f"run_id={run_id}"
        / f"task_id={task_id}"
    )

    resolved_dir = task_dir.resolve()

    if resolved_dir != task_dir:
        raise ValueError("Log identity path contains a symlink")

    if not resolved_dir.is_relative_to(root):
        raise ValueError("Log path escapes the allowed root")

    found = []

    if task_dir.is_dir():
        for path in task_dir.glob("attempt=*.log"):
            match = re.fullmatch(r"attempt=(\d+)\.log", path.name)
            if not match or not path.is_file():
                continue

            if path.is_symlink():
                continue

            if not path.resolve().is_relative_to(task_dir):
                continue

            found.append((int(match.group(1)), path))

    found.sort(key=lambda item: item[0])

    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "task_id": task_id,
        "source": "LOCAL_AIRFLOW_TASK_LOGS",
        "available": bool(found),
        "truncated": len(found) > MAX_LOG_REFS,
        "references": [
            {
                "attempt_number": number,
                "relative_path": path.relative_to(root).as_posix(),
            }
            for number, path in found[:MAX_LOG_REFS]
        ],
    }
