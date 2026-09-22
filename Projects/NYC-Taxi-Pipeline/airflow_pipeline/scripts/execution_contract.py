from __future__ import annotations

from datetime import date
from typing import Any, Mapping


WINDOW_OVERRIDE_KEYS = {
    "year",
    "month",
    "months",
    "start_date",
    "end_date",
    "window_start",
    "window_end",
    "window_end_exclusive",
    "backfill",
}


def _next_month(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1)

    return date(year, month + 1, 1)


def build_current_execution_contract(
    settings: Mapping[str, Any],
    dag_run_conf: Mapping[str, Any] | None = None,
) -> dict[str, Any]:

    data_config = settings.get("data_config", {})

    year = int(data_config["year"])
    months = [int(month) for month in data_config["months"]]

    if not months:
        raise ValueError(
            "data_config.months must not be empty"
        )

    if any(month < 1 or month > 12 for month in months):
        raise ValueError(
            f"data_config.months contains invalid month values: {months}"
        )

    normalized_months = sorted(set(months))

    if normalized_months != months:
        raise ValueError(
            "data_config.months must be sorted and contain no duplicates"
        )

    expected_contiguous = list(
        range(
            normalized_months[0],
            normalized_months[-1] + 1,
        )
    )

    if normalized_months != expected_contiguous:
        raise ValueError(
            "Current execution contract requires contiguous configured months. "
            f"Received: {months}"
        )

    conf = dict(dag_run_conf or {})

    unsupported_overrides = sorted(
        WINDOW_OVERRIDE_KEYS.intersection(conf.keys())
    )

    if unsupported_overrides:
        raise ValueError(
            "Window/backfill overrides are not supported yet because the "
            "pipeline still uses full-table overwrite semantics. "
            "Refusing to silently ignore unsafe DAG-run parameters: "
            f"{unsupported_overrides}"
        )

    window_start = date(
        year,
        normalized_months[0],
        1,
    )

    window_end_exclusive = _next_month(
        year,
        normalized_months[-1],
    )

    return {
        "mode": "CONFIGURED_FULL_REFRESH",
        "year": year,
        "months": normalized_months,
        "window_start": window_start.isoformat(),
        "window_end_exclusive": window_end_exclusive.isoformat(),
        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics": "RECOMPUTE_CONFIGURED_WINDOW",
    }