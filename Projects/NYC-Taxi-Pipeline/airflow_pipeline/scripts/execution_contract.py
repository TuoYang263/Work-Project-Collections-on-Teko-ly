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
}


def _next_month(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1)

    return date(year, month + 1, 1)


def _previous_month(
    year: int,
    month: int,
) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12

    return year, month - 1


def _build_single_month_backfill_contract(
    backfill_config: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_keys = {
        "year",
        "month",
    }

    unknown_keys = sorted(
        set(backfill_config.keys())
        - allowed_keys
    )

    if unknown_keys:
        raise ValueError(
            "Unsupported backfill parameters: "
            f"{unknown_keys}"
        )

    missing_keys = sorted(
        allowed_keys
        - set(backfill_config.keys())
    )

    if missing_keys:
        raise ValueError(
            "Missing required backfill parameters: "
            f"{missing_keys}"
        )

    try:
        year = int(
            backfill_config["year"]
        )
        month = int(
            backfill_config["month"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "backfill year and month must be integers"
        ) from exc

    if month < 1 or month > 12:
        raise ValueError(
            f"Invalid backfill month: {month}"
        )

    window_start = date(
        year,
        month,
        1,
    )

    window_end_exclusive = _next_month(
        year,
        month,
    )

    return {
        "mode": "BACKFILL",
        "execution_scope": "GOLD_ONLY",

        "year": year,
        "months": [month],

        "window_start":
            window_start.isoformat(),

        "window_end_exclusive":
            window_end_exclusive.isoformat(),

        "logical_window": {
            "start":
                window_start.isoformat(),
            "end_exclusive":
                window_end_exclusive.isoformat(),
            "anchor": "PICKUP_DATETIME",
        },

        "read_scope": {
            "input_layer": "SILVER",
            "boundary_policy":
                "REUSE_EXISTING_SILVER_STATE",
        },

        "write_scope":
            "WINDOW_REPLACE",

        "state_mode":
            "STATELESS",

        "rerun_semantics":
            "REPLACE_LOGICAL_WINDOW_IDEMPOTENT",
    }


def build_current_execution_contract(
    settings: Mapping[str, Any],
    dag_run_conf: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    conf = dict(dag_run_conf or {})
    
    unsupported_overrides = sorted(
        WINDOW_OVERRIDE_KEYS.intersection(conf.keys())
    )

    if unsupported_overrides:
        raise ValueError(
            "Direct window overrides are not supported. "
            "Use the controlled backfill contract instead. "
            "Refusing unsafe DAG-run parameters: "
            f"{unsupported_overrides}"
        )

    backfill_config = conf.get(
        "backfill"
    )

    if backfill_config is not None:
        if not isinstance(
            backfill_config,
            Mapping,
        ):
            raise ValueError(
                "backfill must be an object "
                "with year and month"
            )

        return _build_single_month_backfill_contract(
            backfill_config
        )

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

    window_start = date(
        year,
        normalized_months[0],
        1,
    )

    window_end_exclusive = _next_month(
        year,
        normalized_months[-1],
    )

    previous_year, previous_month = _previous_month(
        year,
        normalized_months[0],
    )

    raw_file_months = [
        f"{previous_year}-{previous_month:02d}",
        *[
            f"{year}-{month:02d}"
            for month in normalized_months
        ],
    ]

    return {
        "mode": "CONFIGURED_FULL_REFRESH",
        "year": year,
        "months": normalized_months,
        "window_start": window_start.isoformat(),
        "window_end_exclusive": window_end_exclusive.isoformat(),
        "write_scope": "FULL_TABLE_OVERWRITE",
        "state_mode": "STATELESS",
        "rerun_semantics": "RECOMPUTE_CONFIGURED_WINDOW",
        "logical_window": {
            "start": window_start.isoformat(),
            "end_exclusive": window_end_exclusive.isoformat(),
            "anchor": "PICKUP_DATETIME",
        },
        "read_scope": {
            "raw_file_months": raw_file_months,
            "boundary_policy": "INCLUDE_PREVIOUS_MONTH",
        },
    }