from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

HELSINKI_TZ = ZoneInfo("Europe/Helsinki")


def format_minutes_as_age(minutes: int | float | None) -> str:
    """
    Convert a minute value into a compact human-readable age string
    """
    if minutes is None or pd.isna(minutes):
        return "N/A"

    minutes = int(max(0, minutes))

    if minutes < 60:
        return f"{minutes} min"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    if hours < 24:
        if remaining_minutes == 0:
            return f"{hours}h"
        return f"{hours}h {remaining_minutes}m"

    days = hours // 24
    remaining_hours = hours % 24

    if remaining_hours == 0:
        return f"{days}d"
    return f"{days}d {remaining_hours}h"


def compute_snapshot_age_minutes(snapshot_time) -> int | None:
    """
    Compute snapshot age in minutes against current Helsinki time.

    This helper expects a timestamp from exported dashboard data.
    It does not query live source systems.
    """
    if snapshot_time is None or pd.isna(snapshot_time):
        return None

    ts = pd.to_datetime(snapshot_time, errors="coerce", utc=True)

    if pd.isna(ts):
        return None

    snapshot_local = ts.tz_convert(HELSINKI_TZ)
    now_local = datetime.now(HELSINKI_TZ)

    return max(
        0,
        int((now_local - snapshot_local.to_pydatetime()).total_seconds() / 60),
    )


def explain_snapshot_status(snapshot_age_minutes: int | None) -> str:
    """
    Explain snapshot age in the context of a scheduled dashboard.
    """
    if snapshot_age_minutes is None:
        return (
            "Snapshot status is unavailable because the latest exported data timestamp "
            "could not be determined."
        )

    age_text = format_minutes_as_age(snapshot_age_minutes)

    if snapshot_age_minutes <= 180:
        return (
            f"The latest exported snapshot is {age_text} old. "
            "This is consistent with the scheduled snapshot refresh model."
        )

    if snapshot_age_minutes <= 24 * 60:
        return (
            f"The latest exported snapshot is {age_text} old. "
            "This is consistent with the daily scheduled refresh model."
        )

    return (
        f"The latest exported snapshot is {age_text} old. "
        "The next step would be to check the latest Databricks scheduled run and Blob export time."
    )


def explain_pipeline_metrics(
    total_events: int | None,
    avg_ingest_delay_sec: float | None,
    dq_status: str | None,
) -> list[str]:
    """
    Generate deterministic explanations for Pipeline Overview metrics.

    These insights only explain precomputed Gold-layer metrics.
    They do not infer root causes or live operational status.
    """
    insights: list[str] = []

    if total_events is not None and total_events > 0:
        insights.append(
            f"The latest valid transit window contains {total_events:,} processed events."
        )
    else:
        insights.append(
            "No valid transit events are available in the latest window, so the pipeline output should be checked."
        )

    if avg_ingest_delay_sec is not None and pd.notna(avg_ingest_delay_sec):
        insights.append(
            f"Average ingest delay is about {avg_ingest_delay_sec:.1f} seconds in the latest valid transit window."
        )
    else:
        insights.append(
            "Average ingest delay is unavailable for the latest valid transit window."
        )

    if dq_status:
        insights.append(
            f"The current data quality status is shown as {dq_status} based on available Gold-layer indicators."
        )

    insights.append(
        "These insights are based only on precomputed Gold-layer metrics. "
        "They do not infer live operational causes."
    )

    return insights


def explain_route_metrics(
    selected_route: str,
    observed_events: int | None,
    avg_delay,
    late_rate,
) -> list[str]:
    """
    Generate deterministic explanations for Route Performance metrics.
    """
    insights: list[str] = []

    route_label = (
        "all available routes" if selected_route == "All" else f"route {selected_route}"
    )

    insights.append(
        f"The current route view summarizes {route_label} using the latest exported Gold-layer route window."
    )

    if observed_events is not None:
        insights.append(
            f"The latest route window includes {observed_events:,} observed events."
        )

    if avg_delay != "N/A":
        insights.append(
            f"Average delay in the latest route window is {avg_delay} seconds."
        )

    if late_rate != "N/A":
        insights.append(
            f"The late-rate indicator is {late_rate} for the latest route window."
        )

    insights.append(
        "Route insights are descriptive only and should be read as snapshot-level summaries, not live service alerts."
    )

    return insights


def render_insight_box(st, title: str, insights: list[str]) -> None:
    """
    Render a compact expandable insight section in Streamlit.
    """
    with st.expander(title, expanded=False):
        for item in insights:
            st.markdown(f"- {item}")
