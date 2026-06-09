from __future__ import annotations

import time

import altair as alt
import pandas as pd
import streamlit as st

from utils.quality import load_quality_report_artifact

REPORT_SOURCES = {
    "Pipeline Outputs": "pipeline_quality_report.json",
    "HSL Source Snapshot": "hsl_source_validation_report.json",
    "FMI Weather Snapshot": "fmi_source_validation_report.json",
}


def normalize_status(status: str | None) -> str:
    if not status:
        return "unknown"
    return str(status).replace("_", " ")


def render_status_message(label: str, report: dict | None) -> None:
    """Render one compact status card."""
    if report is None:
        st.info(f"**{label}**\n\nNo report available.")
        return

    status = report.get("status", "unknown")
    total_checks = len(report.get("checks", []))
    warnings = len(report.get("warnings", []))
    errors = len(report.get("errors", []))

    text = (
        f"**{label}**\n\n"
        f"Status: **{normalize_status(status)}**  \n"
        f"Checks: **{total_checks}**  \n"
        f"Warnings: **{warnings}** · Failed: **{errors}**"
    )

    if status == "passed":
        st.success(text)
    elif status == "passed_with_warnings":
        st.warning(text)
    elif status == "failed":
        st.error(text)
    else:
        st.info(text)


def infer_dataset_name(check_name: str, known_datasets: list[str]) -> str:
    """
    Infer dataset/source name from the check name.

    The validation report stores checks as names such as:
    gold_route_window_file_exists
    hsl_map_points_coordinate_bounds
    """
    for dataset in sorted(known_datasets, key=len, reverse=True):
        if check_name.startswith(f"{dataset}_"):
            return dataset
    return "unknown"


def infer_check_type(check_name: str, dataset_name: str) -> str:
    """Extract the validation rule type from a check name."""
    if dataset_name != "unknown" and check_name.startswith(f"{dataset_name}_"):
        return check_name.removeprefix(f"{dataset_name}_")
    return check_name


def checks_to_dataframe(report: dict | None) -> pd.DataFrame:
    """Convert report checks into a dashboard-friendly dataframe."""
    if report is None:
        return pd.DataFrame(
            columns=["dataset", "check_type", "name", "status", "severity", "details"]
        )

    record_count = report.get("record_count", {}) or {}
    known_datasets = list(record_count.keys())

    rows = []
    for check in report.get("checks", []):
        name = check.get("name", "")
        dataset = infer_dataset_name(name, known_datasets)
        check_type = infer_check_type(name, dataset)

        rows.append(
            {
                "dataset": dataset,
                "check_type": check_type,
                "name": name,
                "status": check.get("status", "unknown"),
                "severity": check.get("severity", "unknown"),
                "details": check.get("details", ""),
            }
        )

    return pd.DataFrame(rows)


def record_counts_to_dataframe(report: dict | None) -> pd.DataFrame:
    """Convert report record counts into a dataframe."""
    if report is None:
        return pd.DataFrame(columns=["dataset", "records"])

    record_count = report.get("record_count", {}) or {}
    rows = [{"dataset": key, "records": value} for key, value in record_count.items()]
    return pd.DataFrame(rows)


def render_horizontal_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
    height: int = 260,
) -> None:
    """Render a readable horizontal bar chart for long category labels."""
    if df.empty:
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:Q", title=None),
            y=alt.Y(
                f"{y_col}:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=260),
            ),
            tooltip=[y_col, x_col],
        )
        .properties(height=height)
    )

    if title:
        chart = chart.properties(title=title)

    st.altair_chart(chart, use_container_width=True)


def render_report_summary(
    label: str,
    report: dict | None,
    show_visual_breakdown: bool = True,
) -> None:
    """Render detailed summary for one validation report."""
    st.subheader(label)

    if report is None:
        st.info("No validation report is available for this scope.")
        return

    checks_df = checks_to_dataframe(report)
    records_df = record_counts_to_dataframe(report)

    total_checks = len(checks_df)
    passed = int((checks_df["status"] == "passed").sum()) if not checks_df.empty else 0
    warnings = (
        int((checks_df["status"] == "warning").sum()) if not checks_df.empty else 0
    )
    failed = int((checks_df["status"] == "failed").sum()) if not checks_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total checks", total_checks)
    col2.metric("Passed", passed)
    col3.metric("Warnings", warnings)
    col4.metric("Failed", failed)

    st.caption(
        f"Source: {report.get('source', 'unknown')} · "
        f"Last validation time: {report.get('ingest_time', 'unknown')}"
    )

    st.caption(
        "Check count refers to validation rules executed, not the number of input records."
    )

    if show_visual_breakdown and not checks_df.empty:
        st.markdown("#### Check result breakdown")

        status_counts = (
            checks_df["status"]
            .value_counts()
            .reindex(["passed", "warning", "failed"], fill_value=0)
            .rename_axis("status")
            .reset_index(name="checks")
        )

        render_horizontal_bar_chart(
            status_counts,
            x_col="checks",
            y_col="status",
            title=None,
            height=180,
        )

    if not records_df.empty:
        st.markdown("#### Datasets / snapshots covered")

        if show_visual_breakdown and len(records_df) > 1:
            records_chart_df = records_df.sort_values("records", ascending=True)
            render_horizontal_bar_chart(
                records_chart_df,
                x_col="records",
                y_col="dataset",
                title=None,
                height=max(260, len(records_chart_df) * 36),
            )

        with st.expander("Dataset / snapshot record counts", expanded=False):
            st.dataframe(records_df, use_container_width=True, hide_index=True)

    if not checks_df.empty:
        if show_visual_breakdown:
            st.markdown("#### Checks by dataset")

            checks_by_dataset = (
                checks_df.groupby(["dataset", "status"])
                .size()
                .reset_index(name="checks")
            )

            pivot = checks_by_dataset.pivot_table(
                index="dataset",
                columns="status",
                values="checks",
                aggfunc="sum",
                fill_value=0,
            )

            for col in ["passed", "warning", "failed"]:
                if col not in pivot.columns:
                    pivot[col] = 0

            pivot = pivot[["passed", "warning", "failed"]]
            render_checks_by_dataset_chart(pivot)

        warnings_df = checks_df[checks_df["status"] == "warning"].copy()
        failed_df = checks_df[checks_df["status"] == "failed"].copy()

        if not failed_df.empty:
            st.markdown("#### Failed checks")
            st.dataframe(
                failed_df[["dataset", "check_type", "severity", "details"]],
                use_container_width=True,
                hide_index=True,
            )

        if not warnings_df.empty:
            st.markdown("#### Warning checks")
            st.dataframe(
                warnings_df[["dataset", "check_type", "severity", "details"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Full check details")

        dataset_options = ["All"] + sorted(checks_df["dataset"].dropna().unique())
        status_options = ["All"] + sorted(checks_df["status"].dropna().unique())

        col_filter_1, col_filter_2 = st.columns(2)
        selected_dataset = col_filter_1.selectbox(
            "Filter by dataset",
            dataset_options,
            key=f"{label}_dataset_filter",
        )
        selected_status = col_filter_2.selectbox(
            "Filter by status",
            status_options,
            key=f"{label}_status_filter",
        )

        filtered_df = checks_df.copy()

        if selected_dataset != "All":
            filtered_df = filtered_df[filtered_df["dataset"] == selected_dataset]

        if selected_status != "All":
            filtered_df = filtered_df[filtered_df["status"] == selected_status]

        st.dataframe(
            filtered_df[["dataset", "check_type", "status", "severity", "details"]],
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    metadata = report.get("metadata", {}) or {}
    if metadata:
        with st.expander("Report metadata", expanded=False):
            metadata_df = pd.DataFrame(
                [{"key": key, "value": value} for key, value in metadata.items()]
            )
            st.dataframe(metadata_df, use_container_width=True, hide_index=True)


def render_checks_by_dataset_chart(pivot: pd.DataFrame) -> None:
    """Render checks by dataset as a horizontal stacked bar chart."""
    if pivot.empty:
        return

    chart_df = pivot.reset_index().melt(
        id_vars="dataset",
        value_vars=["passed", "warning", "failed"],
        var_name="status",
        value_name="checks",
    )

    chart_df = chart_df[chart_df["checks"] > 0]

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("checks:Q", title="Checks"),
            y=alt.Y(
                "dataset:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=260),
            ),
            color=alt.Color("status:N", title="Status"),
            tooltip=["dataset", "status", "checks"],
        )
        .properties(height=max(320, len(pivot) * 42))
    )

    st.altair_chart(chart, use_container_width=True)


start_time = time.time()

st.title("Data Quality & Source Validation")
st.caption(
    "Read-only view of validation reports generated during scheduled snapshot refreshes. "
    "The reports cover pipeline output quality checks and HSL/FMI source compatibility checks. "
    "This page displays generated validation artifacts; it does not run checks, call live APIs, or modify pipeline outputs."
)

reports = {
    label: load_quality_report_artifact(filename)
    for label, filename in REPORT_SOURCES.items()
}

st.subheader("Validation coverage overview")

card1, card2, card3 = st.columns(3)
with card1:
    render_status_message("Pipeline Outputs", reports["Pipeline Outputs"])
with card2:
    render_status_message("HSL Source Snapshot", reports["HSL Source Snapshot"])
with card3:
    render_status_message("FMI Weather Snapshot", reports["FMI Weather Snapshot"])

st.divider()

tab_pipeline, tab_hsl, tab_fmi = st.tabs(
    ["Pipeline Outputs", "HSL Source Snapshot", "FMI Weather Snapshot"]
)

with tab_pipeline:
    render_report_summary(
        "Pipeline output validation",
        reports["Pipeline Outputs"],
        show_visual_breakdown=True,
    )

with tab_hsl:
    render_report_summary(
        "HSL source compatibility validation",
        reports["HSL Source Snapshot"],
        show_visual_breakdown=False,
    )

with tab_fmi:
    render_report_summary(
        "FMI weather source compatibility validation",
        reports["FMI Weather Snapshot"],
        show_visual_breakdown=False,
    )

st.caption(
    "Validation check count refers to the number of validation rules executed, "
    "not the number of input records."
)

st.caption(f"Page rendered in {time.time() - start_time:.2f}s")
