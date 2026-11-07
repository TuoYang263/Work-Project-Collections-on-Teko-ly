"""Private, AI-off Streamlit view of validated historical incident snapshots.

Run from airflow_pipeline/ and bind to localhost, not a public service:
  streamlit run replay_app.py --server.address 127.0.0.1
"""

import os
from pathlib import Path

import streamlit as st

from scripts.diagnostics.incident_replay import (
    ReplayError,
    available_cases,
    load_verified_replay,
)

st.set_page_config(page_title="NYC V2 | Incident Replay", layout="wide")
st.title("Incident Investigation")
st.caption("Offline evidence replay · AI OFF · Read-only")

archive_root = os.environ.get("NYC_V2_ARCHIVE_ROOT")
index_path = os.environ.get("NYC_V2_TRUSTED_INDEX")
if not archive_root or not index_path:
    st.warning(
        "Configure NYC_V2_ARCHIVE_ROOT and NYC_V2_TRUSTED_INDEX to open pinned cases."
    )
    st.stop()

try:
    choices = available_cases(Path(index_path))
except ReplayError as exc:
    st.error(f"{exc.code}: {exc}")
    st.stop()

if not choices:
    st.info("No incident revisions are registered in the trusted index yet.")
    st.stop()

def fixed_choice(label, options, *, state_key, format_func, page_size=6):
    """Pick a registered value without any text-editable or free-form field.

    For up to six entries, use radio buttons. Larger collections use
    six fixed buttons per page. Page changes never select a different case.
    Selection persists in Streamlit session state across reruns.
    """
    options = list(options)
    if not options:
        st.error(f"No registered options for {label}.")
        st.stop()

    current = st.session_state.get(state_key)
    if current not in options:
        current = options[0]
        st.session_state[state_key] = current

    if len(options) <= page_size:
        choice = st.radio(
            label,
            options=options,
            index=options.index(current),
            format_func=format_func,
            horizontal=len(options) <= 5,
            key=f"{state_key}_radio",
        )
        st.session_state[state_key] = choice
        return choice

    page_count = (len(options) + page_size - 1) // page_size
    page_key = f"{state_key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = options.index(current) // page_size
    page = min(max(st.session_state[page_key], 0), page_count - 1)
    st.session_state[page_key] = page

    st.markdown(f"**{label}**")
    previous, indicator, following = st.columns([1, 2, 1])

    def set_page(target_page):
        st.session_state[page_key] = target_page

    with previous:
        st.button(
            "◀ Previous", key=f"{state_key}_previous", disabled=page == 0,
            on_click=set_page, args=(page - 1,), use_container_width=True,
        )
    with indicator:
        st.caption(f"Page {page + 1} of {page_count} · {len(options)} options")
    with following:
        st.button(
            "Next ▶", key=f"{state_key}_next",
            disabled=page == page_count - 1,
            on_click=set_page, args=(page + 1,), use_container_width=True,
        )

    def select_option(value):
        st.session_state[state_key] = value

    visible = options[page * page_size:(page + 1) * page_size]
    for i in range(0, len(visible), 2):
        columns = st.columns(2)
        for offset, value in enumerate(visible[i:i + 2]):
            with columns[offset]:
                st.button(
                    format_func(value),
                    key=f"{state_key}_option_{page * page_size + i + offset}",
                    type="primary" if value == current else "secondary",
                    use_container_width=True,
                    on_click=select_option, args=(value,),
                )
    st.caption(f"Selected: {format_func(st.session_state[state_key])}")
    return st.session_state[state_key]


case_map = {item["case_id"]: item for item in choices}
case_ids = list(case_map)
label_counts = {}
for case_id in case_ids:
    label = case_map[case_id]["label"]
    label_counts[label] = label_counts.get(label, 0) + 1


def case_display(case_id):
    label = case_map[case_id]["label"]
    if len(case_ids) > 6 or label_counts[label] > 1:
        return f"{label} · {case_id}"
    return label


left, right = st.columns(2)
with left:
    selected_case = fixed_choice(
        "Incident Case", case_ids,
        state_key="replay_selected_case", format_func=case_display,
    )
with right:
    selected_revision = fixed_choice(
        "Snapshot Revision", case_map[selected_case]["revisions"],
        state_key=f"replay_revision_{selected_case}",
        format_func=lambda revision: f"Revision {revision:04d}",
    )

try:
    replay = load_verified_replay(
        archive_root=Path(archive_root),
        trusted_index_path=Path(index_path),
        case_id=selected_case,
        revision=selected_revision,
    )
except ReplayError as exc:
    st.error(f"{exc.code}: {exc}")
    st.caption(
        "This case is INCONCLUSIVE. No unverified evidence or live fallback is shown."
    )
    st.stop()

source = replay["source"]
st.subheader(replay["label"])
st.write({
    "Original DAG": source.get("dag_id"),
    "Run ID": source.get("run_id"),
    "Final run status": source.get("status"),
    "Failure stage": source.get("failure_stage"),
    "Revision": replay["revision"],
})
st.success(f"{replay['verification_status']} · {replay['provenance']}")
st.caption(
    f"Capture status: {replay['capture_status']}. "
    "Local integrity verification is NOT independent-backup or "
    "SNAPSHOT_VERIFIED acceptance."
)

execution, provenance = st.tabs(["Observed execution", "Pinned evidence"])
with execution:
    st.markdown("#### Recorded TaskInstance snapshots")
    if replay["task_instances"]:
        st.dataframe(
            replay["task_instances"], use_container_width=True, hide_index=True
        )
    else:
        st.info("No TaskInstance rows captured.")

    st.markdown("#### Recorded TaskFail rows")
    if replay["task_failures"]:
        st.dataframe(replay["task_failures"], use_container_width=True, hide_index=True)
    else:
        st.caption(
            "No TaskFail rows captured. This alone does not prove there were no errors."
        )

    st.markdown("#### Captured attempt-log markers")
    if not replay["log_attempts"]:
        st.info("No log attempts are present in this snapshot.")
    for attempt in replay["log_attempts"]:
        if not isinstance(attempt, dict):
            st.warning("Invalid attempt projection; see evidence completeness.")
            continue
        number = attempt.get("attempt_number", "unknown")
        source_path = attempt.get("source_path", "source unknown")
        with st.expander(f"Attempt {number} · {source_path}"):
            excerpts = attempt.get("excerpts")
            if not isinstance(excerpts, list) or not excerpts:
                st.caption("No approved excerpts captured for this attempt.")
            else:
                for item in excerpts:
                    if isinstance(item, dict):
                        st.caption(
                            f"{item.get('event', 'UNKNOWN')} · "
                            f"source line {item.get('line_number', '?')}"
                        )
                        st.code(str(item.get("excerpt", "")), language=None)
    st.caption(
        "Observed markers only; not a reconstruction of complete attempt history."
    )

with provenance:
    st.markdown("#### Exact pinned fingerprints")
    st.code(
        f"Snapshot SHA256: {replay['snapshot_sha256']}\n"
        f"Manifest SHA256: {replay['manifest_sha256']}"
    )
    st.markdown("#### Capture scope and limitations")
    st.json(replay["capture_scope"])
    for limitation in replay["limitations"]:
        st.caption(str(limitation))
    st.markdown("#### Verified manifest")
    st.json(replay["manifest"])

st.caption(
    "No Airflow, PostgreSQL, AWS, model inference, DAG trigger or write operations."
)
