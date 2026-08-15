# GitHub Projects Plan

## Document note

This file started as the project-board plan during the early milestones.

It is kept as a lightweight planning reference. The current milestone status is maintained in:

```text
docs/project_management.md
```

Do not use older issue placeholders in this file as proof of the current implementation state.

---

## Board name

```text
Olist E-Commerce Analytics & Pipeline Monitoring Portal
```

---

## Suggested board columns

```text
Backlog
Ready
In Progress
Review / Validation
Done
```

The exact GitHub UI can change. The important point is to keep current work separate from completed work and future ideas.

---

## Milestones

| Milestone | Focus | Current status |
|---|---|---:|
| M1 | Project setup and source understanding | Completed |
| M2 | BigQuery raw layer | Completed |
| M3 | Staging planning | Completed |
| M4 | dbt staging | Completed |
| M5 | Dimensional marts | Completed |
| M6 | Documentation cleanup | Completed |
| M7 | Cloud orchestration | Completed |
| M8 | Pipeline monitoring | Completed |
| M9 | Pipeline quality reviewer | Completed |
| M10 U1 | Window / watermark control | Completed |
| M10 portal / analytics | Portal and geospatial analytics | In progress |
| M11 | Replay / backfill / recovery | Planned |

---

## Labels

Useful labels:

```text
area:docs
area:bigquery
area:dbt
area:monitoring
area:control
area:portal
area:analytics
area:cloud

type:feature
type:fix
type:validation
type:cleanup

priority:high
priority:medium
priority:low
```

Keep labels small and useful. Add a label only when it helps filtering or planning.

---

## Completed work summary

### M1-M3

- repository setup
- source inventory
- BigQuery naming and raw-load planning
- staging design and cleanup rules

### M4-M6

- dbt staging models
- dbt tests
- dimensional model and marts
- dbt docs and lineage review
- README and architecture cleanup

### M7-M9

- Dockerized dbt runtime
- Cloud Run Job and Cloud Scheduler
- append-only dbt monitoring tables
- pipeline review rules R001-R006
- historical runtime / row-count comparisons where evidence exists
- optional explanation for triggered findings

### M10 U1

- BigQuery control-state table
- append-only window event table
- explicit state initialization
- forward windows
- retry attempts
- windowed dbt transaction path
- incremental fact `MERGE`
- exact M8/M9 run correlation
- BigQuery stale-version protection
- real success, failure, retry, and CAS validation

---

## Current M10 board focus

Suggested current issues:

```text
M10-P1  Define portal route and API boundaries
M10-P2  Build overview / reliability vertical slice
M10-A1  Create state-level Brazil analytics aggregate
M10-A2  Add first CARTO + deck.gl state map
M10-A3  Link map selection to KPIs and detail data
M10-D1  Update portal and deployment documentation
```

Keep the first portal and analytics slices small enough to validate end to end.

---

## M11 placeholders

Keep M11 issues as future placeholders until M10 is stable.

```text
M11-R1  Replay one historical window
M11-R2  Backfill several windows
M11-R3  Resume after partial failure
M11-R4  Validate idempotent business writes
M11-R5  Compare incremental and replay results
M11-R6  Keep replay state separate from forward watermark state
```

---

## Working rule

For each issue:

1. define the problem
2. define the acceptance check
3. implement the smallest useful unit
4. validate it
5. document the important result
6. move it to Done only after validation
