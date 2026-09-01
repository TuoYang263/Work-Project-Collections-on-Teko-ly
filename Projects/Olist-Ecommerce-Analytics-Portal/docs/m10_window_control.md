# M10 U1 - Window and Watermark Control

## Status

```text
Completed: 2026-08-15
```

M10 U1 adds explicit processing state around the Olist dbt workload.

The main rule is:

> A window advances only after the full controlled workload succeeds.

If the workload fails, the watermark stays where it was and the same window can be retried with a new attempt ID.

---

## Why this was added

Before M10, the project could run dbt, store monitoring history, and review that history. It did not yet keep a durable record of which business-data window had been successfully processed.

M10 U1 adds that missing control state.

It answers four questions:

1. What window should run next?
2. Has that window succeeded?
3. If it failed, which exact window should be retried?
4. Can two writers update the same control state at the same time?

---

## Main components

```text
dbt/control/bootstrap_window_control.py
dbt/control/run_window_controller.py
dbt/control/window_controller/models.py
dbt/control/window_controller/transitions.py
dbt/control/window_controller/service.py
dbt/control/window_controller/repository.py
dbt/control/window_controller/controller.py
```

SQL setup:

```text
dbt/sql/control/create_olist_control_dataset.sql
dbt/sql/control/create_window_control_tables.sql
dbt/sql/control/validate_cutover_baseline.sql
```

Monitoring correlation:

```text
dbt/monitoring/monitoring_run_resolver.py
dbt/monitoring/resolve_monitoring_run.py
```

---

## BigQuery control tables

### `pipeline_control_state`

This table stores the current state for one pipeline/environment pair.

Important fields:

```text
pipeline_name
environment
state
last_successful_window_start
last_successful_window_end
active_window_start
active_window_end
active_attempt_id
active_attempt_number
active_retry_of_attempt_id
control_version
last_error_code
last_error_message
updated_at
```

The row is updated in place because it represents current state.

### `pipeline_window_events`

This table is append-only.

It stores every accepted transition with:

```text
event_id
attempt_id
pipeline_name
environment
window_start
window_end
attempt_number
from_state
to_state
from_control_version
to_control_version
event_type
event_time
retry_of_attempt_id
error_code
error_message
metadata_json
```

The current state table answers "where are we now?".

The event table answers "how did we get here?".

---

## State model

```text
IDLE
  ↓ start new window
RUNNING
  ├── success → IDLE
  └── failure → FAILED
                  ↓ schedule retry
             WAITING_RETRY
                  ↓ start retry
               RUNNING
```

`QUARANTINED` is also defined as a state.

U1 does not add automatic retry limits, automatic quarantine, or a release workflow. Those controls can be added later without changing the basic forward-window rule.

---

## Explicit initialization

The runtime does not create control state automatically.

Initialization is done with:

```text
bootstrap_window_control.py
```

Expected first state:

```text
state = IDLE
control_version = 0
last_successful_window = NULL
active_attempt = NULL
```

A second initialization attempt is rejected.

---

## Window rule

The first window starts from an explicit `initial_start`.

After the first success:

```text
next_window.start = last_successful_window.end
```

For a 24-hour window:

```text
[start, start + 24h)
```

The end is exclusive.

This keeps adjacent windows continuous without overlap.

---

## dbt windowing

The controller sets:

```text
CONTROL_ATTEMPT_ID
CONTROL_WINDOW_START
CONTROL_WINDOW_END
```

`run_dbt_job.sh` passes the window to dbt as variables.

`int_orders_windowed` applies the window to `stg_orders` using `order_purchase_timestamp`.

```text
stg_orders
    ↓
int_orders_windowed
    ↓
current order IDs
 ↙      ↓       ↘
items payments reviews
 ↘      ↓       ↙
incremental facts
```

Dimensions remain full-history tables.

Facts use incremental `MERGE` with stable unique keys.

This matters for retry: the same window can run again without simply appending another copy of the same fact key.

---

## Full-history compatibility mode

`run_dbt_job.sh` can still run without control window variables.

In that case it prints:

```text
No control window supplied.
Running in full-history compatibility mode.
```

This keeps the existing M7/M8 Cloud Run path working while M10 is introduced separately.

The current scheduled Cloud Run Job still uses this script as its container entry point.

The scheduled job has not yet been changed to start the M10 controller.

---

## Success behavior

A successful attempt changes:

```text
RUNNING → IDLE
```

It then:

- moves `last_successful_window` to the completed window
- clears `active_attempt`
- clears the last error
- increments `control_version`
- writes `WINDOW_SUCCEEDED`

The watermark is the end of the last successful window.

---

## Failure behavior

A failed attempt changes:

```text
RUNNING → FAILED
```

It:

- keeps the active window
- keeps the failed attempt ID
- records the error
- increments `control_version`
- writes `WINDOW_FAILED`
- does not advance the last successful window

This is the key safety rule for forward processing.

---

## Retry behavior

A retry starts from `FAILED` or `WAITING_RETRY`.

Normal path:

```text
FAILED
  ↓ WINDOW_RETRY_SCHEDULED
WAITING_RETRY
  ↓ WINDOW_RETRY_STARTED
RUNNING
```

The retry uses:

```text
same window
new attempt_id
attempt_number + 1
retry_of_attempt_id = previous attempt_id
```

If the runtime stops after `WAITING_RETRY` is already stored, the next retry call can continue from that state instead of scheduling it twice.

---

## BigQuery compare-and-set protection

Every control transition uses the expected `control_version`.

The update condition includes:

```text
pipeline_name
environment
control_version = expected_control_version
```

If the version is stale, zero rows are updated.

The repository treats this as a concurrent state update and rejects the transition.

This prevents an older writer from overwriting a newer state.

---

## Atomic state and audit write

The repository writes the state update and audit event inside one BigQuery transaction.

```text
BEGIN TRANSACTION
    update current state with expected version
    check affected row count
    insert audit event
COMMIT
```

If any step fails, the transaction rolls back.

A stale-writer validation confirmed that:

- the stale state update was rejected
- the current state stayed unchanged
- the test audit event was not inserted

---

## M8 correlation

M10 adds this field to `olist_monitoring.pipeline_runs`:

```text
control_attempt_id
```

The artifact parser reads `CONTROL_ATTEMPT_ID` from the runtime environment and stores it with the monitoring run.

The resolver then uses:

```text
control_attempt_id
        ↓
pipeline_runs
        ↓
monitoring_run_id
```

It requires exactly one matching row.

This gives the controller a direct link to the monitoring evidence created by its own attempt.

---

## M9 exact-run review

In window-controlled mode, after M8 finishes:

```text
resolve exact monitoring_run_id
        ↓
run M9 with --monitoring-run-id
```

The runtime does not guess the latest monitoring run.

The M9 review remains deterministic for rule results. Vertex AI is optional and only explains triggered findings.

A deterministic `TRIGGERED` finding is not by itself a controller failure. The controlled workload fails only if the reviewer process itself cannot complete as required by the script.

---

## Validation environment

M10 was tested with:

```text
environment = validation
dbt base dataset = olist_validation
```

This creates isolated staging, intermediate, and marts datasets for validation while using the shared raw source.

The runtime blocks a non-production controller run from using the default `olist` dbt dataset.

---

## Cutover baseline check

Before running historical validation windows, the existing full-history marts were compared with the source-aligned historical range.

The check covered:

```text
orders
order items
payments
reviews
```

For the checked range, expected and existing counts matched with no missing or unexpected keys.

This provided evidence that the existing full-history marts were a safe reference point before validating the new windowed path.

The project did not force a production watermark reset or historical backfill as part of M10 U1.

---

## Real happy-path validation

A controlled historical window completed this full path:

```text
controller
→ BigQuery state claim
→ windowed dbt build
→ dbt tests
→ M8 monitoring load
→ exact monitoring run resolution
→ M9 review
→ WINDOW_SUCCEEDED
→ watermark advance
```

Current dbt validation result:

```text
22 models
96 tests
118 / 118 PASS
```

A second successful window confirmed that the controller derived the next window from the previous successful window end.

---

## Real failure and retry validation

The window:

```text
2016-09-06 00:00:00
→ 2016-09-07 00:00:00
```

was used for the retry test.

The audit history was:

```text
v4  → v5   WINDOW_STARTED          attempt 1
v5  → v6   WINDOW_FAILED           attempt 1
v6  → v7   WINDOW_RETRY_SCHEDULED  attempt 1
v7  → v8   WINDOW_RETRY_STARTED    attempt 2
v8  → v9   WINDOW_FAILED           attempt 2
v9  → v10  WINDOW_RETRY_SCHEDULED  attempt 2
v10 → v11  WINDOW_RETRY_STARTED    attempt 3
v11 → v12  WINDOW_SUCCEEDED        attempt 3
```

The same window was used for all three attempts.

Attempt 2 linked to attempt 1. Attempt 3 linked to attempt 2.

The watermark did not move after attempt 1 or attempt 2 failed.

After attempt 3 succeeded, the final state was:

```text
state = IDLE
control_version = 12
last_successful_window = 2016-09-06 → 2016-09-07
active attempt = NULL
last error = NULL
```

---

## Real stale-write validation

After the final state reached version 12, a test writer tried to persist a transition using expected version 11.

Result:

```text
ConcurrentStateUpdateError
```

The follow-up check showed:

```text
state = IDLE
control_version = 12
cas_probe_event_count = 0
```

This proves the stale transaction did not change the state or append a false event.

---

## Unit tests

Current test inventory:

```text
window controller:           52
monitoring-run resolver:      5
M9 reviewer:                 53
                            ----
total:                      110
```

Controller tests cover models, allowed transitions, service rules, repository behavior, execution flow, and retry behavior.

---

## Useful commands

### Controller help

```bash
python dbt/control/run_window_controller.py --help
```

### Run controller tests

```bash
python -m unittest discover \
  -s dbt/control/window_controller/tests \
  -t dbt/control \
  -v
```

### Run a new validation window

```bash
python dbt/control/run_window_controller.py \
  --project-id "$DBT_PROJECT_ID" \
  --dataset-id olist_control \
  --pipeline-name olist-dbt-build-job \
  --environment validation \
  --dbt-dataset olist_validation \
  --location EU \
  --initial-start 2016-09-04T00:00:00+00:00 \
  --window-size-hours 24
```

### Retry the current failed window

```bash
python dbt/control/run_window_controller.py \
  --project-id "$DBT_PROJECT_ID" \
  --dataset-id olist_control \
  --pipeline-name olist-dbt-build-job \
  --environment validation \
  --dbt-dataset olist_validation \
  --location EU \
  --retry
```

---

## U1 boundary

M10 U1 is complete for:

- normal forward windows
- successful watermark advance
- failure without watermark advance
- same-window retry
- repeated retry
- audit history
- stale-write rejection
- windowed dbt facts
- exact M8/M9 run correlation

Not included in U1:

- automatic retry scheduling
- automatic retry limit
- full quarantine workflow
- replay
- backfill
- backward watermark movement
- portal screens
- alert delivery

Replay and backfill belong to M11. Portal and analytics work continues in M10.
