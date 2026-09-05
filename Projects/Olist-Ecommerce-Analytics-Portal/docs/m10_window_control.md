# M10 - Window and Watermark Control

## Status

```text
U1 controller completed:      2026-08-15
Production-cycle deployment: 2026-09-05
```

M10 adds explicit governed processing state around the Olist dbt workload and deploys the Window Controller as the scheduled production entry point.

The central invariant is:

> A processing window advances the successful watermark only after the complete controlled workload succeeds.

If the workload fails:

```text
successful watermark stays unchanged
failed window stays identifiable
retry reuses the same window
```

---

## Why this exists

Before M10, the project could run dbt, persist monitoring evidence, review monitoring history, and serve analytical marts, but it did not keep durable production state describing which business-data window was being processed, which window completed successfully, which exact failed window should be retried, which attempt created which monitoring evidence, or whether another writer already changed state.

M10 adds that control plane.

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

Monitoring correlation:

```text
dbt/monitoring/monitoring_run_resolver.py
dbt/monitoring/resolve_monitoring_run.py
```

---

## BigQuery control tables

### `pipeline_control_state`

Important fields:

```text
pipeline_name
environment
state
cycle_id
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

### `pipeline_window_events`

This table is append-only and stores every accepted transition with cycle, window, attempt, state, version, retry, event, and error context.

The state table answers "where are we now?". The event table answers "how did we get here?".

---

## Cycle migration

M10 originally validated forward processing using 24-hour historical windows before the production-cycle design was finalized.

Existing control history was preserved and classified as:

```text
cycle_id = 0
```

The first deployed calendar-month production simulation begins with:

```text
cycle_id = 1
```

The migration preserved current state, `control_version`, successful watermark, event count, and audit history.

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

`QUARANTINED` is also defined.

---

## Explicit initialization

Runtime code does not silently create control state.

Initialization is performed separately through:

```text
bootstrap_window_control.py
```

Initial state:

```text
state = IDLE
control_version = 0
last_successful_window = NULL
active_attempt = NULL
```

A second bootstrap attempt is rejected.

---

## Production source bounds

```text
SOURCE_START = 2016-09-01T00:00:00+00:00
SOURCE_END   = 2018-11-01T00:00:00+00:00
```

These bounds define the historical production simulation.

---

## Calendar-month window rule

Production processing uses half-open calendar-month windows:

```text
[2016-09-01, 2016-10-01)
[2016-10-01, 2016-11-01)
[2016-11-01, 2016-12-01)
```

Within one cycle:

```text
next_window.start = last_successful_window.end
```

The next end is the following calendar-month boundary.

---

## Cycle rollover

If the current successful window ends before `SOURCE_END`, the controller stays in the same `cycle_id` and derives the next month.

If:

```text
last_successful_window_end == SOURCE_END
```

the controller starts:

```text
cycle_id = previous cycle_id + 1
window_start = SOURCE_START
```

The previous cycle is not rewound.

---

## Production-cycle interpretation

The bounded source range contains approximately 26 monthly windows.

With the current Scheduler cadence:

```text
0 * * * *
```

one successful historical cycle takes roughly 26 scheduled hours, assuming one successful monthly execution per trigger.

This creates an observable production simulation for the public portfolio system; it is not intended to mimic the original wall-clock chronology of Olist events.

---

## M11 boundary

A new `cycle_id` at `SOURCE_END` is normal production-cycle behavior. It is not the same thing as an arbitrary replay API.

M10 does not implement arbitrary old-month replay, multi-window backfill, backward watermark movement, independent replay state, or replay-versus-incremental consistency verification.

Those belong to M11.

---

## dbt window propagation

The controller exports:

```text
CONTROL_ATTEMPT_ID
CONTROL_WINDOW_START
CONTROL_WINDOW_END
```

`run_dbt_job.sh` passes the window to dbt variables and `int_orders_windowed` filters `stg_orders` using `order_purchase_timestamp`.

Dimensions remain full-history reference tables. Transactional facts use incremental `MERGE` with stable unique keys, allowing the same failed window to run again without blindly appending duplicate fact keys.

---

## Full-history compatibility mode

`run_dbt_job.sh` still supports execution without control-window variables:

```text
No control window supplied.
Running in full-history compatibility mode.
```

This compatibility mode is retained for historical/manual use.

It is **not** the current scheduled production entry point.

The deployed Cloud Run Job starts:

```text
python
/app/dbt/control/run_window_controller.py
```

---

## Scheduler and Cloud Run boundary

Current production path:

```text
Cloud Scheduler
        ↓
Cloud Run Job
        ↓
M10 Window Controller
        ↓
run_dbt_job.sh
        ↓
dbt / monitoring / M9
```

Scheduler configuration:

```text
resource: olist-dbt-daily-trigger
schedule: 0 * * * *
timezone: Europe/Helsinki
```

The historical resource name is retained even though the job is now hourly.

---

## Platform retry boundary

The Cloud Run Job is configured with:

```text
maxRetries = 0
```

Retry ownership stays inside the controller.

Controller retry preserves the same failed window and `cycle_id`, while creating a new attempt ID, incremented attempt number, retry lineage, state transition, and audit event.

---

## Success behavior

A successful attempt performs:

```text
RUNNING → IDLE
```

It moves `last_successful_window` to the completed monthly window, clears active attempt/error fields, increments `control_version`, and appends `WINDOW_SUCCEEDED`.

The successful watermark is `last_successful_window_end`.

---

## Failure behavior

A failed attempt performs:

```text
RUNNING → FAILED
```

It preserves the active processing window and failed attempt identity, records error evidence, increments `control_version`, appends `WINDOW_FAILED`, and does not advance the successful watermark.

---

## Retry behavior

Normal retry path:

```text
FAILED
   ↓ WINDOW_RETRY_SCHEDULED
WAITING_RETRY
   ↓ WINDOW_RETRY_STARTED
RUNNING
```

Retry preserves `window_start`, `window_end`, and `cycle_id`, but creates a new `attempt_id`, increments `attempt_number`, and sets `retry_of_attempt_id`.

---

## Compare-and-set protection

Every transition uses the expected `control_version`.

The update condition includes:

```text
pipeline_name
environment
control_version = expected_control_version
```

A stale writer updates zero rows and is rejected as `ConcurrentStateUpdateError`.

---

## Atomic state and audit persistence

State mutation and audit insertion happen inside one BigQuery transaction:

```text
BEGIN TRANSACTION
update current state using expected control_version
verify affected-row count
insert audit event
COMMIT
```

If any step fails, the transaction rolls back.

---

## M8 monitoring correlation

M10 adds `control_attempt_id` to `olist_monitoring.pipeline_runs`.

Correlation path:

```text
controller attempt_id
        ↓
pipeline_runs.control_attempt_id
        ↓
monitoring_run_id
```

The resolver requires exactly one matching row.

---

## M9 exact-run review

After M8 persistence:

```text
resolve exact monitoring_run_id
        ↓
run M9 with --monitoring-run-id
```

M9 remains deterministic for rule results. Optional Vertex AI only explains triggered findings.

---

## Validation environment

M10 controller behavior was validated independently using:

```text
environment = validation
dbt base dataset = olist_validation
```

This isolates non-production staging, intermediate, and marts outputs while still using the shared raw source.

---

## Historical cutover baseline

Before validating windowed processing, existing full-history marts were compared with source-aligned historical ranges for orders, order items, payments, and reviews.

Expected and existing keys/counts matched for the validated range.

---

## Original failure and retry validation

Before production monthly cutover, controller retry semantics were exercised using a controlled historical test window.

The audit history included:

```text
WINDOW_STARTED
WINDOW_FAILED
WINDOW_RETRY_SCHEDULED
WINDOW_RETRY_STARTED
WINDOW_FAILED
WINDOW_RETRY_SCHEDULED
WINDOW_RETRY_STARTED
WINDOW_SUCCEEDED
```

Three attempts used the same processing window. The successful watermark did not advance after either failed attempt and moved only after the third attempt succeeded.

---

## Real deployed production validation

The production deployment completed:

```text
Cloud Scheduler
→ Cloud Run Job
→ M10 controller
→ BigQuery state claim
→ monthly dbt build
→ dbt tests
→ M8 monitoring load
→ exact monitoring run resolution
→ M9 deterministic review
→ WINDOW_SUCCEEDED
→ successful watermark advance
```

Validated production window:

```text
cycle_id = 1
2016-09-01T00:00:00+00:00
→
2016-10-01T00:00:00+00:00
```

Audit history:

```text
WINDOW_STARTED
WINDOW_SUCCEEDED
```

The control-version transition was:

```text
0 → 1
1 → 2
```

Final state:

```text
state = IDLE
cycle_id = 1
control_version = 2
last_successful_window_start = 2016-09-01T00:00:00+00:00
last_successful_window_end   = 2016-10-01T00:00:00+00:00
active attempt = NULL
```

---

## Exact production monitoring evidence

The same controller attempt appeared in `olist_monitoring.pipeline_runs.control_attempt_id`.

The matching monitoring run completed with:

```text
22 / 22 models successful
96 / 96 tests passed
```

---

## Analytics watermark integration

The production analytical serving layer uses `last_successful_window_end` as its upper bound.

Therefore active or failed processing windows are not exposed as completed business data. Analytics advances only after `WINDOW_SUCCEEDED`.

---

## Analytics state-universe integration

A direct fact-row filter would remove states with no currently eligible orders and violate the Portal's 27-state integrity invariant.

The final design therefore uses:

```text
complete state universe
        ↓
watermark-filtered eligible orders
        ↓
LEFT JOIN
```

This preserves all states while keeping evidence-dependent metrics nullable when no observation exists.

---

## Unit tests

Current Python regression inventory:

```text
Window Controller:       52
Monitoring resolver:      5
M9 reviewer:             59
                         ---
Total:                  116
```

Controller tests cover models, state transitions, calendar-window derivation, cycle behavior, repository persistence, execution orchestration, retry semantics, and stale-writer handling.

---

## Useful commands

### Controller help

```bash
python dbt/control/run_window_controller.py --help
```

### Run controller tests

```bash
python -m unittest discover \\
  -s dbt/control/window_controller/tests \\
  -t dbt/control \\
  -v
```

### Run a new validation window

```bash
python dbt/control/run_window_controller.py \\
  --project-id "$DBT_PROJECT_ID" \\
  --dataset-id olist_control \\
  --pipeline-name olist-dbt-build-job \\
  --environment validation \\
  --dbt-dataset olist_validation \\
  --location EU \\
  --source-start 2016-09-01T00:00:00+00:00 \\
  --source-end 2018-11-01T00:00:00+00:00
```

### Retry the current failed window

```bash
python dbt/control/run_window_controller.py \\
  --project-id "$DBT_PROJECT_ID" \\
  --dataset-id olist_control \\
  --pipeline-name olist-dbt-build-job \\
  --environment validation \\
  --dbt-dataset olist_validation \\
  --location EU \\
  --source-start 2016-09-01T00:00:00+00:00 \\
  --source-end 2018-11-01T00:00:00+00:00 \\
  --retry
```

---

## Final M10 boundary

M10 is complete for:

- normal forward calendar-month windows
- explicit production `cycle_id`
- bounded source cycling
- successful watermark advancement
- failure without watermark advancement
- exact-window retry
- repeated retry
- append-only audit history
- stale-writer rejection
- transactional state + event persistence
- windowed dbt fact processing
- exact M8 monitoring correlation
- exact M9 review scope
- scheduled Cloud Run controller deployment
- hourly Cloud Scheduler invocation
- successful-watermark analytical serving
- full 27-state analytical integrity

Not included in M10:

- arbitrary historical replay
- multi-window backfill
- backward movement of the normal production watermark
- independent replay-state management
- replay versus incremental consistency verification
- automatic quarantine-release workflow
- alert delivery

Replay, backfill, and recovery orchestration belong to M11.
