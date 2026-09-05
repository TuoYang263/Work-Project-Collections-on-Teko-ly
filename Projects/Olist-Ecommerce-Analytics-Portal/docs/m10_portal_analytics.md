# M10 Portal and Analytics

## Status

```text
Portal / analytics implementation: 2026-08-30
Final production close-out:        2026-09-05
```

This document records the final M10 Portal and analytics scope.

\---

## What M10 delivers

M10 connects operational control and business analytics into one governed product.

```text
calendar-month production control
        ↓
successful watermark
        ↓
BigQuery analytical serving
        ↓
business decisions \+ reliability evidence
        ↓
Next.js Portal
```

The key user-facing rule is:

\> The Portal may expose business data only from successfully completed production scope.

Operational `/overview` therefore shows current controller state.

Business-facing `/analytics` exposes data only through the last successful watermark.

M10 contains two major concerns:

\- Window and watermark control
\- Operational and business-facing Portal functionality

Detailed control design is documented in [`m10_window_control.md`](m10_window_control.md).

\---

## Portal scope

The Portal uses Next.js, React, and TypeScript.

Main routes:

```text
/overview
/analytics
/reliability
/findings/[findingId]
/health
```

The layout uses one shared navigation shell. Pages are server-rendered where possible and client-side interaction is used where it materially improves the product, such as Brazil state selection on the analytics map.

\---

## Server-side data path

```text
Next.js Server Component
        ↓
Service
        ↓
Repository
        ↓
BigQuery
```

Repositories perform data access. Services validate, enforce integrity assumptions, and map data into application types. Pages render validated results.

\---

## Overview

`/overview` shows the current operational state of the M10 Window Controller.

It answers:

\- what state the pipeline is in
\- which production cycle is active
\- whether a processing attempt is active
\- which window is currently being processed
\- which window completed successfully most recently
\- what the current control version is
\- whether there is current failure/retry evidence

The page reads governed control state from BigQuery.

\---

## Reliability

`/reliability` shows persisted deterministic M9 review evidence.

M9 preserves three evaluation states:

```text
PASS
TRIGGERED
NOT_EVALUATED
```

`NOT_EVALUATED` remains visible when evidence cannot support a deterministic decision.

Triggered findings link to `/findings/[findingId]`.

\---

## Finding identifier boundary

Finding IDs are carried in the URL path. The page decodes the path segment once and the service validates the decoded identifier before it can reach the repository.

BigQuery access remains parameterized.

\---

## Analytics

`/analytics` is a Brazil state-level business decision view.

It combines:

\- order count
\- GMV
\- average order value
\- delivery observation count
\- late-delivery rate
\- reviewed-order count
\- average review score
\- deterministic business action
\- business priority
\- historical statistical review-risk diagnostic

The map covers all 27 Brazilian states.

Selecting a state updates linked KPI, business-action, and diagnostic cards.

The geospatial implementation uses MapLibre, react-map-gl, deck.gl, and a CARTO basemap.

\---

## Successful-watermark analytical scope

The current KPI and state-summary serving layer is bounded by:

```text
last_successful_window_end
```

The serving layer intentionally uses the **successful watermark**, not the active processing window.

Behavior:

```text
RUNNING
→ Portal remains on previous successful analytical scope

FAILED
→ Portal remains on previous successful analytical scope

WINDOW_SUCCEEDED
→ successful watermark advances
→ Portal analytical scope advances
```

This prevents partially processed or failed business data from appearing as completed analytics.

\---

## Cumulative-within-cycle semantics

The analytical serving layer uses an upper successful-watermark bound:

```text
order_purchase_timestamp
\<
last_successful_window_end
```

Within one production cycle this produces a cumulative analytical view from the configured historical source beginning through the successful watermark.

Example:

```text
after Sep success
→ Sep scope

after Oct success
→ Sep \+ Oct scope

after Nov success
→ Sep \+ Oct \+ Nov scope
```

When a new production cycle begins, Analytics remains on the previous successful scope while the first new window is running. Once that first monthly window succeeds, the successful watermark returns to the first calendar-month boundary and Analytics begins growing through the new cycle again.

\---

## Actual data coverage versus processing window

A processing window and the observed business-data period are not necessarily identical.

The first production processing window was:

```text
[2016-09-01, 2016-10-01)
```

Eligible Olist orders in that scope were observed only on:

```text
2016-09-04
→
2016-09-15
```

The Portal therefore distinguishes processing window from actual observed data coverage.

\---

## Complete 27-state analytical universe

The state-summary serving layer preserves a complete Brazil state universe.

Watermark-filtered orders are left-joined onto that state universe.

Validated first-window result:

```text
state_count       \= 27
total_orders      \= 4
zero_order_states \= 24
```

The three states with eligible orders were:

```text
RR
RS
SP
```

\---

## Zero evidence is not fabricated evidence

For a state with no eligible orders:

```text
order_count \= 0
gmv         \= 0
aov         \= 0
```

Metrics requiring actual observations preserve evidence absence:

```text
delivery_observation_count \= 0
late_delivery_rate         \= NULL
reviewed_order_count       \= 0
average_review_score       \= NULL
```

A missing observation is therefore not converted into a fake zero rate or fake zero review score.

\---

## Business Decision Model v1

The business decision model is deterministic.

Input fields include:

```text
stateCode
gmv
gmvGrowthRate
lateDeliveryRate
averageReviewScore
```

Peer-relative thresholds:

```text
GMV                 P75
GMV growth          P75
late-delivery rate  P75
review score        P25
```

Actions:

```text
RECOVER_SERVICE
PROTECT_VALUE
EXPAND
INVESTIGATE
MONITOR
```

Priority levels:

```text
P1
P2
P3
```

The current successful-watermark serving layer does not yet contain a governed previous-period growth metric, so `gmvGrowthRate \= null`.

The `EXPAND` path remains reserved until a governed comparison series is available.

\---

## Sparse early-cycle behavior

The first monthly production scope contains only four eligible orders.

That means peer-relative thresholds may be extreme or unstable during early-cycle snapshots. This is expected behavior for a deterministic peer-relative model operating on sparse evidence.

The decision model is not trained. Thresholds are recalculated deterministically from the currently visible successful-watermark state summaries.

\---

## Historical Review Diagnostic v2

The second analytics layer estimates negative-review risk after accounting for order and delivery mix.

Its analytical scope is intentionally different from the current business-action scope.

```text
Business Decision Model v1
→ current successful-watermark state summaries

Historical Review Diagnostic v2
→ persisted historical statistical model output
```

The statistical diagnostic is not refitted every hour from the tiny current production slice.

The UI explicitly labels the section:

```text
Historical review risk vs expected
```

and its sample size:

```text
Historical orders evaluated
```

This prevents users from interpreting a historical statistical estimate as if it came from the current sparse watermark scope.

\---

## Review Diagnostic v2 target

```text
negative_review \= average review score \<= 2
```

Baseline predictors:

```text
is_late_delivery
log1p(delivery_days)
log1p(order_gross_value)
multi_item_order
```

Persisted state output includes:

```text
evidence_count
actual_negative_review_rate
expected_negative_review_rate
residual_pp
ci_lower_pp
ci_upper_pp
z_score
diagnostic_state
model_version
generated_at
```

Current model version:

```text
business_decision_v2_logit_001
```

\---

## Fixed diagnostic rule

```text
evidence_count \< 100
→ INSUFFICIENT_EVIDENCE

residual_pp \>= 1
and ci_lower_pp \> 0
→ WORSE_THAN_EXPECTED

residual_pp \<= \-1
and ci_upper_pp \< 0
→ BETTER_THAN_EXPECTED

otherwise
→ AS_EXPECTED
```

\---

## Deterministic verification

Persisted statistical output is checked again by the Portal service before display.

Principle:

\> Persisted data is not automatically trusted.

Field-level checks include complete/unique state codes, valid probabilities, finite numeric values, valid confidence intervals, valid diagnostic states, model version, and generation timestamp.

Row-level verification recomputes the residual and diagnostic state.

Snapshot-level verification requires all 27 rows to share the same `model_version` and `generated_at`.

\---

## State-summary integrity verification

The state-summary service expects exactly 27 Brazilian state rows and rejects missing or duplicate states.

The earlier direct watermark filter temporarily caused sparse state rows to disappear. The correct serving design preserved the 27-state universe and left-joined eligible orders instead of weakening this integrity guard.

\---

## Security baseline

M10 Portal close-out includes:

\- `Content-Security-Policy`
\- `X-Content-Type-Options: nosniff`
\- `Referrer-Policy: strict-origin-when-cross-origin`
\- `Permissions-Policy`
\- `X-Frame-Options: DENY`

The CSP permits the CARTO basemap required by the analytics map.

The public Portal does not currently include application-level login. A shared organizational deployment should add an authentication boundary or equivalent platform control.

\---

## Validation at final M10 close-out

```text
Vitest:                    21 / 21 PASS
ESLint:                    PASS
Next.js production build:  PASS
```

\---

## Real successful-watermark validation

After the first monthly production run:

```text
successful processing window:
2016-09-01
→
2016-10-01
```

the national analytical serving result was:

```text
order_count \= 4
gmv         \= R$354.75
aov         \= R$88.69
```

Observed data period:

```text
2016-09-04
→
2016-09-15
```

State validation:

```text
27 states
4 total orders
24 zero-order states
```

Non-zero state rows:

```text
RR: 1 order
RS: 1 order
SP: 2 orders
```

State-level GMV summed exactly to the national GMV result.

\---

## Runtime checks

Final runtime checks confirmed:

\- `/analytics` follows the successful production watermark
\- active/failed windows do not leak into completed analytics
\- all 27 states remain present
\- national KPI totals match state totals
\- no-evidence states remain semantically distinguishable from measured zero values
\- historical review-risk diagnostics are clearly separated from current business actions
\- the analytics map renders under the production CSP
\- valid finding IDs open finding-detail pages
\- invalid finding IDs enter the controlled unavailable state
\- security headers are present in production responses

\---

## M11 handoff

M11 is reserved for controlled historical playback, backfill, and recovery.

M10 already provides normal forward monthly production cycles. That normal cycle mechanism should not be confused with arbitrary replay.

M11 planned boundaries include:

```text
one-window replay
multi-window backfill
resume after failure
replay idempotency
replay audit history
separate replay state
replay versus incremental consistency checks
```

Replay must remain isolated from the normal production watermark and must never silently move normal incremental state backward.
