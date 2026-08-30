# M10 Portal and Analytics

## Status

```text
Completed: 2026-08-30
```

This document records the final M10 portal and analytics scope.

M10 had two main parts:

- window and watermark control
- operational and business-facing portal work

The window-control design is documented separately in [`m10_window_control.md`](m10_window_control.md).

---

## Portal scope

The portal uses Next.js, React, and TypeScript.

Main routes:

```text
/overview
/analytics
/reliability
/findings/[findingId]
```

The layout uses one shared navigation shell. The pages are server-rendered where possible and use client-side interaction only where it is useful, such as map selection.

---

## Server-side data path

The portal uses a simple server-side path:

```text
Next.js Server Component
        ↓
     service
        ↓
   repository
        ↓
     BigQuery
```

The repository reads data. The service validates and maps it. The page renders the validated result.

Two early API routes for overview and reliability were removed because no client code used them. An internal HTTP layer is not added unless there is a clear need for it.

---

## Overview

`/overview` shows the current operational state of the window controller.

The page is intended to answer basic operational questions quickly:

- what state is the pipeline in
- what window is active
- what was the last successful window
- whether there is a current failure or retry state

The page reads the governed control state from BigQuery.

---

## Reliability

`/reliability` shows the latest persisted deterministic review.

Current validated review summary:

```text
179 evaluations
166 PASS
1 TRIGGERED
12 NOT_EVALUATED
```

Triggered findings link to `/findings/[findingId]`.

The finding page shows persisted evidence and context for one deterministic finding. The current real triggered example is M9-R006 for `fct_order_payments`.

---

## Finding identifier boundary

Finding IDs are carried in the URL path.

The page decodes the path segment once. The service then validates the decoded identifier before it can reach the repository.

The service allows only the character set used by the persisted IDs and limits the identifier length to 512 characters.

BigQuery access remains parameterized.

This is an input boundary rather than a replacement for query parameterization.

---

## Analytics

`/analytics` is a state-level business decision view for Brazil.

The page combines:

- order count
- GMV
- average order value
- delivery observation count
- late-delivery rate
- reviewed-order count
- average review score
- deterministic business action
- statistical negative-review diagnostic

The map covers all 27 Brazilian states. Selecting a state updates the KPI, action, and diagnostic cards.

The map uses MapLibre / react-map-gl with deck.gl and a CARTO basemap.

---

## Business Decision Model v1

The first decision model is deterministic.

Input fields:

```text
stateCode
gmv
gmvGrowthRate
lateDeliveryRate
averageReviewScore
```

Peer-relative thresholds use:

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

The current full-history snapshot has no governed previous-window growth measure, so `gmvGrowthRate` is `null`. The `EXPAND` path remains reserved until M11 provides monthly playback and previous-window comparison.

Current action mix:

```text
Recover Service  1
Protect Value    6
Investigate      7
Monitor         13
```

---

## Review Diagnostic v2

The second analytics layer estimates negative-review risk after accounting for order and delivery mix.

The binary target is:

```text
negative_review = average review score <= 2
```

The baseline model uses:

```text
is_late_delivery
log1p(delivery_days)
log1p(order_gross_value)
multi_item_order
```

The persisted state diagnostic includes:

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

The fixed diagnostic rule is:

```text
evidence_count < 100
→ INSUFFICIENT_EVIDENCE

residual_pp >= 1
and ci_lower_pp > 0
→ WORSE_THAN_EXPECTED

residual_pp <= -1
and ci_upper_pp < 0
→ BETTER_THAN_EXPECTED

otherwise
→ AS_EXPECTED
```

The current model version is:

```text
business_decision_v2_logit_001
```

---

## Deterministic verification

Persisted statistical results are checked again in the portal service before they are shown.

The idea is simple:

> Persisted data is not automatically trusted.

The service checks three levels.

### Field level

- exactly 27 state rows
- valid and unique state codes
- non-negative evidence counts
- probabilities between 0 and 1
- finite numeric values
- valid confidence intervals
- valid diagnostic-state values
- non-empty model version
- valid generation timestamp

### Row level

The stored residual must match:

```text
(actual_negative_review_rate - expected_negative_review_rate) × 100
```

A small tolerance is allowed for numeric representation.

The stored `diagnostic_state` is also recalculated from evidence count, residual, and confidence interval. A mismatch is rejected.

### Snapshot level

All 27 rows must use the same:

```text
model_version
generated_at
```

A mixed snapshot is rejected.

This keeps the service as the trust boundary between persisted data and the UI.

---

## Security baseline

M10 close-out added browser security headers:

- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy`
- `X-Frame-Options: DENY`

The CSP allows the external CARTO basemap used by the analytics map. Development keeps the setting required by the local Next.js toolchain; the production policy is stricter.

Unused API routes were removed to keep the exposed surface small.

The portal does not currently include application-level login. A public or shared production deployment should add organization authentication or an equivalent platform control. The portal runtime should use a least-privilege BigQuery service account.

---

## Validation at M10 close-out

Portal checks:

```text
Vitest:                 21 / 21 PASS
ESLint:                 PASS
Next.js production build: PASS
npm audit:              0 vulnerabilities
```

Runtime checks also confirmed:

- `/analytics` loads with the real 27-state diagnostic snapshot
- the map still loads under the CSP
- a valid finding ID opens the finding detail page
- an invalid finding ID returns the controlled unavailable state
- security headers are present in the production response
- the removed API routes are no longer part of the build route table

---

## M11 handoff

M11 will focus on historical playback and recovery.

The default playback window will be one month.

The planned boundary is:

- monthly playback
- multi-window backfill
- retry and resume
- replay state kept separate from the normal forward watermark
- replay versus incremental consistency checks

The first goal is to produce a reliable monthly history. Trend analysis is not part of the initial M11 scope. It will only be considered after the monthly series exists and the data shows that the analysis would be useful.
