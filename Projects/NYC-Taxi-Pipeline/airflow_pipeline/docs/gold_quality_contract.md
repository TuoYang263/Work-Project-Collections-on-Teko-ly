
# NYC V2 — Gold Quality Gate Contract

## Scope
Validate six Gold Delta tables after computation.

Applies to:
- CONFIGURED_FULL_REFRESH
- BACKFILL / GOLD_ONLY

## Common blocking rules
- Required columns must exist.
- Business key columns must not contain NULL.
- Business keys must be unique within the logical window.
- Business dates must be within [start, end).
- Trip and flow counts must not be negative.
- A failed check must block downstream serving publication.
- A failed check must fail the pipeline run.

## Table-specific rules

| Gold table | Business key | Additional rule |
|---|---|---|
| trip_summary_hourly | pickup_hour | trip_count > 0 |
| zone_summary_pickup_daily | pickup_day, zone_id | trip_count > 0 |
| zone_summary_dropoff_daily | dropoff_day, zone_id | trip_count > 0 |
| flow_imbalance_daily | service_day, zone_id | total_flow > 0; imbalance between -1 and 1 |
| airport_economics_daily | service_day, airport_code, direction | trip_count > 0; direction is ARRIVAL or DEPARTURE |
| od_corridor_daily | service_day, pickup_zone_id, dropoff_zone_id | trip_count > 0 |

## Failure behavior
- FAIL: raise an explicit quality error.
- FAIL: record the failing table and rule.
- FAIL: do not execute serving export.
- PASS: permit the next pipeline task.

## Current limitation
Gold Delta tables are written before validation.
This gate protects downstream publication, not atomic
promotion of all six Gold tables.

## Deferred
- Statistical anomaly detection
- Historical baseline comparisons
- Complex staging and promotion
- Automatic data repair
