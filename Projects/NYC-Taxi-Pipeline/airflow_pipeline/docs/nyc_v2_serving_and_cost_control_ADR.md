# NYC V2 — Published Serving and Unified Cost Control (ADR SRV-01)

**Date:** 2026-11-20
**Revision:** 2026-11-23 — scale-aware serving and cost-routing clarification
**Status:** Accepted design; R1/R2 verified, R3 configured locally but not deployed, later Serving/Routing phases not implemented
**Branch:** `feature/nyc-v2`
**Related:** [ADR CW-01 — Cost-Aware Workload Routing](nyc_v2_cost_aware_workload_routing_ADR.md)

## 1. Context and current implementation checkpoint

NYC V2 is an AWS-first, scale-oriented data platform. The three-month NYC TLC window is a **correctness and reproducibility baseline**, not the platform's intended data-volume ceiling.

**Confirmed project evidence (as of 2026-11-23):**

- The real Airflow Full Refresh succeeded for `[2023-01-01, 2023-04-01)`: versioned S3 Raw -> local Spark Delta Bronze -> Silver -> six Gold tables -> Quality Gate -> published Active Gold Manifest. The active release is `airflow-a521894129ca813bd8977d28`.
- The isolated Silver full-scale validation reported 9,384,487 Bronze rows -> 8,807,293 Silver rows, 577,194 filtered and zero invalid output rows. These are historical observed results, not guaranteed outcomes for future windows.
- **R1/R2 completed and pushed** at `56cf821` (2026-11-21): six verified Gold tables, 7 Parquet files (about 13.39 MiB), 6 Redshift COPY manifests, 1 release manifest, 14/14 S3 objects independently confirmed. The release is version-bound and immutable.
- Gold Delta remains on the local Docker Linux volume. It is **not** yet a cloud-shared Delta Lake accessible to future distributed-compute backends.
- `scripts/workload_router.py` (`cw-v1.0`) remains a **pure offline compute decision function**: `NO_COMPUTE`, `LOCAL_SPARK`, `BLOCKED_REVIEW`, always `auto_launch=False`. It is not wired to paid compute execution or Redshift Serving.
- R3 Terraform network/IAM/Redshift declarations have passed **local** `fmt`/`validate` checks; S3 Remote Backend initialization succeeded. These are preparation steps, **not** Redshift provisioning. No `terraform apply`, Redshift COPY, live SQL, or Serving publication has been performed.
- Redshift SQL Serving, published cached Dashboard, real Serving Router and Databricks execution remain planned. Do not describe them as operational.

**Scale intent:** first demonstrate the complete, safe Serving path on the already-certified baseline; then run evidence-based 12-month and (if justified) multi-year scale experiments without discarding the baseline or bypassing release validation.

## 2. Architecture decision

**One unified Cost-Aware Control Plane, with two separate decisions:**

1. **Compute Routing** determines whether a certified data product must be recomputed and, if so, with which approved backend.
2. **Serving Routing** determines how an already-certified result is consumed or refreshed: cached public data, Redshift SQL, or a controlled Serving refresh.

The key principles are:

- **Compute is ephemeral; published data products are durable; a public cached Dashboard remains accessible.**
- **A Dashboard visit must never automatically launch paid compute, Terraform, Airflow, or Databricks.**
- **Reuse verified published versions** rather than reprocessing Bronze/Silver/Gold unnecessarily.
- **Fail closed** on missing lineage, stale versions, unsupported scope, unknown spend, unavailable backend, insufficient permissions, or missing approval.
- **Publish only after validation**. A failed export or Redshift COPY must not displace the previous serving revision.
- Keep infrastructure declaration, workflow execution, query consumption and policy approval as different responsibilities.

## 3. Logical architecture

```text
NYC TLC -> S3 Raw (versioned)
              |
           Airflow
              |
        Spark + Delta Lake
         Bronze / Silver
              |
      Six Gold candidates
              |
     Quality Gate + Active Manifest
              |
        pinned Delta versionAsOf
              |
        validated Parquet snapshot
              |
         S3 Serving release
              |
       +------+------------------------+
       |                               |
       v                               v
small, sanitized chart aggregates   Redshift Serverless COPY
       |                            into release-scoped tables
       v                               |
versioned static JSON/CSV            v
       |                        SQL/count/type verification
       v                               |
Cloudflare Pages public Dashboard     v
       |                        approved Serving revision
       +-------- optional live backend --+
                 (authenticated, gated)

Unified Router policy:
  Compute: NO_COMPUTE / LOCAL_SPARK / future DATABRICKS_JOB / BLOCKED_REVIEW
  Serving: CACHE_READ / REDSHIFT_QUERY / SERVING_REFRESH / BLOCKED_REVIEW
```

The diagram describes the **target**. The S3 Serving, Redshift and public cached branches are planned work, not claims of deployed infrastructure.

### Scale-aware capacity target (clarification, not an executed benchmark)

The end-state is a platform capable of **multi-year batch ingestion, reliable transformation and cost-governed SQL Serving**. Scale the data and infrastructure based on measurements, not on a calendar or a fixed byte-count threshold.

| Validation stage | Scope | Evidence to capture | Deployment status |
|---|---|---|---|
| **Baseline — complete** | Three months: 2023-01 to 2023-03 | Correctness, Airflow full refresh, pinned versions, Gold quality, immutable S3 export | R1/R2 verified |
| **Scale 1 — candidate** | Twelve months of NYC TLC history | End-to-end runtime, raw and Gold sizes, rows, shuffle, retries/backfills, cost, capacity headroom | Not executed/approved |
| **Scale 2 — conditional** | Three to five years, only if Scale 1 justifies it | Sustained capacity, scheduling, recovery, workload routing, price/performance | Not executed/approved |
| **Serving benchmark — conditional** | Same certified Gold release, representative queries | Latency, concurrency, SQL flexibility, bytes transferred, total cost across cache/local SQL/Redshift | Not executed/approved |

**Processing scale is not Serving scale.** Millions of raw trips can collapse into comparatively small Gold aggregates. A large Bronze input alone does not justify Redshift: decide from the **actual Gold size, query shape, concurrency, SLO and cost**. Benchmark candidate engines using the same release and equivalent SQL semantics. Athena/DuckDB are comparison candidates, not implemented NYC V2 routes or approved deployments.

Preserve the current 3-month release as a reproducible regression baseline. Do not trigger another full refresh solely to exercise a new Serving backend.

## 4. Storage and public availability

| Data product | Intended location | Lifecycle and purpose |
|---|---|---|
| Source files, versions and checksums | S3 Raw | Durable source and reproducible ingestion |
| Current Bronze/Silver/Gold Delta | Docker Linux volume | Local MVP computation; must not be called cloud-durable |
| Gold Active Manifest + pinned version references | Local now; preserve a certified copy with Serving releases | Lineage, immutable references, audit and retry |
| Full certified Serving Parquet | Versioned, immutable S3 prefix `serving/releases/<release_id>/<table>/` | Redshift staging; separate from Raw and Delta logs |
| Small public chart-specific aggregates | Versioned JSON/CSV assets with a static website | 24/7 portfolio visualization without a running warehouse |
| Larger static files (only if justified) | Optional R2 / object storage | Introduce after measuring file sizes and request volume |
| Warehouse tables | Redshift Serverless, release-specific staging and published SQL views | Optional live SQL serving |

A public website must not simply expose complete Gold or raw trip-level records. Export only aggregated, privacy-reviewed metrics necessary for the UI. The cached dashboard may continue to work after the warehouse has been destroyed; **arbitrary live SQL cannot**.

**Recommended public hosting:** Cloudflare Pages, contingent on measuring actual static asset size and functionality. Cloudflare Pages cannot directly run the existing Python Streamlit application. Keep/adapt Streamlit for development and Redshift SQL verification; defer a static frontend rewrite until metrics and warehouse queries are verified. Render or Streamlit Community Cloud remain optional for Python demonstrations, with their own availability constraints.

The public cache and the complete warehouse data are **two separate published artifacts**, both linked to the same certified Gold release. A `generated_at` timestamp must not be confused with business-data freshness (the sample window is historical).

## 5. Unified routing semantics

The existing `cw-v1.0` remains the verified baseline. Future extensions must build on it rather than introducing a competing router.

| Situation | Compute decision | Serving decision | Execution constraints |
|---|---|---|---|
| Certified Gold and published cache meet request | `NO_COMPUTE` | `CACHE_READ` | No Spark or paid SQL |
| Certified Gold; explicit live SQL needed, permitted and available | `NO_COMPUTE` | `REDSHIFT_QUERY` | Secure backend, timeouts, quota and approval |
| Gold certified, but Serving revision behind | `NO_COMPUTE` | `SERVING_REFRESH` | Approved S3/COPY/validation workflow |
| Supported full refresh, local host qualified | `LOCAL_SPARK` | Serving refresh only after Gold certification | Source, capacity and spend preflight |
| Proven heavy job beyond local capacity | Future `DATABRICKS_JOB` | Serving refresh after certification | Shared durable storage, parity, benchmark and explicit approval |
| Any required safety or cost evidence missing | `BLOCKED_REVIEW` | `BLOCKED_REVIEW` when applicable | No compute started |

**`NO_COMPUTE` means no new Gold calculation; it does not mean that every SQL query is free.** `CACHE_READ` and `REDSHIFT_QUERY` belong to the separate Serving decision.

A future route decision records `policy_version`, product/window, input versions, Gold/Serving release IDs, freshness evidence, estimated cost, permission and approval state, reason codes and actual outcome. `DRY_RUN` never launches infrastructure. The current `publication_verified`, `freshness_verified` and related flags are **caller-provided assertions**, not automatic proof until real evidence verifiers exist.

### Serving-route eligibility and cost evidence (target contract; not implemented in cw-v1.0)

The router should separate *need to recompute Gold* from *need to serve an already certified result*. The existence of a Redshift Workgroup must never imply that queries should run on it.

| Request / evidence | Proposed Serving decision | Rationale |
|---|---|---|
| Published, revision-matched chart result already available | `CACHE_READ` | No live warehouse compute or new Gold job |
| Development or bounded local query experiment | Local SQL evaluation (candidate, not a formal V1 route) | Compare whether paid SQL adds measurable value |
| Dynamic SQL, validated release, approved spend, warehouse available | `REDSHIFT_QUERY` | Use authorized, bounded interactive analytics when warranted |
| New certified Gold release needs warehouse publication | `SERVING_REFRESH` with approval | COPY to separate release tables, validate before activation |
| Missing quality/lineage/freshness/cost evidence, unsupported scope or failed approval | `BLOCKED_REVIEW` | Fail closed; never auto-provision paid resources |

Record input release, query template and parameters, estimated/actual RPU-hours, cost assumptions, latency, concurrency, approval state and decision reason. Require representative measurements before declaring Redshift cheaper/faster than alternatives. A Dashboard page load must never auto-run Terraform, Spark, COPY or paid live SQL.

**Stable fallback is not automatic rollback:** retaining the current `serving_release_id` Terraform default provides a known-good configuration input; it does not restore Redshift tables or change the active Serving pointer. Rollback requires existing validated release tables and tested publication/activation logic.

## 6. Serving publication contract

### R1 — Version-pinned Gold to local Parquet

1. Read the published Active Gold Manifest; validate six expected table names, logical window, release ID and pinned versions.
2. Read each table via Delta `versionAsOf`, never silently switch to the latest version.
3. Validate required columns, business keys, data types, row counts and a few representative business-time values. Spark `TIMESTAMP_NTZ` must preserve NYC wall-clock semantics into Redshift `TIMESTAMP`; verify actual value parity, not just schema names.
4. Export standard Parquet to a **new immutable release directory**, matching field order to generated Redshift DDL. The Parquet `COPY` mapping is order-sensitive.
5. Store schema, column order, Delta versions, file integrity and row counts. Incomplete exports are **not published**.

### R2–R4 — S3 staging and Redshift

1. Upload Parquet data files (not Delta `_delta_log`) into release-scoped S3 prefixes; preserve file checksums and provenance.
2. Use reviewed Terraform to provision a **temporary/approved Redshift Serverless** environment with least-privilege IAM, network policy, resource and usage limits, and a teardown plan.
3. `COPY` into new, release-scoped Redshift staging tables; do not overwrite active Serving tables while loading.
4. Validate all six: row counts, column order/types, timestamp value parity, key metrics and representative SQL aggregations. Confirm release identity and source versions.
5. Activate a serving revision **only after every table passes**. Keep the prior revision on failure. Do not claim cross-table atomic cutover until a concrete, tested Redshift publication mechanism proves it.
6. Record validation evidence, published revision, failure reason and rollback/retry procedure.

### R5 — Public cache and Dashboard

- Export only the limited aggregated measures needed for the public charts, keyed by Gold release ID.
- Deploy immutable static assets and switch a small active pointer only after complete verification. Prevent mixed-release charts through revision-aware URLs.
- Show dataset business window, published release ID, and whether the view is `Cached` or `Live Redshift`.
- A cache hit must not launch Spark, Redshift or Databricks. When the warehouse is absent, cached mode remains usable; live SQL mode states that it is unavailable.
- Do not represent historical batch data as live or real time.

## 7. Infrastructure, runtime and spend control

| System | Owns | Does not own |
|---|---|---|
| Terraform | Redshift workgroup/namespace, IAM, networking and approved usage guardrails; lifecycle changes | Re-applying infrastructure for each query or refresh |
| Airflow / approved executors | Jobs, snapshot staging, COPY, validation, publication, retries and evidence | Launching paid jobs because someone opened a dashboard |
| Unified Router | Deterministic decisions, policy and approval preflight | Autonomous infrastructure permissions |
| Redshift Serverless | Optional, on-demand warehouse SQL computation | Guaranteed free storage or unlimited free queries |
| Databricks Jobs Compute (future) | Ephemeral approved heavy ETL jobs with termination | Replacing the required warehouse-serving SQL proof |
| Cloudflare Pages/static files | Low-cost read-only cached visualization | Arbitrary SQL without a backend |

**Serverless, not Provisioned pause/resume, is the default Redshift evaluation choice.** Provisioned pause/resume can be valid for predictable workloads, but Serverless scales compute according to workload without requiring explicit pause/resume. Neither avoids ongoing applicable storage and related charges.

No fixed price or free-tier claim authorizes deployment. Before each paid AWS experiment, verify current Frankfurt (`eu-central-1`) prices, quotas and estimated duration, and require explicit user approval. Apply Redshift usage limits, budget alerts, IAM boundaries, query timeouts, concurrency caps and no continuous polling/heartbeat queries. Limits and budget alerts are layered protections, **not a guarantee of zero overspend**.

Terraform teardown must consider workgroup, namespace, snapshots, related networking and retained S3 objects; check the real billing state afterward. Destroying Redshift disables live SQL while the independently hosted public cache stays available. Do not issue `terraform apply/destroy` for every data request.

### On-demand Redshift lifecycle (cost policy, not yet automated)

Treat Redshift Serverless as a **real but optional Serving backend**. Infrastructure is declared through reviewed Terraform; requests are routed separately. Do not run `terraform apply`/`destroy` for each query or simply because a Dashboard is opened. An idle Workgroup normally does not bill active compute, but managed storage, secrets and other retained resources can still cost money; open transactions or repeated health checks can trigger compute billing.

The initial R3 Terraform defaults currently target 4 base RPUs, 4 maximum RPUs, a daily 2 RPU-hour compute usage limit and `deactivate` on breach. **These are local declarations, not active AWS guardrails or a fixed monetary ceiling**. A first deployment requires separate account-specific Frankfurt pricing confirmation, explicit spend approval, reviewed `plan`, verified IAM/network, and an emergency teardown/checklist. Keep AWS Budgets and usage alerts as defense-in-depth, not a hard stop.

Retain or destroy the Workgroup only after considering projected real usage, the cost of keeping validated warehouse tables, reinstatement time, and alternatives. Deleting the Namespace is a data-lifecycle decision with rollback consequences; it must never be silently triggered by Cost Routing.

## 8. Delivery order and acceptance evidence

| Phase | Goal | Acceptance |
|---|---|---|
| **R1** | Certified version-pinned Gold -> local standard Parquet + Redshift DDL | Schema/order/type/row count tests and representative timestamp value checks |
| **R2** | Immutable S3 Serving staging | Integrity and release references, idempotent uploads, no `_delta_log` |
| **R3** | Temporary Terraform-managed Redshift Serverless | Reviewed plan, IAM/network/cost controls, documented teardown |
| **R4** | Six-table COPY, SQL verification, approved publication | Failure/rollback tests, source release parity, query evidence |
| **R5** | Working SQL prototype + small public cached Dashboard | Static mode survives Redshift shutdown; live SQL works only when enabled |
| **R6** | Extend Router to Serving decisions | Tested evidence, approvals and no dashboard-triggered paid jobs |
| **Scale validation (after R4 baseline)** | Expand to 12 months, then conditionally multi-year history | Correctness parity, elapsed time, recovery, query requirements and measured spend; no speculative scaling |
| **Serving benchmark** | Compare cache/local query candidates with approved Redshift SQL | Equivalent-release query results, latency/concurrency and true cost evidence |
| **Optional later** | Databricks heavy-compute routing | Shared durable lake storage, equivalent results, measured need and approved spend |

No cloud provisioning is approved by this ADR. Keep existing YAML files until Redshift end-to-end verification, then perform repository cleanup with path/reference review.

## 9. Deferred and rejected scope

- No automatic warehouse start on a public dashboard visit, no uncontrolled public SQL endpoint.
- No speculative data-size routing threshold or claim that Databricks is cheaper without benchmark evidence.
- No immediate migration of local Delta volumes to shared cloud storage.
- No early React/other frontend rewrite before serving metrics are clear.
- No extra orchestration microservice, multi-agent supervisor or RAG as a prerequisite.
- No pretending a static cache is current for arbitrary ad hoc queries.
- No claim that multi-year scale or Redshift price/performance has been validated before real benchmarks.
- No production Redshift SQL use based solely on raw row count; Serving decisions require query and cost evidence.

## References

- [Original Cost-Aware ADR (CW-01)](nyc_v2_cost_aware_workload_routing_ADR.md)
- [Gold Quality Contract](gold_quality_contract.md)
- [AWS Redshift COPY for columnar formats](https://docs.aws.amazon.com/redshift/latest/dg/copy-usage_notes-copy-from-columnar.html)
- [AWS Redshift Serverless billing](https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-billing.html)
- [AWS Redshift Serverless usage limits](https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-workgroup-max-rpu.html)
- [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/)
