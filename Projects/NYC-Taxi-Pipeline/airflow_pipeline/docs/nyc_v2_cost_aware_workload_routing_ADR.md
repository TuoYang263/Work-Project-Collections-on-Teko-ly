# NYC V2 — Cost-Aware Workload Routing (ADR CW-01)

**Date:** 2026-11-04  
**Status:** Architecture direction accepted for planning; **NOT IMPLEMENTED**  
**Target branch:** `feature/nyc-v2`  
**Scope:** AWS-first NYC Yellow Taxi V2 data plane, serving freshness, compute selection, and cost controls.

## 1. Context and problem

NYC V2 is intended to serve data products, not to run the largest available Spark engine for every refresh. Dashboard consumers need specific metrics at specific freshness levels. A small change to one certified serving aggregate should not automatically require a full Bronze/Silver/Gold recomputation or an on-demand Databricks cluster.

Current verified and implemented boundaries at this ADR's creation:

- S3 Raw ingestion uses versioned raw references and checksums. The isolated S3-to-Bronze smoke run processed 2,964,624 real January 2024 rows.
- The active Medallion runtime uses **local filesystem Delta paths** for Bronze, Silver, and six Gold tables; this is not yet a shared-storage, multi-node Databricks implementation.
- `CONFIGURED_FULL_REFRESH` overwrites Bronze, Silver, and Gold within the configured scope. Controlled `BACKFILL` is **GOLD_ONLY** with `WINDOW_REPLACE`/Delta `replaceWhere`.
- Six Gold tables are written separately and validated **after** writes. Quality Gate failure blocks downstream publication but does not atomically roll back all six Gold tables.
- The AWS V2 Medallion DAG currently ends at the Gold Quality Gate; the previous BigQuery export was disconnected. **Redshift publication is not implemented.**
- Unit 1C protected incident snapshots and offline replay remain active development priorities.

This document makes **no claim** that incremental Bronze/Silver writes, workload routing, Databricks execution, or a dashboard-driven refresh API currently exist.

## 2. Decision

Adopt **demand-first, cost-aware routing** as the target architecture:

1. **The data product defines the requirement.** Record requested Gold/serving datasets, metric definitions, logical window, maximum acceptable staleness, and completion deadline (SLO) independently of the compute backend.
2. **Determine whether recomputation is needed.** First check whether existing validated/published data meets the requirement. Dashboard viewing alone must **never** launch compute.
3. **Calculate affected scope.** Identify source versions, changed partitions, dependencies, and whether the requested result can be produced with currently supported write scopes. Do not infer that a one-day dashboard view implies only one day of source changes.
4. **Choose the least costly qualified execution path.** Compare expected total cost, elapsed time, reliability, correctness, resource limits, and cold-start overhead. Do not pick a backend based solely on input bytes.
5. **Fail closed on unsupported or unbounded changes.** No automatic expensive job when cost, permissions, scope, source version, or safe publication cannot be established. Defer or require human approval.
6. **Publish only after verification.** Preserve Delta transaction semantics, evidence, lineage and Quality Gate checks; do not expose partial Gold updates as a certified serving revision.

Airflow remains a suitable **orchestration/control plane**; an execution backend is selected independently. Terraform declares durable infrastructure and job definitions, while Databricks Jobs Compute (if later justified) creates and releases compute for an actual run. Avoid `terraform apply`/`destroy` on every data refresh.

## 3. Candidate execution routes (future; not wired into runtime)

| Route | Intended use | Important constraint |
|---|---|---|
| `NO_COMPUTE` | Certified serving data already fresh enough; reuse/cache/query | Must be a previously validated serving revision |
| `SERVING_REFRESH` | Refresh an eligible SQL materialization/serving projection using existing certified Gold data | Only where target engine supports equivalent query and safe publish semantics |
| `LIGHTWEIGHT` | Bounded Python/SQL transformation or simple partition recompute | No direct edit of underlying Delta Parquet files; correctness parity required |
| `LOCAL_SPARK` | Moderate batch transforms for which one bounded container/host is sufficient | Enforce memory, runtime and concurrency limits |
| `DATABRICKS_JOB` | Large historical reprocessing, complex joins/shuffles or SLOs proven to require distributed execution | Optional, explicit cost/permission approval; on-demand job compute and automatic release |
| `BLOCKED_REVIEW` | Insufficient estimate, unsupported write scope, unclear lineage, unsafe publication or budget overrun | No compute launched |

**No fixed GB, row-count, or partition-count cutoffs are adopted in this ADR.** A single large scan can be cheap; a smaller but skewed join can be expensive. Thresholds must be calibrated with measured benchmarks.

## 4. Refresh and dependency semantics

A planner must separate **what must be refreshed** from **how it is computed**.

- **Serving-only:** A different dashboard query, visualization, or a supported serving materialization refresh may use already certified Gold without touching Bronze/Silver/Gold.
- **Gold-window rebuild:** Use the *existing controlled* Gold-only Backfill only where its logical window and source-state assumptions are satisfied. The current code reads the latest Silver table; a version-pinning requirement remains open for reliable replay/retry.
- **Full Medallion refresh:** Use the current configured full-refresh contract for the intended source scope. Do not assert that selective Bronze/Silver updates are already supported.
- **Late-arriving trip behavior:** Pickup and dropoff business timestamps drive different Gold summaries. Include upstream month boundaries and both event-time dependencies when deciding affected dates. `INCLUDE_PREVIOUS_MONTH` in the current full-refresh contract is not a substitute for a general dependency planner.
- **Six Gold tables:** A request for one serving product must not silently assume independent production is safe when shared inputs or the Quality Gate require a consistent set. Cross-table publication is an open design gate.

Data-product freshness policies should be defined for each of the six current Gold tables using `product_id`, business-time grain, source dependencies, validated revision, freshness target, and allowed execution routes. These are **planned metadata**, not additional live services.

## 5. Routing inputs, decision record and evidence

A future dry-run planner should accept the following **logical inputs** (not a new runtime API yet):

- `requested_products`, `logical_window`, `freshness_slo`, `deadline`
- `available_certified_revision`, `source_object_versions` / `silver_delta_version` (when available)
- `changed_partitions`, estimated input bytes/files/rows, expected shuffle/joins, small-file or skew indicators
- approved compute backends, concurrent jobs, historical runtime/cost observations, estimated total cost and uncertainty
- per-run spend ceiling, project/month quota, explicit approval requirements and actor identity

The planned decision record should capture `plan_id`, linked `dag_id/run_id`, `policy_version`, `selected_route`, `reason_codes`, `estimate_source`, `estimated_cost_range`, `estimated_runtime_range`, `approval_status`, `source_versions`, `write_scope`, `quality_gate_result`, `published_revision`, and actual cost/runtime when known.

**Source measurements and historical usage are evidence; an estimated cost is not a guarantee.** Never label an unmeasured route as "cheapest" or an unverified plan as "safe".

## 6. Cost and safety guardrails

- **Default:** Reuse certified data when its freshness SLO is met. A dashboard load never starts a costly refresh directly.
- **Preflight:** Check supported contract/write scope, source version, resource quota, budget estimate and approval *before* a compute request.
- **Heavy compute:** Cap maximum worker/instance size, job timeout, concurrency and retry count; abort before launch if no acceptable estimate or approval exists. Failed and canceled jobs must release compute.
- **Accounting:** Measure actual startup time, compute time, data scanned, bytes shuffled, cloud/network/storage overhead and billed costs where available. Compare estimates with outcomes; update route policy only after sufficient evidence.
- **Budget limits:** Cloud budget alerts and Databricks budget notifications are monitoring mechanisms, **not hard guarantees of zero overspend**. Enforce project-side preflight/approval and AWS permissions in addition to alerting.
- **Correctness:** Preserve Delta ACID operations and supported `replaceWhere` semantics; never directly mutate Delta data files. Validate schema, keys, business windows, counts, provenance and idempotence.
- **Publication:** Quality Gate success is necessary, not sufficient, to claim an atomic six-table serving revision; design a protected publication boundary before an automatic serving switch.
- **Security:** Least-privilege role separation. No model, dashboard, or read-only Incident Investigator can launch Databricks, change routing policy, modify protected snapshots, or bypass cost approvals.

## 7. Terraform and Databricks ownership

**Terraform (durable definitions):** permitted AWS storage/access boundaries, Databricks workspace/job definitions if adopted, policy controls, and approved network/IAM resources. Changes are reviewed separately from data refreshes.

**Airflow (run orchestration):** evaluate a reviewed routing decision, submit the selected job, pass the execution contract and immutable input references, monitor/cancel, persist evidence and evaluate post-run gates.

**Databricks Jobs Compute (ephemeral execution, optional):** start for the selected heavy workload, process the approved scope, terminate after use. The pipeline's durable Delta data lives on authorized shared object storage, not worker-local paths. A job cluster is not automatically the least expensive solution: compare against local Spark or serverless alternatives once operational costs are known.

**Redshift (serving target, not implemented):** choose a serving pattern and publication contract based on actual dashboard query needs. Avoid keeping expensive warehouse resources running merely to demonstrate a technology.

No new AWS or Databricks resource deployment is authorized by this ADR.

## 8. Evaluation plan and acceptance gates

Implement this direction only after the existing **Unit 1C** and core data-plane verification checkpoints permit it:

1. Complete protected Unit 1C snapshots and first offline replay; keep AI Investigator read-only.
2. Run an isolated Bronze -> Silver -> six Gold -> Quality Gate integration with bounded local data; collect runtime and resource metrics.
3. Identify 2-3 concrete dashboard requirements; document each Gold product's update cadence, freshness, grain and dependencies.
4. Benchmark `NO_COMPUTE`/`SERVING_REFRESH`, lightweight SQL/Python (when equivalent), and local Spark on representative small and moderate scenarios. Prove result parity and correct partial refresh scope.
5. Implement a **pure offline `DRY_RUN` decision function** with policy versioning and tests, without cloud credentials or job-launch permissions. Test unknown cost, over-budget, unsupported scope, stale source, manual approval, same-input rerun and no-compute decisions.
6. **Optional later:** With explicit spend approval, run one isolated, capped Databricks job on disposable data; compare end-to-end cost (including startup) and output parity to local Spark. Promote heavy routing only if measurements justify it.
7. Integrate runtime routing only after source-version pinning, safe serving publication, cost controls and audit evidence pass targeted fault tests.

Suggested acceptance scenarios:

| Scenario | Expected policy outcome |
|---|---|
| Dashboard reads fresh, certified Gold | `NO_COMPUTE` |
| A serving projection can refresh from certified Gold | `SERVING_REFRESH` if verified |
| Small bounded update with proven equivalent SQL/Python | `LIGHTWEIGHT` only after support is implemented and tested |
| Moderate recompute within local capacity | `LOCAL_SPARK` |
| Distributed recompute fits measured benefit and approved spend | `DATABRICKS_JOB` only after pilot approval |
| Cost unknown, budget exceeded or unsupported window semantics | `BLOCKED_REVIEW` |

## 9. Non-goals and explicit deferrals

- No immediate migration of existing NYC V2 Spark code to Databricks.
- No new dashboard-triggered DAG, routing microservice, ML-based optimizer, multi-agent supervisor or automatic resource procurement.
- No speculative hard-coded data size thresholds.
- No incremental Bronze/Silver writes without a dedicated contract and transaction tests.
- No claim of an automatically refreshed Redshift dashboard until Redshift serving is implemented and verified.
- No new priority ahead of Unit 1C or existing pipeline completion.

## 10. Decision review

**Revisit when:** Local end-to-end measurements and concrete serving SLOs are available, or when an approved heavy workload demonstrably exceeds local constraints.

**Decision metric:** lowest **observed total cost** among routes that satisfy correctness, freshness, reliability and security requirements—not minimum compute cost in isolation.

**Status at writing:** Design documented, routing **not implemented**, no new paid resources provisioned.

### References

- Databricks Jobs compute types and per-job lifecycle: https://docs.databricks.com/aws/en/jobs/compute
- Databricks budgets and caveats: https://docs.databricks.com/aws/en/admin/account-settings/budgets
- Delta Lake transactional updates and merges: https://docs.delta.io/delta-update/
- Existing NYC V2 contracts: `gold_quality_contract.md`, `nyc_v2_deferred_architecture_decisions_2026-10-27.md`, `scripts/execution_contract.py`.
