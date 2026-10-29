# NYC v2 — Deferred Architecture Decisions

Date: 2026-10-27\
Status: **Deferred / review checkpoints only**\
Target branch: `feature/nyc-v2`

This document records three optional decisions for later evaluation, including a distinct schema/mapping-proposal experiment added 2026-10-29. It does **not** authorize implementation, expand current runtime permissions, or claim that any planned component has been deployed.

## Context and current priorities

NYC v2 targets an AWS data platform (S3, Airflow, Databricks/PySpark, Delta Bronze/Silver/Gold, quality gates, Redshift) with deterministic evidence collection, incident replay / optional isolated fault injection, and a **provider-neutral, initially local read-only Investigator** (optional Bedrock inference/AgentCore only after evaluation and spend approval). Prioritize finishing and verifying the data plane, Evidence Reader, replay, reliability tests, and a working, evaluated single-agent vertical slice. Distinguish planned from implemented capabilities.

## D-01 — Optional Phase 4: Supervisor / Multi-Agent / RAG

**Decision:** Defer Supervisor, specialized sub-agents, and RAG until the single-agent investigator, controlled tool calling/MCP, AgentCore-related operations, evaluation, and safety boundaries are demonstrably working.

**Revisit when:** A reproducible evaluation shows single-agent limitations, or a real need arises to retrieve runbooks, ADRs, or prior incident histories semantically.

**Acceptance criteria:** Document a concrete use case, baseline against the simpler approach, measurable diagnosis quality/latency, added operating cost, failure modes, auditability, and rollback/kill-switch behavior. Prefer a single agent and direct structured evidence access when adequate. Do not add RAG or multi-agent solely to showcase technologies.

## D-02 — Post-core identity and network security review

**Decision:** After core NYC v2 delivery, review enterprise identity and access requirements: Microsoft Entra ID (where Azure/Databricks integration warrants it), OAuth 2.0 / OpenID Connect, workload identities/service principals, and network security (VPC, private endpoints, TLS, isolation, least privilege).

**During current development:** Preserve clear trust boundaries, least-privilege AWS IAM, secrets handling, authenticated tool interfaces, authorization checks, and auditable agent actions. Do not postpone essential baseline security or prematurely introduce an Entra dependency into an AWS-first architecture.

**Revisit when:** The pipeline and initial agent are working and an actual cross-cloud integration, user login, machine-to-machine authorization, or network exposure requirement can be articulated.

**Acceptance criteria:** Threat model, principal/resource permission matrix, authentication vs authorization design, secret/credential lifecycle, network exposure diagram, cost/operational tradeoffs, and targeted tests. Implement only components justified by the architecture.

## D-03 — Optional post-MVP Agent-Assisted Schema/Mapping Evaluation

**Decision:** Defer the Schema/Mapping Proposal Agent (ADR AE-01, Track B) until the data plane, read-only evidence interfaces, replay and a working **evaluated single Investigator** meet their acceptance gates. It is not an authorization for autonomous SQL/Delta changes, schema migration or data repair.

**Revisit when:** Incident Investigator evaluation has working ground-truth separation, deterministic policy/citation gates and an audited baseline; a concrete schema-drift or semantic mapping problem justifies a separate experiment.

**Acceptance criteria:** Reviewed reference mapping pairs and semantic rules, an isolated fixture store, proposed changes only (no writes to protected raw/Bronze/Silver/Gold), independent correctness/type-contract checks, pair-level precision/recall/F1, uncertainty/abstention and unsafe-proposal rate, and reproducible comparison with a non-agent baseline. A human must review any change proposal. No claim of production self-healing.

**Learning-by-building:** Agent evaluation fundamentals, tool/trajectory tests, safety gates and short English explanation cards are **part of existing NYC v2 engineering milestones**, not a new Phase 4 or a standalone course. See `nyc_v2_agent_evaluation_learning_by_building_ADR.md`. Unit 1B contracts are now stable and pushed (`f13890e`); a small offline evaluator may be developed without delaying Unit 1C or advancing D-01 or D-02.

## Change control

**Checkpoint (2026-10-29):** Unit 1B has been implemented and pushed as `f13890e` (44 combined Unit 1A/1B regression tests); Unit 1C protected evidence capture and Replay remain the current next gate. No deferred feature is thereby promoted.

All three decisions are **optional, not scheduled commitments**. Reassess at the relevant milestone; record implement / postpone / reject with evidence. No change to existing development priority. The local, read-only **Agent Evaluation Harness** in ADR AE-01 is a quality gate for the accepted single-agent path, **not** early approval of any deferred D-01/D-02/D-03 expansion.
