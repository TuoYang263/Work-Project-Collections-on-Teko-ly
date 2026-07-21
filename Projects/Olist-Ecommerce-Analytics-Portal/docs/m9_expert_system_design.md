# M9 — Evidence-Grounded Pipeline Quality Expert System

## 1. Document Status

- Project: Olist E-Commerce Analytics & Pipeline Monitoring Portal
- Branch: `feature/olist-analytics-portal`
- Current milestone: M9.1 — Expert system design
- M8 completion commit: `92298ab docs: complete M8 monitoring documentation`
- Design status: Initial MVP design
- Implementation status: Not started

This document defines the initial architecture, evidence boundaries, review dimensions, and rule readiness for M9.

M9 must remain a lightweight, explainable, and evidence-driven extension of the existing M8 monitoring layer.

---

## 2. M9 Positioning

M9 is an:

> Evidence-Grounded Pipeline Quality and Production Readiness Expert System

It may also be described as an:

> AI-Assisted dbt Pipeline Quality Expert System

M9 is not a general-purpose chatbot.

It is a controlled review system that:

1. reads pipeline evidence persisted by M8;
2. evaluates deterministic engineering rules;
3. produces structured findings and missing-evidence records;
4. traces downstream impact through dbt lineage;
5. optionally uses an LLM to explain verified findings.

The deterministic rule engine is the source of truth.

The LLM is an explanation layer only.

---

## 3. Core Design Principles

M9 follows these principles:

- Deterministic rules run before any LLM review.
- Findings must be supported by persisted evidence.
- Missing evidence must be reported explicitly.
- Facts, inferences, risks, and recommendations must remain separate.
- An anomaly is not automatically an error.
- Passing dbt tests does not prove that all business logic is correct.
- The LLM must not invent models, columns, tests, failures, metrics, or relationships.
- The LLM must not create, remove, or modify deterministic findings.
- Structured JSON is the primary output contract.
- Invalid LLM JSON must be retried and validated.
- A deterministic fallback report must remain available.
- Prompt-injection content must not override system rules or evidence boundaries.
- M9 must remain understandable and testable without requiring an LLM.
- M9 should be MCP-ready in architecture: resources, read-only tools, and controlled prompts must be separable from the deterministic core.
- M9 must not claim to implement the MCP protocol until an actual MCP server/client and protocol transport exist.

---

## 4. Scope

### 4.1 In Scope

M9 will continue using:

- BigQuery monitoring tables from M8
- dbt artifact-derived metadata
- `model_lineage_edges`
- SQL
- Python
- deterministic rule evaluation
- evidence packages
- structured JSON reports
- Markdown reports
- controlled LLM explanations
- schema validation
- prompt-injection detection
- deterministic fallback behavior

### 4.2 Out of Scope

M9 will not introduce:

- Neo4j
- RDF
- SPARQL
- ontology platforms
- a dedicated graph database
- graph visualization UI
- React portal development within M9 itself; a usable React monitoring MVP is planned immediately after M9
- an enterprise knowledge graph platform
- a general AI agent platform
- Airflow
- Cloud Composer
- Dagster
- Prefect
- a replacement for the existing Cloud Run Job orchestration

---

## 5. High-Level Architecture

```text
M8 BigQuery Monitoring Tables
            |
            v
Evidence Extraction
            |
            v
Normalized Evidence Package
            |
            v
Graph Context Enrichment
            |
            v
Deterministic Rule Evaluation
            |
            v
Structured Findings
            |
            +--------------------+
            |                    |
            v                    v
Deterministic JSON Report   Deterministic Markdown Report
            |
            v
Controlled LLM Explanation
            |
            v
JSON Schema and Semantic Validation
            |
            +--------------------+
            |                    |
            v                    v
Validated LLM Report       Deterministic Fallback
```

---

## 6. Architecture Layers

### 6.1 Fact Base

The M8 BigQuery dataset `olist_monitoring` is the M9 fact base.

It currently contains six append-only monitoring tables:

1. `pipeline_runs`
2. `model_run_results`
3. `test_run_results`
4. `model_metadata_snapshots`
5. `model_column_snapshots`
6. `model_lineage_edges`

M9 should read the persisted monitoring tables rather than treating local dbt artifact files as its primary runtime interface.

### 6.2 Knowledge Base

The knowledge base will contain:

- review rubric definitions;
- deterministic rule definitions;
- rule applicability conditions;
- severity policies;
- configurable thresholds;
- required evidence definitions;
- conclusion boundaries;
- optional model expectations;
- optional business expectations.

Two knowledge categories must remain separate:

#### Monitoring-derived knowledge

Examples:

- a model execution failed;
- a test did not pass;
- runtime increased relative to history;
- a column disappeared;
- a model description is missing.

#### Explicit business or modeling expectations

Examples:

- `fct_orders` has grain `order_id`;
- `order_id` is the declared primary key;
- `customer_unique_id` represents a real customer across orders;
- a revenue metric must not multiply payment values after a join.

Business expectations must be explicitly configured.

They must not be inferred by the LLM without evidence.

### 6.3 Graph-Structured Knowledge Layer

M9 will use `model_lineage_edges` to build in-memory graph indexes such as:

```text
upstream_by_node
downstream_by_node
sources_by_model
models_by_source
```

Normal Python dictionaries and sets are sufficient for the MVP:

```python
dict[str, set[str]]
```

Breadth-first search or depth-first search may later be used to calculate downstream impact.

No dedicated graph database is required.

### 6.4 Deterministic Inference Engine

Each rule will evaluate:

```text
EvidencePackage
+ RuleDefinition
+ GraphContext
+ OptionalModelExpectation
```

A rule evaluation may produce:

- zero or more findings;
- missing-evidence records;
- a rule evaluation status;
- affected entities;
- downstream impact;
- evidence references.

Allowed rule statuses will include:

- `PASS`
- `TRIGGERED`
- `NOT_APPLICABLE`
- `NOT_EVALUATED`
- `ERROR`

`ERROR` means that the rule itself could not complete.

It does not mean that the pipeline failed.

### 6.5 Controlled LLM Explanation Layer

The LLM may receive only approved structured inputs, including:

- deterministic findings;
- evidence records;
- missing-evidence records;
- lineage impact;
- rubric definitions;
- rule conclusion boundaries.

The LLM must not:

- create a new finding;
- remove a finding;
- change severity;
- change a rule result;
- invent an entity or metric;
- describe an anomaly as a confirmed error without evidence;
- describe passing tests as proof of complete business correctness.

### 6.6 MCP-Ready Service Boundary

M9 should follow MCP-style separation of concerns without making the MCP protocol a dependency of the deterministic reviewer.

The architecture should distinguish:

- resources that expose approved context;
- read-only tools that retrieve scoped evidence;
- controlled prompts that request bounded explanations;
- the deterministic inference engine that remains the source of truth.

Candidate MCP-style resources include:

```text
olist://runs/{run_id}/report
olist://runs/{run_id}/findings
olist://models/{model_id}/history
olist://models/{model_id}/downstream
olist://rules/catalog
olist://schemas/review-report
```

Candidate read-only tools include:

```text
get_pipeline_run
get_findings
get_model_history
get_test_evidence
get_downstream_models
```

These interfaces must use strict input and output schemas and expose only the minimum evidence required for the requested task.

The following capabilities must not be exposed to an LLM or future MCP client:

```text
execute_arbitrary_sql
change_rule
change_severity
delete_monitoring_history
modify_pipeline_evidence
unrestricted_pipeline_execution
```

MCP does not replace deterministic inference.

The intended sequence remains:

```text
BigQuery facts
        |
        v
Deterministic rules
        |
        v
Verified findings
        |
        v
MCP-ready resources and read-only tools
        |
        v
Controlled LLM explanation
```

The current M9 implementation should therefore be described as:

> MCP-ready or MCP-inspired in architecture

It must not be described as:

> using the MCP protocol

until an actual MCP server/client, discovery flow, transport, and protocol implementation are added.

A future MCP protocol layer may be added through the same backend service layer used by the React monitoring portal, while React continues to consume ordinary REST endpoints.

---

## 7. Core Entities

The initial M9 domain model includes:

- `PipelineRun`
- `Model`
- `Source`
- `Column`
- `Test`
- `Rule`
- `Finding`
- `Evidence`

---

## 8. Core Relationships

The initial relationship model includes:

- `PipelineRun EXECUTED Model`
- `PipelineRun PRODUCED TestResult`
- `Model HAS_COLUMN Column`
- `Model HAS_TEST Test`
- `Model DEPENDS_ON Model`
- `Model DEPENDS_ON Source`
- `Test VALIDATES Model`
- `Rule PRODUCED Finding`
- `Finding SUPPORTED_BY Evidence`

These relationships form a graph-structured knowledge representation without requiring a graph database.

---

## 9. Review Rubric

The initial review rubric contains eight dimensions:

1. Correctness
2. Data Quality / Integrity
3. Reliability
4. Observability
5. Maintainability
6. Business Readiness
7. Scope Discipline
8. Communication / Documentation

### 9.1 Dimension Status Values

Each dimension may return:

- `PASS`
- `WARN`
- `FAIL`
- `NOT_EVALUATED`

### 9.2 Status Meaning

| Status | Meaning |
|---|---|
| `PASS` | Applicable rules were evaluated and produced no findings or material missing evidence. |
| `WARN` | Medium/low findings or important missing evidence exist. |
| `FAIL` | At least one critical/high finding exists. |
| `NOT_EVALUATED` | The available evidence or rule coverage is insufficient to reach a conclusion. |

### 9.3 MVP Scoring Decision

The M9 MVP will not produce a percentage score such as `87/100`.

A numeric score would create false precision because:

- evidence coverage differs across dimensions;
- business expectations are not yet complete;
- a critical issue must not be averaged away by multiple passing checks;
- some dimensions will initially remain `NOT_EVALUATED`.

---

## 10. Verified M8 Evidence Inventory

The following inventory is based on the actual BigQuery schemas exported from:

```text
olist_monitoring.INFORMATION_SCHEMA.COLUMNS
```

The six tables contain 98 fields in total.

| Table | Field Count |
|---|---:|
| `pipeline_runs` | 23 |
| `model_run_results` | 16 |
| `test_run_results` | 17 |
| `model_metadata_snapshots` | 20 |
| `model_column_snapshots` | 12 |
| `model_lineage_edges` | 10 |
| **Total** | **98** |

All fields are currently reported as nullable by BigQuery.

M9 must therefore validate required evidence values at runtime rather than assuming that key fields are always populated.

---

## 11. Table-Level Evidence

### 11.1 `pipeline_runs`

Verified fields include:

```text
monitoring_run_id
dbt_invocation_id
job_name
environment
dbt_version
generated_at
ingested_at
run_started_at
run_completed_at
total_elapsed_time_seconds
status
models_total
models_success
models_error
models_skipped
tests_total
tests_passed
tests_failed
tests_warned
tests_error
artifact_manifest_path
artifact_run_results_path
artifact_catalog_path
```

Primary M9 uses:

- identify a pipeline run;
- select the latest comparable run;
- evaluate pipeline status;
- evaluate run-level duration;
- access model and test summary counts;
- locate persisted artifact metadata paths.

### 11.2 `model_run_results`

Verified fields include:

```text
monitoring_run_id
dbt_invocation_id
unique_id
model_name
resource_type
package_name
database_name
schema_name
alias
materialized
status
execution_time_seconds
thread_id
message
adapter_response_json
ingested_at
```

Primary M9 uses:

- evaluate model execution status;
- identify failed, skipped, or errored models;
- compare execution times;
- retain dbt messages and adapter evidence.

### 11.3 `test_run_results`

Verified fields include:

```text
monitoring_run_id
dbt_invocation_id
unique_id
test_name
test_type
test_metadata_name
model_unique_id
model_name
column_name
status
severity
failures
execution_time_seconds
thread_id
message
adapter_response_json
ingested_at
```

Primary M9 uses:

- evaluate non-passing tests;
- identify test severity and failure count;
- associate a test with a model;
- associate a test with a column;
- identify test type without parsing the test name;
- evaluate declared key-test coverage once model expectations exist.

### 11.4 `model_metadata_snapshots`

Verified fields include:

```text
monitoring_run_id
dbt_invocation_id
unique_id
model_name
resource_type
package_name
database_name
schema_name
alias
relation_name
materialized
path
original_file_path
description
tags_json
meta_json
row_count
bytes
catalog_metadata_json
ingested_at
```

Primary M9 uses:

- compare model inventories;
- identify models absent from a comparable run;
- compare row counts;
- compare storage bytes;
- evaluate model-description coverage;
- inspect materialization and model metadata.

### 11.5 `model_column_snapshots`

Verified fields include:

```text
monitoring_run_id
dbt_invocation_id
model_unique_id
model_name
resource_type
column_name
data_type
column_index
description
tests_json
catalog_column_metadata_json
ingested_at
```

Primary M9 uses:

- identify added or removed columns;
- identify data-type changes;
- compare column ordering;
- evaluate column documentation coverage;
- access supplementary column-level test metadata.

### 11.6 `model_lineage_edges`

Verified fields include:

```text
monitoring_run_id
dbt_invocation_id
parent_unique_id
parent_name
parent_resource_type
child_unique_id
child_name
child_resource_type
dependency_type
ingested_at
```

Primary M9 uses:

- build upstream and downstream indexes;
- identify direct dependencies;
- traverse downstream model impact;
- distinguish source-to-model and model-to-model relationships.

Lineage evidence is limited to dependencies represented in the dbt manifest.

It does not prove the full set of external dashboards, manual SQL consumers, or systems outside dbt.

---

## 12. Evidence Availability Matrix

| Evidence Requirement | Source | Availability | Initial Use |
|---|---|---|---|
| Pipeline status | `pipeline_runs.status` | Available | R001 |
| Pipeline duration | `pipeline_runs.total_elapsed_time_seconds` | Available | Run-level comparison |
| Model execution status | `model_run_results.status` | Available | R002 |
| Model runtime | `model_run_results.execution_time_seconds` | Available | R006 |
| Test status | `test_run_results.status` | Available | R003 |
| Test failure count | `test_run_results.failures` | Available | R003 |
| Test-to-model mapping | `test_run_results.model_unique_id` | Available | R003, R010 |
| Test-to-column mapping | `test_run_results.column_name` | Available | R010 |
| Test type | `test_run_results.test_type`, `test_metadata_name` | Available | R010 |
| Model inventory | `model_metadata_snapshots.unique_id` | Available | R004 |
| Model row count | `model_metadata_snapshots.row_count` | Available | R005 |
| Model description | `model_metadata_snapshots.description` | Available | R008 |
| Column inventory | `model_column_snapshots.column_name` | Available | R007 |
| Column data type | `model_column_snapshots.data_type` | Available | R007 |
| Column description | `model_column_snapshots.description` | Available | R009 |
| Downstream lineage | `model_lineage_edges` | Available | Impact analysis |
| Declared primary key | Knowledge base | Missing | R010 |
| Declared model grain | Knowledge base | Missing | Grain validation |
| Approved metric definition | Knowledge base | Missing | Revenue validation |
| Source freshness result | Current fact base | Missing | Freshness review |
| Compiled SQL / join logic | Current fact base | Missing | Double-counting review |
| Approved implementation specification | Knowledge base | Missing | Spec–implementation drift |

---

## 13. Initial Deterministic Rule Readiness

| Rule ID | Rule | Readiness |
|---|---|---|
| M9-R001 | Pipeline Run Unsuccessful | `READY` |
| M9-R002 | Model Execution Non-Success | `READY` |
| M9-R003 | Test Result Non-Passing | `READY` |
| M9-R004 | Model Missing from Current Run | `READY` |
| M9-R005 | Row-Count Anomaly | `READY` |
| M9-R006 | Runtime Regression | `READY` |
| M9-R007 | Schema Drift Detected | `READY` |
| M9-R008 | Missing Model Description | `READY` |
| M9-R009 | Low Column Documentation Coverage | `READY` |
| M9-R010 | Declared Key Tests Missing | `BLOCKED_BY_MISSING_EXPECTATION` |

R010 is not blocked by missing dbt test metadata.

The current fact base already contains test-to-model, test-to-column, and test-type evidence.

It is blocked because M9 does not yet have an explicit knowledge-base declaration of each model's expected grain or primary key.

---

## 14. Evidence and Conclusion Boundaries

### 14.1 Model Absence

Evidence may prove:

> A model observed in a previous comparable run is absent from the current run.

It does not prove whether the model was:

- deleted;
- renamed;
- disabled;
- excluded intentionally;
- excluded accidentally.

### 14.2 Row-Count Anomaly

Evidence may prove:

> The current row count differs materially from a historical baseline.

It does not prove:

- duplicate rows exist;
- data is missing;
- a join exploded;
- the business volume could not have changed legitimately.

### 14.3 Runtime Regression

Evidence may prove:

> The current runtime is slower than the selected historical baseline.

It does not prove whether the cause is:

- SQL logic;
- data volume;
- BigQuery resource conditions;
- an upstream service;
- temporary infrastructure behavior.

### 14.4 Passing Tests

Evidence may prove:

> The configured dbt tests passed for the reviewed run.

It does not prove:

- all business logic is correct;
- all important risks have tests;
- revenue cannot be double counted;
- model grain is correct;
- documentation is accurate.

### 14.5 Missing Evidence

Missing evidence must not automatically become a finding that claims a defect exists.

It means:

> M9 cannot evaluate the rule reliably with the available evidence.

The correct rule status is normally:

```text
NOT_EVALUATED
```

A corresponding missing-evidence record should explain:

- what evidence is missing;
- why it is required;
- which conclusion cannot be reached;
- what future source could provide it.

---

## 15. Current M9.1 Status

Completed:

- M9 positioning defined
- scope and non-scope defined
- five-layer architecture defined
- core entities and relationships defined
- initial rubric defined
- six BigQuery table schemas verified
- 98 persisted fields inventoried
- evidence availability matrix created
- initial rule readiness classified
- MCP-ready service boundaries and future read-only resource/tool concepts defined

Not yet completed:

- detailed rule catalog file
- strict JSON Schema
- model expectations file
- evidence extraction query
- Python evidence package
- deterministic rule implementation
- JSON reporter
- Markdown reporter
- LLM integration
- guardrail implementation

---

## 16. Post-M9 Product Sequence

After M9 is complete, the next priority is to build a usable pipeline monitoring product rather than moving immediately into additional analytics or modeling work.

The immediate sequence is:

```text
M9 Expert System
        |
        v
Lightweight FastAPI Backend
        |
        v
React Pipeline Monitoring MVP
```

The React MVP should consume:

- `olist_monitoring`;
- deterministic M9 review reports;
- optional controlled LLM explanations;
- lineage impact and model history.

The first React MVP should focus on:

1. latest pipeline status;
2. run history;
3. model and test results;
4. deterministic findings and evidence;
5. downstream impact.

React must not access BigQuery directly.

A lightweight Python/FastAPI service deployed to Cloud Run should provide controlled endpoints and reuse the same service layer that may later support MCP resources and read-only tools.

Power BI business analytics, historical replay, broader production hardening, and the separate Data Vault 2.0 prototype remain in the wider roadmap.

Their timing and implementation depth will be reconsidered after the React MVP based on project progress, user value, development effort, and cost.

The separate Data Vault 2.0 prototype will continue to use another independent Olist-related dataset and will not modify the main Olist platform.

---

## 17. Next Planned Step

The next M9.1 step is to create:

```text
dbt/monitoring/reviewer/config/rule_catalog.yml
```

The first version will define the initial deterministic rules without implementing them.

Each rule definition should include:

```text
rule_id
version
name
description
dimensions
default_severity
applicability
required_evidence
trigger_logic
can_prove
cannot_prove
risk
recommendation
implementation_status
```

No Python rule engine should be written before the rule catalog has been reviewed.