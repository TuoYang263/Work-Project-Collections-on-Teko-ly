# M9 Closing Notes — Pipeline Quality Reviewer and Explanation Layer

## Document status

```text
Completed: 2026-08-10
```


## 1. Scope

M9 adds a rule-based review layer on top of the monitoring data created in M8.

The main goal is simple:

- read pipeline evidence from BigQuery
- evaluate known reliability rules
- keep the rule result deterministic and auditable
- build a small finding package for triggered issues
- use Vertex AI only to explain existing findings
- keep the deterministic review valid even if the AI call fails

M9 does not let the model decide whether the pipeline passed, failed, or is healthy.

The reviewer is the source of truth.

---

## 2. Position in the project

```text
dbt build
  ↓
dbt artifacts
  ↓
M8 artifact parser and BigQuery monitoring tables
  ↓
M9 evidence loader
  ↓
deterministic reviewer
  ↓
rule evaluations
  ↓
finding package
  ↓
Vertex AI explanation
  ↓
final review JSON
```

M8 provides the historical evidence.

M9 reads that evidence and turns it into review results.

---

## 3. Monitoring tables used by M9

M9 reads from the append-only tables created in M8:

- `pipeline_runs`
- `model_run_results`
- `test_run_results`
- `model_metadata_snapshots`

The other M8 metadata tables remain available for later use:

- `model_column_snapshots`
- `model_lineage_edges`

Historical comparison uses previous successful runs with the same:

- `job_name`
- `environment`

Comparable runs are ordered from newest to oldest.

The reviewer currently keeps up to 5 previous runs for historical baselines.

---

## 4. Evaluation results

Each rule returns one of three results:

### PASS

The required evidence exists and the rule condition is not triggered.

### TRIGGERED

The required evidence exists and the configured rule condition is triggered.

### NOT_EVALUATED

The reviewer cannot safely evaluate the rule because required evidence is missing, invalid, duplicated, or not applicable.

`NOT_EVALUATED` is not treated as `PASS`.

This was important during R005 because BigQuery views do not provide physical row counts in the same way as tables.

For example:

```text
materialized = view
row_count = None
```

For those models, R005 returns:

```text
NOT_EVALUATED
```

instead of treating the missing value as zero.

---

## 5. Rules implemented in M9

### R001 — Pipeline Run Unsuccessful

Checks the selected pipeline run status.

Main source:

```text
pipeline_runs
```

Expected healthy state:

```text
status = success
```

A non-successful run triggers the rule.

---

### R002 — Model Execution Non-Success

Checks each model execution in the selected run.

Main source:

```text
model_run_results
```

A non-success model execution triggers the rule.

---

### R003 — Test Result Non-Passing

Checks dbt test results.

Main source:

```text
test_run_results
```

Main behavior:

- `pass` → PASS
- `warn` → TRIGGERED / MEDIUM
- other non-passing states → TRIGGERED / HIGH

---

### R004 — Model Missing from Current Run

Compares the current model inventory with the immediately previous comparable successful run.

Main source:

```text
model_metadata_snapshots
```

Comparison:

```text
previous model inventory
vs
current model inventory
```

If a model existed in the previous comparable run but is missing from the current run:

```text
TRIGGERED
severity = MEDIUM
```

The previous run must match the same:

- job
- environment

The selected run and baseline run must also contain valid inventory evidence.

---

### R005 — Row-Count Anomaly

Compares the current model row count with the median of recent comparable successful runs.

Main source:

```text
model_metadata_snapshots.row_count
```

Baseline:

```text
median of up to 5 recent comparable runs
```

Trigger rule:

```text
relative change >= 30%
AND
absolute change >= 100 rows
```

Both increases and decreases are treated as possible anomalies.

Severity:

```text
relative change >= 100% → HIGH
relative change >= 30%  → MEDIUM
```

For a zero baseline, relative change is undefined.

In that case, the reviewer uses the absolute threshold instead of returning a false relative value.

Example:

```text
baseline = 0
current = 150
absolute change = 150
```

This can still trigger the rule.

### Important R005 evidence boundary

During the real BigQuery run:

```text
R005 evaluations: 21
PASS: 9
NOT_EVALUATED: 12
```

The 12 `NOT_EVALUATED` models were views.

Example:

```text
model.olist_ecommerce_analytics.stg_orders
materialized = view
row_count = None
```

This is expected because a view does not have the same stored row-count metadata as a physical table.

No code change was made to force these models into PASS.

---

### R006 — Runtime Regression

Compares the current model execution time with the median of recent successful comparable runs.

Main source:

```text
model_run_results.execution_time_seconds
```

Baseline:

```text
median of up to 5 recent successful model executions
```

Only slower execution is treated as regression.

Trigger rule:

```text
relative increase >= 50%
AND
absolute increase >= 5 seconds
```

Severity:

```text
relative increase >= 100%
AND
absolute increase >= 30 seconds
→ HIGH
```

Other triggered regressions use:

```text
MEDIUM
```

A zero or negative historical median cannot be used safely and returns:

```text
NOT_EVALUATED
```

---

## 6. Historical baseline helper

The historical rules share a small median helper.

```python
from collections.abc import Iterable
from statistics import median


def median_baseline(
    values: Iterable[int | float | None],
) -> float | None:
    usable_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not usable_values:
        return None

    return float(median(usable_values))
```

Important behavior:

- ignores `None`
- accepts zero as a real value
- returns `None` when no usable value exists

---

## 7. Evidence loading

The BigQuery evidence loader supports:

- explicit run selection
- latest run selection by job and environment
- comparable historical runs
- historical model execution rows
- historical model metadata rows

Comparable run selection follows these rules:

```text
same job_name
same environment
status = success
older than current run
newest first
```

Historical model rows are loaded with:

```sql
WHERE monitoring_run_id IN UNNEST(@monitoring_run_ids)
```

Array query parameters are validated before execution.

---

## 8. Review service

The review service now runs both rule groups:

```text
status rules:
R001
R002
R003
```

and:

```text
historical rules:
R004
R005
R006
```

The results are merged into one review.

Conceptually:

```python
status_evaluations = evaluator.evaluate_status_rules(...)

historical_evaluations = evaluator.evaluate_historical_rules(...)

evaluations = [
    *status_evaluations,
    *historical_evaluations,
]
```

This keeps the service simple.

The evaluator decides rule behavior.

The loader decides how evidence is read.

---

## 9. Finding Package

The full evaluation set can be large.

Vertex AI does not need every PASS evaluation.

M9 therefore builds a smaller finding package.

The package contains:

```text
monitoring_run_id
summary
findings
```

Summary example:

```json
{
  "total_evaluations": 179,
  "pass": 166,
  "triggered": 1,
  "not_evaluated": 12
}
```

Only evaluations with:

```text
result = TRIGGERED
```

are included in `findings`.

This reduces the amount of data sent to the model and keeps the explanation focused.

Each finding has a stable `finding_id`.

The ID is built from deterministic data, including:

- monitoring run
- rule
- entity type
- entity

The model is not allowed to create new finding IDs.

---

## 10. Vertex AI role

Vertex AI is only an explanation layer.

It receives the deterministic finding package and explains:

- what happened
- possible impact
- useful investigation steps

It must not change:

- rule result
- severity
- evidence
- threshold
- entity identifier
- finding identifier

It must not invent new findings.

The runtime model used in M9 is:

```text
gemini-2.5-flash
```

The client uses Vertex AI through Google Gen AI SDK.

Example client setup:

```python
client = genai.Client(
    enterprise=True,
    project=project_id,
    location=location,
    http_options=types.HttpOptions(api_version="v1"),
)
```

---

## 11. Structured output

The explanation call uses JSON output:

```python
response_mime_type="application/json"
```

and a response schema.

The response contains:

```text
pipeline_summary
findings[]
```

Each returned finding contains:

```text
finding_id
explanation
impact
recommended_actions
```

This makes the result easier to validate and easier to use later in the portal.

---

## 12. AI response validation

The model response is parsed and checked before it is accepted.

The validator checks that:

- the response is a JSON object
- `findings` is a list
- no finding ID is duplicated
- no finding ID is invented
- no deterministic finding is missing

The returned finding ID set must exactly match the deterministic finding ID set.

Conceptually:

```text
deterministic finding IDs
==
returned explanation finding IDs
```

This is stronger than only checking whether the model invented an ID.

---

## 13. AI runtime states

M9 uses three AI states.

### SKIPPED

If there are no triggered findings:

```text
findings = []
```

Vertex AI is not called.

This avoids unnecessary API calls.

### SUCCESS

If findings exist and the structured explanation is generated and validated:

```text
ai_status = SUCCESS
```

### UNAVAILABLE

If the model call, parsing, validation, or SDK path fails:

```text
ai_status = UNAVAILABLE
```

The deterministic review still remains valid.

This was a core M9 design rule:

```text
AI failure must not invalidate deterministic review.
```

---

## 14. Testing approach

M9 uses standard library `unittest`.

`pytest` is not required for the reviewer.

The final reviewer test suite contains:

```text
53 tests
```

Final result:

```text
Ran 53 tests
OK
```

Tests cover:

- median baseline behavior
- BigQuery evidence loading
- status rules
- R004
- R005
- R006
- immutable evidence behavior
- finding package
- AI response validation
- AI success path
- AI skip path
- AI failure fallback

---

## 15. Mocking the AI dependency

AI unit tests do not call Vertex.

`unittest.mock.patch` temporarily replaces the real explanation function.

Example idea:

```python
@patch(
    "pipeline_reviewer.ai_explainer.explain_finding_package"
)
```

A fake success response can be set with:

```python
mock_explain.return_value = {...}
```

A fake Vertex failure can be created with:

```python
mock_explain.side_effect = RuntimeError(
    "Vertex unavailable"
)
```

Useful mental model:

```text
patch
=
control how an external dependency behaves

assert
=
check how our code handles that behavior
```

This keeps tests:

- fast
- deterministic
- independent from network access
- free from API cost

---

## 16. Rule catalog validation

The rule catalog is validated separately.

Command:

```bash
python dbt/monitoring/reviewer/tests/validate_rule_catalog.py
```

Final validation passed:

```text
YAML parsing passed.
Whitespace and final-newline checks passed.
Top-level evaluation policy validation passed.
Rule ID uniqueness and R001-R006 completeness passed.
R003 evidence, trigger, and severity validation passed.
R004 model-inventory comparison validation passed.
R005 row-count baseline and threshold validation passed.
R006 runtime baseline and threshold validation passed.
All referenced evidence sources are valid M8 monitoring tables.
```

---

## 17. Useful commands

### Format reviewer code

```bash
black dbt/monitoring/reviewer
```

### Compile a Python file

```bash
python -m py_compile \
  dbt/monitoring/reviewer/run_status_review.py
```

### Run all reviewer tests

```bash
python -m unittest discover \
  -s dbt/monitoring/reviewer/tests \
  -p "test_*.py" \
  -v
```

### Run only AI tests

```bash
python -m unittest \
  dbt.monitoring.reviewer.tests.test_ai_explainer \
  -v
```

### Run only finding package tests

```bash
python -m unittest \
  dbt.monitoring.reviewer.tests.test_finding_package \
  -v
```

### Run only R006 tests

```bash
python -m unittest \
  dbt.monitoring.reviewer.tests.test_evaluator_runtime_rule \
  -v
```

### Validate rule catalog

```bash
python dbt/monitoring/reviewer/tests/validate_rule_catalog.py
```

### Check Git whitespace problems

```bash
git diff --check
```

LF/CRLF warnings on Windows are not the same as a whitespace failure.

### Review changed files

```bash
git status --short
```

```bash
git diff --stat
```

---

## 18. Run the reviewer against BigQuery

Latest production-like review:

```bash
python dbt/monitoring/reviewer/run_status_review.py \
  --project-id balmy-nuance-468118-g4 \
  --dataset-id olist_monitoring \
  --job-name olist-dbt-build-job \
  --environment prod
```

Save output:

```bash
python dbt/monitoring/reviewer/run_status_review.py \
  --project-id balmy-nuance-468118-g4 \
  --dataset-id olist_monitoring \
  --job-name olist-dbt-build-job \
  --environment prod \
  > /tmp/m9_integrated_review_final.json
```

---

## 19. Vertex AI setup and connectivity check

The Vertex AI API must be enabled for the project.

Command used:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  --project=balmy-nuance-468118-g4
```

Verify:

```bash
gcloud services list \
  --enabled \
  --project=balmy-nuance-468118-g4 \
  --filter="name:aiplatform.googleapis.com"
```

Expected service:

```text
aiplatform.googleapis.com
```

---

## 20. Minimal Vertex connectivity test

A small client test was useful before integrating the reviewer.

Example:

```python
from google import genai
from google.genai import types

client = genai.Client(
    enterprise=True,
    project="balmy-nuance-468118-g4",
    location="us-central1",
    http_options=types.HttpOptions(api_version="v1"),
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly: VERTEX_OK",
)

print(response.text)

client.close()
```

Expected output:

```text
VERTEX_OK
```

This isolated:

- credentials
- API access
- project
- region
- model access
- SDK setup

before adding M9 logic.

---

## 21. Structured-output troubleshooting

A second small test checked the response schema separately from the main reviewer.

This was useful when the integrated call failed.

The standalone schema test succeeded and proved that:

```text
Vertex connectivity        OK
gemini-2.5-flash           OK
response_schema            OK
JSON structured output     OK
```

This narrowed the problem to local integration code instead of the Vertex service.

This was a useful debugging pattern:

```text
reduce the system
test one boundary
confirm it
move to the next boundary
```

---

## 22. Datetime serialization issue

The finding package contains BigQuery timestamps.

Python may keep these as `datetime` values.

The first Vertex integration failed because:

```text
Object of type datetime is not JSON serializable
```

The CLI output already used:

```python
json.dumps(
    payload,
    default=str,
)
```

The Vertex request needed the same boundary handling.

The fix was:

```python
contents=json.dumps(
    finding_package,
    indent=2,
    ensure_ascii=False,
    default=str,
)
```

This converts datetime values to strings when preparing the model input.

The evaluator and evidence objects did not need to be changed.

The conversion stays at the JSON boundary.

---

## 23. Final real validation

Final selected monitoring run:

```text
20260810T030139Z_35356a7d
```

Final evaluation summary:

```text
total evaluations: 179
PASS:              166
TRIGGERED:           1
NOT_EVALUATED:      12
```

The triggered rule was:

```text
M9-R006
```

Entity:

```text
model.olist_ecommerce_analytics.fct_order_payments
```

Observed runtime:

```text
current runtime:
22.507 seconds
```

Historical median:

```text
2.898 seconds
```

Approximate increase:

```text
absolute increase:
19.608 seconds

relative increase:
676.57%
```

This passed both R006 trigger thresholds:

```text
relative increase >= 50%
absolute increase >= 5 seconds
```

The finding was passed to Vertex AI.

Final AI status:

```text
SUCCESS
```

Returned AI findings:

```text
1
```

The returned `finding_id` matched the deterministic finding.

This validated the full M9 path:

```text
BigQuery evidence
→ deterministic R001-R006 evaluation
→ finding package
→ Vertex AI
→ structured JSON response
→ finding ID validation
→ final integrated report
```

---

## 24. Final code validation

Final unit test command:

```bash
python -m unittest discover \
  -s dbt/monitoring/reviewer/tests \
  -p "test_*.py" \
  -v
```

Result:

```text
Ran 53 tests in 0.308s

OK
```

Final rule catalog validation also passed.

`git diff --check` reported only Windows line-ending warnings:

```text
LF will be replaced by CRLF
```

No blocking whitespace error was reported.

---

## 25. Main design decisions kept in M9

M9 deliberately keeps several boundaries.

### Deterministic rules decide facts

The evaluator decides:

```text
PASS
TRIGGERED
NOT_EVALUATED
severity
```

### AI explains facts

Vertex AI explains existing findings and suggests investigation steps.

### Missing evidence is visible

Missing or unusable evidence returns:

```text
NOT_EVALUATED
```

instead of silently passing.

### Historical checks use comparable runs

Historical baselines use the same job and environment and only older successful runs.

### Findings are traceable

A finding keeps:

```text
rule
entity
evidence
threshold
result
reason
```

### AI is optional

No finding:

```text
SKIPPED
```

AI failure:

```text
UNAVAILABLE
```

AI success:

```text
SUCCESS
```

The deterministic review remains usable in all three cases.

---

## 26. M9 completed scope

M9 now includes:

- R001 pipeline status review
- R002 model execution review
- R003 dbt test review
- R004 missing-model comparison
- R005 row-count anomaly comparison
- R006 runtime regression comparison
- historical median baselines
- comparable-run loading
- immutable evaluation evidence
- finding package builder
- deterministic finding IDs
- Vertex AI explanation layer
- structured JSON output
- finding ID validation
- AI skip path
- AI failure fallback
- unit tests
- rule catalog validation
- real BigQuery integration validation

M9 is complete at this boundary.

The next project stage should build on these outputs rather than add more rule logic to M9.
