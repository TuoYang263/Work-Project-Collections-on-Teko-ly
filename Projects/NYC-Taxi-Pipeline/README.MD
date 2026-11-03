# NYC Yellow Taxi Data Pipeline

This repository contains two related generations of a NYC Yellow Taxi data pipeline:

- **Legacy pipeline and dashboard:** Airflow, PySpark, Delta Lake, BigQuery and Streamlit. The original dashboard and supporting jobs remain in the repository.
- **NYC V2 (in progress):** an AWS-first batch data platform using versioned S3 raw inputs, Airflow, PySpark/Delta Bronze–Silver–Gold processing, quality gates and read-only operational diagnostics. Redshift serving and an AI-assisted incident investigator are planned work, not deployed features.

The two tracks share code but do not have the same verification status. In particular, passing isolated tests does not establish that the full AWS V2 DAG has run successfully end to end.

## Verified milestones and current limits

| Area | Evidence to date | Limit |
| --- | --- | --- |
| Versioned S3 raw ingestion | Object-version references, manifests, integrity checks and isolated tests | Does not imply a completed V2 production pipeline |
| S3 to Delta Bronze | An isolated Airflow smoke run processed January 2024 NYC Yellow Taxi data: **2,964,624 rows**, **21 columns** | Local Spark compute and isolated Delta output; not an end-to-end main DAG run |
| Bronze / Silver / Gold transformations | Transformation and quality-gate code with focused tests | Full V2 orchestration, Silver, Gold and serving integration remain unverified |
| Run Evidence | Per-run V2 lifecycle records and read-only evidence/status adapters | Run-level status does not describe all task retries |
| Task diagnostics | Read-only TaskInstance, TaskFail and attempt-log reference APIs; verified against a copied PostgreSQL 13 database | TaskInstance is a current snapshot, not a complete attempt history |
| Incident Library (Unit 1C) | Pointer manifests, metadata hashes, private staging and scoped capture for three historical lifecycle-test runs | No published, independently verified or restorable incident snapshot yet |
| Redshift | Intended V2 serving target | Not implemented or validated |
| AI investigator, agents, Bedrock | Design and evaluation plan | Not implemented or validated |

The isolated Bronze test and its limits are documented in [`airflow_pipeline/docs/nyc_v2_bronze_smoke_runtime.md`](airflow_pipeline/docs/nyc_v2_bronze_smoke_runtime.md). Incident preservation and agent evaluation are described in the Observability ADR and Agent Evaluation ADR under `airflow_pipeline/docs/`. Those documents distinguish proposed controls from implemented behavior.

## Architecture

```text
NYC TLC monthly Parquet
         |
         v
S3 raw object + VersionId + manifest
         |
         v
Airflow-controlled ingestion -> Delta Bronze -> Silver -> Gold
                                                |
                                                v
                                         Gold quality checks
                                                |
                                   planned Redshift serving

Airflow Run Evidence ------+
TaskInstance / TaskFail ---+--> read-only diagnostics
Attempt log references ---+             |
                                        v
                            incident capture (in progress)
                                        |
                            verified snapshot + offline replay
                                  (not yet delivered)
                                        |
                             read-only AI investigator
                                  (not yet delivered)
```

The data plane is authoritative. A future AI investigator may explain evidence but must not change pipeline state, execute a recovery action, or override deterministic quality decisions.

## Project layout

```text
Projects/NYC-Taxi-Pipeline/
  airflow_pipeline/
    dags/                     Airflow DAGs; includes isolated test DAGs
    scripts/                  Ingestion, transformations, control and diagnostics
    scripts/diagnostics/      Read-only evidence and incident modules
    tests/                    Unit and targeted integration tests
    docs/                     Design decisions and verification notes
    config/                   Local runtime configuration
    data/                     Local outputs; mostly excluded from Git
  streamlit_app/              Legacy BigQuery-backed dashboard
  docker-compose*.yaml        Local Airflow and isolated test environments
  (repository root)/.github/workflows/  Active GitHub Actions for this monorepo
```

GitHub Actions must be created at the **monorepo root** `.github/workflows/`, not under this project folder. Historical workflow files under `Projects/NYC-Taxi-Pipeline/.github/workflows/` are not active GitHub Actions workflows.

The large `delta_medallion_pipeline.py` remains the existing compute implementation. Its eventual split should follow verified interfaces rather than precede the remaining end-to-end checks.

## Local development and verification

Developed on Windows with WSL Ubuntu, Python 3.11 and Docker. The Airflow runtime uses a container image with Airflow 2.8.1 and PostgreSQL 13 in the verified lifecycle-test environment. Your local WSL Python environment does not need to install Airflow for the focused incident tests.

Clone the monorepo and change into the project directory:

```bash
git clone https://github.com/TuoYang263/Work-Project-Collections-on-Teko-ly.git
cd Work-Project-Collections-on-Teko-ly/Projects/NYC-Taxi-Pipeline
```

The following checks require only Python and `pytest`; they do not start a DAG or contact AWS:

```bash
PYTHONPATH=airflow_pipeline python -m pytest -q \
  airflow_pipeline/tests/test_incident_manifest.py \
  airflow_pipeline/tests/test_incident_capture.py \
  airflow_pipeline/tests/test_incident_staging.py \
  airflow_pipeline/tests/test_incident_db_capture.py \
  airflow_pipeline/tests/test_incident_log_capture.py
```

Tests involving real S3, Spark/Delta or Airflow/PostgreSQL runtime have separate prerequisites and may have operational costs. Read their test/runbook instructions before execution. Do not start the main medallion DAG as part of a documentation or code review.

## Local runtime and credentials

The Docker Compose files are **development configurations**, not a production security baseline. Review bound ports, default passwords, session keys, mounted directories and active services before exposing anything beyond localhost.

Never commit AWS login caches, cloud keys, `.env` files, private incident snapshots or original Airflow logs. In a deployed environment, replace local test credentials with managed identities and appropriately scoped roles. The current S3 and Delta smoke run has its own protected test storage and restart guidance in the runbook.

## Legacy dashboard

The original BigQuery-backed Streamlit dashboard remains in `streamlit_app/`. Its historical deployment URL, recorded in the older README, was:

https://nyc-taxi-dashboard-render.onrender.com

The link's present availability has not been verified as part of the NYC V2 architecture review. This dashboard is not evidence that Redshift serving or the AWS V2 main DAG has been deployed.

## Before presenting the project

- Distinguish **implemented and tested** components from planned architecture.
- State whether a result came from a unit test, an isolated runtime check or a complete DAG run.
- Describe the incident log artifacts as **bounded excerpts**, not full preserved logs.
- Do not mark a historical incident `SNAPSHOT_VERIFIED` until publication, integrity verification, backup/restore and offline replay meet their acceptance checks.

NYC Yellow Taxi data is published by the NYC Taxi & Limousine Commission (TLC).
