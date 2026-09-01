# M4 - dbt Staging Layer Validation

## Goal

M4 implemented and validated the dbt staging layer for the Olist E-Commerce Analytics Portal.

The objective was to connect dbt to BigQuery, register raw source tables, create source-aligned staging views, and add basic model documentation and data quality tests.

## Completed Scope

- Initialized dbt project under `dbt/`
- Configured BigQuery connection through local `profiles.yml`
- Registered 9 raw BigQuery tables as dbt sources in `sources.yml`
- Created 9 staging models as BigQuery views
- Added model and column documentation in `schema.yml`
- Added 39 dbt data tests
- Validated staging models with `dbt run` and `dbt test`

## dbt Environment

- dbt-core: 1.11.11
- dbt-bigquery: 1.11.3
- BigQuery project: `balmy-nuance-468118-g4`
- Raw dataset: `olist_raw`
- Staging dataset: `olist_staging`

## Staging Models Created

- `stg_customers`
- `stg_geolocation`
- `stg_orders`
- `stg_order_items`
- `stg_order_payments`
- `stg_order_reviews`
- `stg_products`
- `stg_sellers`
- `stg_product_category_translation`

## Validation Commands

```bash
dbt debug
dbt parse
dbt run --select staging
dbt test --select staging
dbt ls --select staging
```

## Validation Results

**dbt debug**

The BigQuery connection was validated successfully.

```text
Connection test: [OK connection ok]
```

**dbt run**

All 9 staging views were created successfully.

```text
PASS=9 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=9
```

**dbt test**

All 39 staging data tests passed successfully.

```text
PASS=39 WARN=0 ERROR=0 SKIP=0 NO-OP=0 TOTAL=39
```

**dbt ls**

`dbt ls --select staging` confirmed that all 9 staging models and their associated data tests are registered in the dbt project.

## Notes

The warning about the unused `marts` configuration path is expected at this stage because mart models have not been implemented yet.

M4 focuses only on the raw-to-staging transformation layer. Fact tables, dimension tables, reporting marts, and dashboard models are out of scope for this milestone. Local BigQuery authentication and `profiles.yml` are kept outside the repository.

## Outcome

M4 successfully established a working dbt staging layer on top of the BigQuery raw layer.

The project now has:

```text
BigQuery raw tables
        ↓
dbt sources
        ↓
dbt staging views
        ↓
dbt documentation and tests
```

## Next Milestone

M5 will focus on dimensional modeling and analytics marts, including fact and dimension models for reporting.