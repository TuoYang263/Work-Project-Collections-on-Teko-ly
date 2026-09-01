# Documentation Index

This directory contains both **current system documentation** and **historical milestone/design records**.

The project keeps historical documents because they show how decisions were made and validated, but the root README should not repeat all of that detail.

## Start here

| Document | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | Current system architecture and implementation boundaries |
| [`deployment.md`](deployment.md) | Current Render, GCP runtime, Scheduler, GitHub CI/CD, OIDC/WIF boundaries |
| [`m10_window_control.md`](m10_window_control.md) | Window/watermark controller, retry, transactions, CAS, validation |
| [`m10_portal_analytics.md`](m10_portal_analytics.md) | Portal, business analytics, reliability UI, statistical diagnostics |
| [`m9_expert_system_closing.md`](m9_expert_system_closing.md) | Final deterministic M9 reviewer behavior |

## Current operational references

- [`orchestration.md`](orchestration.md) - M7 orchestration design with current compatibility notes
- [`gcp_orchestration_commands.md`](gcp_orchestration_commands.md) - historical M7/M8 GCP deployment and validation runbook
- [`metadata_refresh.md`](metadata_refresh.md) - monitoring/artifact refresh design

`gcp_orchestration_commands.md` intentionally preserves the historical M7/M8 deployment commands and old image tags. Use [`deployment.md`](deployment.md) for the current deployment topology.

## Modeling and data design history

- [`source_data_overview.md`](source_data_overview.md)
- [`staging_layer_plan.md`](staging_layer_plan.md)
- [`m4_dbt_staging_validation.md`](m4_dbt_staging_validation.md)
- [`m5_dimensional_modeling_design.md`](m5_dimensional_modeling_design.md)
- [`m5_dbt_marts_validation.md`](m5_dbt_marts_validation.md)

These documents retain detailed source, grain, modeling, and validation decisions. The root README now keeps only a compact summary.

## Reliability design history

- [`m9_expert_system_design.md`](m9_expert_system_design.md) - design reasoning and rule-system plan
- [`m9_expert_system_closing.md`](m9_expert_system_closing.md) - final implemented M9 state

The deterministic reviewer is the source of truth for rule outcomes. Optional AI explanation is downstream and cannot override the rule result.

## Project and metadata records

- [`project_management.md`](project_management.md)
- metadata planning and inventories live under `../metadata/`

## Documentation policy

Use this rule when updating the project:

1. **README.md** explains the product and the most important engineering decisions.
2. **architecture.md** describes the current system, not milestone history.
3. **deployment.md** describes the current deployment and identity boundaries.
4. **Milestone documents** preserve detailed design and validation history.
5. Future work must be clearly separated from implemented behavior.
6. Historical runbooks should be labeled historical instead of silently rewritten as current production instructions.
