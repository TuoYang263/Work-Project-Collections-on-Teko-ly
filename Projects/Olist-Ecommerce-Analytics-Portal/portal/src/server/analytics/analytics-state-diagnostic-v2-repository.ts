import "server-only"

import { getBigQueryClient } from "@/server/bigquery/client"

const PROJECT_ID_PATTERN =
  /^[A-Za-z0-9][A-Za-z0-9_-]*$/

const DATASET_ID_PATTERN =
  /^[A-Za-z_][A-Za-z0-9_]*$/

export type AnalyticsStateDiagnosticV2Row = {
  state_code: unknown
  evidence_count: unknown
  actual_negative_review_rate: unknown
  expected_negative_review_rate: unknown
  residual_pp: unknown
  ci_lower_pp: unknown
  ci_upper_pp: unknown
  z_score: unknown
  diagnostic_state: unknown
  model_version: unknown
  generated_at: unknown
}

type AnalyticsStateDiagnosticRepositoryConfig = {
  projectId: string
  datasetId: string
  location: string
}

function getConfig(): AnalyticsStateDiagnosticRepositoryConfig {
  const projectId = process.env.GCP_PROJECT_ID

  if (!projectId) {
    throw new Error("GCP_PROJECT_ID is required.")
  }

  const datasetId =
    process.env.ANALYTICS_DATASET_ID ??
    "olist_analytics"

  const location =
    process.env.BIGQUERY_LOCATION ??
    process.env.DBT_LOCATION ??
    "EU"

  if (!PROJECT_ID_PATTERN.test(projectId)) {
    throw new Error("Invalid GCP_PROJECT_ID.")
  }

  if (!DATASET_ID_PATTERN.test(datasetId)) {
    throw new Error(
      "Invalid ANALYTICS_DATASET_ID."
    )
  }

  return {
    projectId,
    datasetId,
    location,
  }
}

export async function fetchAnalyticsStateDiagnosticV2Rows(): Promise<
  AnalyticsStateDiagnosticV2Row[]
> {
  const config = getConfig()

  const table =
    `\`${config.projectId}.${config.datasetId}.analytics_state_diagnostics_v2\``

  const query = `
    SELECT
      state_code,
      evidence_count,
      actual_negative_review_rate,
      expected_negative_review_rate,
      residual_pp,
      ci_lower_pp,
      ci_upper_pp,
      z_score,
      diagnostic_state,
      model_version,
      generated_at
    FROM ${table}
    ORDER BY state_code
  `

  const [rows] = await getBigQueryClient(
    config.projectId
  ).query({
    query,
    location: config.location,
  })

  return rows as AnalyticsStateDiagnosticV2Row[]
}