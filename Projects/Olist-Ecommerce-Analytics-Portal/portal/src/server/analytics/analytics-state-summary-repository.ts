import "server-only"

import { getBigQueryClient } from "@/server/bigquery/client"

const PROJECT_ID_PATTERN =
  /^[A-Za-z0-9][A-Za-z0-9_-]*$/

const DATASET_ID_PATTERN =
  /^[A-Za-z_][A-Za-z0-9_]*$/

export type AnalyticsStateSummaryRow = {
  state_code: unknown
  order_count: unknown
  gmv: unknown
  aov: unknown
}

type AnalyticsStateRepositoryConfig = {
  projectId: string
  datasetId: string
  location: string
}

function getConfig(): AnalyticsStateRepositoryConfig {
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

export async function fetchAnalyticsStateSummaryRows(): Promise<
  AnalyticsStateSummaryRow[]
> {
  const config = getConfig()

  const table =
    `\`${config.projectId}.${config.datasetId}.analytics_state_summary\``

  const query = `
    SELECT
      state_code,
      order_count,
      gmv,
      aov
    FROM ${table}
    ORDER BY state_code
  `

  const [rows] = await getBigQueryClient(
    config.projectId
  ).query({
    query,
    location: config.location,
  })

  return rows as AnalyticsStateSummaryRow[]
}
