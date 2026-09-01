import "server-only"

import { getBigQueryClient } from "@/server/bigquery/client"

const PROJECT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/
const DATASET_ID_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/

export type ReliabilityOverviewRow = {
  review_id: unknown
  monitoring_run_id: unknown
  job_name: unknown
  environment: unknown
  total_evaluations: unknown
  pass_count: unknown
  triggered_count: unknown
  not_evaluated_count: unknown
  reviewed_at: unknown

  evaluation_id: unknown
  finding_id: unknown
  rule_id: unknown
  result: unknown
  severity: unknown
  entity_type: unknown
  entity_id: unknown
  evidence_source: unknown
  reason: unknown
}

type ReliabilityRepositoryConfig = {
  projectId: string
  datasetId: string
  jobName: string
  environment: string
  location: string
}

function getConfig(): ReliabilityRepositoryConfig {
  const projectId = process.env.GCP_PROJECT_ID

  if (!projectId) {
    throw new Error("GCP_PROJECT_ID is required.")
  }

  const datasetId =
    process.env.MONITORING_DATASET_ID ?? "olist_monitoring"

  const jobName =
    process.env.RELIABILITY_JOB_NAME ??
    process.env.CONTROL_PIPELINE_NAME ??
    "olist-dbt-build-job"

  const environment =
    process.env.RELIABILITY_ENVIRONMENT ??
    process.env.CONTROL_PIPELINE_ENVIRONMENT ??
    "prod"

  const location =
    process.env.BIGQUERY_LOCATION ??
    process.env.DBT_LOCATION ??
    "EU"

  if (!PROJECT_ID_PATTERN.test(projectId)) {
    throw new Error("Invalid GCP_PROJECT_ID.")
  }

  if (!DATASET_ID_PATTERN.test(datasetId)) {
    throw new Error("Invalid MONITORING_DATASET_ID.")
  }

  return {
    projectId,
    datasetId,
    jobName,
    environment,
    location,
  }
}

export async function fetchLatestReliabilityOverviewRows(): Promise<
  ReliabilityOverviewRow[]
> {
  const config = getConfig()

  const reviewRunsTable =
    `\`${config.projectId}.${config.datasetId}.pipeline_review_runs\``

  const evaluationsTable =
    `\`${config.projectId}.${config.datasetId}.pipeline_review_evaluations\``

  const query = `
    WITH latest_review AS (
      SELECT
        review_id,
        monitoring_run_id,
        job_name,
        environment,
        total_evaluations,
        pass_count,
        triggered_count,
        not_evaluated_count,
        reviewed_at
      FROM ${reviewRunsTable}
      WHERE job_name = @job_name
        AND environment = @environment
      ORDER BY reviewed_at DESC, review_id DESC
      LIMIT 1
    )

    SELECT
      review.review_id,
      review.monitoring_run_id,
      review.job_name,
      review.environment,
      review.total_evaluations,
      review.pass_count,
      review.triggered_count,
      review.not_evaluated_count,
      review.reviewed_at,

      evaluation.evaluation_id,
      evaluation.finding_id,
      evaluation.rule_id,
      evaluation.result,
      evaluation.severity,
      evaluation.entity_type,
      evaluation.entity_id,
      evaluation.evidence_source,
      evaluation.reason

    FROM latest_review AS review

    LEFT JOIN ${evaluationsTable} AS evaluation
      ON evaluation.review_id = review.review_id
      AND evaluation.monitoring_run_id = review.monitoring_run_id
      AND evaluation.result = 'TRIGGERED'

    ORDER BY
      CASE evaluation.severity
        WHEN 'HIGH' THEN 1
        WHEN 'MEDIUM' THEN 2
        WHEN 'LOW' THEN 3
        ELSE 4
      END,
      evaluation.rule_id,
      evaluation.entity_id,
      evaluation.evaluation_id
  `

  const [rows] = await getBigQueryClient(
    config.projectId
  ).query({
    query,
    params: {
      job_name: config.jobName,
      environment: config.environment,
    },
    location: config.location,
  })

  return rows as ReliabilityOverviewRow[]
}
