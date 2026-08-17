import "server-only"

import { getBigQueryClient } from "@/server/bigquery/client"

const PROJECT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/
const DATASET_ID_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/

export type ReliabilityFindingRow = {
  evaluation_id: unknown
  finding_id: unknown

  review_id: unknown
  monitoring_run_id: unknown
  job_name: unknown
  environment: unknown
  reviewed_at: unknown

  rule_id: unknown
  result: unknown
  severity: unknown

  entity_type: unknown
  entity_id: unknown

  evidence_source: unknown
  evidence_json: unknown
  reason: unknown
}

type ReliabilityFindingRepositoryConfig = {
  projectId: string
  datasetId: string
  location: string
}

function getConfig(): ReliabilityFindingRepositoryConfig {
  const projectId = process.env.GCP_PROJECT_ID

  if (!projectId) {
    throw new Error("GCP_PROJECT_ID is required.")
  }

  const datasetId =
    process.env.MONITORING_DATASET_ID ?? "olist_monitoring"

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
    location,
  }
}

export async function fetchReliabilityFindingRows(
  findingId: string
): Promise<ReliabilityFindingRow[]> {
  const config = getConfig()

  const reviewRunsTable =
    `\`${config.projectId}.${config.datasetId}.pipeline_review_runs\``

  const evaluationsTable =
    `\`${config.projectId}.${config.datasetId}.pipeline_review_evaluations\``

  const query = `
    WITH latest_match AS (
      SELECT
        review.review_id,
        review.monitoring_run_id
      FROM ${evaluationsTable} AS evaluation
      JOIN ${reviewRunsTable} AS review
        ON review.review_id = evaluation.review_id
        AND review.monitoring_run_id = evaluation.monitoring_run_id
      WHERE evaluation.finding_id = @finding_id
        AND evaluation.result = 'TRIGGERED'
      ORDER BY
        review.reviewed_at DESC,
        review.review_id DESC
      LIMIT 1
    )

    SELECT
      evaluation.evaluation_id,
      evaluation.finding_id,

      review.review_id,
      review.monitoring_run_id,
      review.job_name,
      review.environment,
      review.reviewed_at,

      evaluation.rule_id,
      evaluation.result,
      evaluation.severity,

      evaluation.entity_type,
      evaluation.entity_id,

      evaluation.evidence_source,
      evaluation.evidence_json,
      evaluation.reason

    FROM latest_match AS latest

    JOIN ${reviewRunsTable} AS review
      ON review.review_id = latest.review_id
      AND review.monitoring_run_id = latest.monitoring_run_id

    JOIN ${evaluationsTable} AS evaluation
      ON evaluation.review_id = latest.review_id
      AND evaluation.monitoring_run_id = latest.monitoring_run_id

    WHERE evaluation.finding_id = @finding_id
      AND evaluation.result = 'TRIGGERED'

    LIMIT 2
  `

  const [rows] = await getBigQueryClient(
    config.projectId
  ).query({
    query,
    params: {
      finding_id: findingId,
    },
    location: config.location,
  })

  return rows as ReliabilityFindingRow[]
}
