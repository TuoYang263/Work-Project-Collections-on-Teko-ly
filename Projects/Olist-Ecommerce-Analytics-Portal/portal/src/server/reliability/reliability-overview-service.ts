import "server-only"

import type {
  ReliabilityFinding,
  ReliabilityOverview,
} from "./reliability-overview"
import {
  fetchLatestReliabilityOverviewRows,
  type ReliabilityOverviewRow,
} from "./reliability-overview-repository"

export class ReliabilityReviewNotFoundError extends Error {
  constructor() {
    super("No persisted reliability review exists for this pipeline.")
    this.name = "ReliabilityReviewNotFoundError"
  }
}

export class ReliabilityReviewIntegrityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "ReliabilityReviewIntegrityError"
  }
}

export async function getReliabilityOverview(): Promise<ReliabilityOverview> {
  const rows = await fetchLatestReliabilityOverviewRows()
  return mapReliabilityOverview(rows)
}

export function mapReliabilityOverview(
  rows: ReliabilityOverviewRow[]
): ReliabilityOverview {
  if (rows.length === 0) {
    throw new ReliabilityReviewNotFoundError()
  }

  const first = rows[0]

  const reviewId = requiredString(first.review_id, "review_id")
  const monitoringRunId = requiredString(
    first.monitoring_run_id,
    "monitoring_run_id"
  )
  const jobName = requiredString(first.job_name, "job_name")
  const environment = requiredString(
    first.environment,
    "environment"
  )

  const total = nonNegativeInteger(
    first.total_evaluations,
    "total_evaluations"
  )
  const pass = nonNegativeInteger(
    first.pass_count,
    "pass_count"
  )
  const triggered = nonNegativeInteger(
    first.triggered_count,
    "triggered_count"
  )
  const notEvaluated = nonNegativeInteger(
    first.not_evaluated_count,
    "not_evaluated_count"
  )

  if (total !== pass + triggered + notEvaluated) {
    throw new ReliabilityReviewIntegrityError(
      "Reliability review summary counts are inconsistent."
    )
  }

  const findings: ReliabilityFinding[] = []
  const findingIds = new Set<string>()

  for (const row of rows) {
    assertSameReview(row, {
      reviewId,
      monitoringRunId,
      jobName,
      environment,
      total,
      pass,
      triggered,
      notEvaluated,
    })

    const hasEvaluation =
      row.evaluation_id !== null &&
      row.evaluation_id !== undefined

    if (!hasEvaluation) {
      continue
    }

    if (requiredString(row.result, "result") !== "TRIGGERED") {
      throw new ReliabilityReviewIntegrityError(
        "Reliability overview returned a non-triggered evaluation."
      )
    }

    const findingId = requiredString(
      row.finding_id,
      "finding_id"
    )

    if (findingIds.has(findingId)) {
      throw new ReliabilityReviewIntegrityError(
        `Duplicate finding_id in reliability review: ${findingId}`
      )
    }

    findingIds.add(findingId)

    findings.push({
      evaluationId: requiredString(
        row.evaluation_id,
        "evaluation_id"
      ),
      findingId,
      ruleId: requiredString(row.rule_id, "rule_id"),
      severity: nullableString(row.severity),
      entityType: requiredString(
        row.entity_type,
        "entity_type"
      ),
      entityId: nullableString(row.entity_id),
      evidenceSource: requiredString(
        row.evidence_source,
        "evidence_source"
      ),
      reason: requiredString(row.reason, "reason"),
    })
  }

  if (findings.length !== triggered) {
    throw new ReliabilityReviewIntegrityError(
      "Triggered count does not match persisted findings."
    )
  }

  return {
    reviewId,
    monitoringRunId,
    jobName,
    environment,
    reviewedAt: toIsoTimestamp(
      first.reviewed_at,
      "reviewed_at"
    ),
    summary: {
      total,
      pass,
      triggered,
      notEvaluated,
    },
    findings,
  }
}

function assertSameReview(
  row: ReliabilityOverviewRow,
  expected: {
    reviewId: string
    monitoringRunId: string
    jobName: string
    environment: string
    total: number
    pass: number
    triggered: number
    notEvaluated: number
  }
): void {
  if (
    requiredString(row.review_id, "review_id") !==
      expected.reviewId ||
    requiredString(
      row.monitoring_run_id,
      "monitoring_run_id"
    ) !== expected.monitoringRunId ||
    requiredString(row.job_name, "job_name") !==
      expected.jobName ||
    requiredString(row.environment, "environment") !==
      expected.environment ||
    nonNegativeInteger(
      row.total_evaluations,
      "total_evaluations"
    ) !== expected.total ||
    nonNegativeInteger(row.pass_count, "pass_count") !==
      expected.pass ||
    nonNegativeInteger(
      row.triggered_count,
      "triggered_count"
    ) !== expected.triggered ||
    nonNegativeInteger(
      row.not_evaluated_count,
      "not_evaluated_count"
    ) !== expected.notEvaluated
  ) {
    throw new ReliabilityReviewIntegrityError(
      "Rows do not belong to one consistent reliability review."
    )
  }
}

function requiredString(
  value: unknown,
  fieldName: string
): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ReliabilityReviewIntegrityError(
      `${fieldName} must be a non-empty string.`
    )
  }

  return value
}

function nullableString(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null
  }

  if (typeof value !== "string") {
    throw new ReliabilityReviewIntegrityError(
      "Expected nullable string value."
    )
  }

  return value
}

function nonNegativeInteger(
  value: unknown,
  fieldName: string
): number {
  const numberValue = Number(value)

  if (
    !Number.isSafeInteger(numberValue) ||
    numberValue < 0
  ) {
    throw new ReliabilityReviewIntegrityError(
      `${fieldName} must be a non-negative safe integer.`
    )
  }

  return numberValue
}

function toIsoTimestamp(
  value: unknown,
  fieldName: string
): string {
  let rawValue: unknown = value

  if (
    value !== null &&
    typeof value === "object" &&
    "value" in value
  ) {
    rawValue = (value as { value: unknown }).value
  }

  if (
    typeof rawValue !== "string" &&
    !(rawValue instanceof Date)
  ) {
    throw new ReliabilityReviewIntegrityError(
      `${fieldName} is not a valid timestamp.`
    )
  }

  const date =
    rawValue instanceof Date
      ? rawValue
      : new Date(rawValue)

  if (Number.isNaN(date.getTime())) {
    throw new ReliabilityReviewIntegrityError(
      `${fieldName} is not a valid timestamp.`
    )
  }

  return date.toISOString()
}
