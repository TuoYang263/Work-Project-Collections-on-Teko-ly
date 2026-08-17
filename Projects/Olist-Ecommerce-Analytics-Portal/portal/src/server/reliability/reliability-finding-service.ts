import "server-only"

import type { ReliabilityFindingDetail } from "./reliability-finding"
import {
  fetchReliabilityFindingRows,
  type ReliabilityFindingRow,
} from "./reliability-finding-repository"

export class ReliabilityFindingNotFoundError extends Error {
  constructor() {
    super("The requested reliability finding does not exist.")
    this.name = "ReliabilityFindingNotFoundError"
  }
}

export class ReliabilityFindingIntegrityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "ReliabilityFindingIntegrityError"
  }
}

export async function getReliabilityFinding(
  findingId: string
): Promise<ReliabilityFindingDetail> {
  if (!findingId.trim()) {
    throw new ReliabilityFindingNotFoundError()
  }

  const rows = await fetchReliabilityFindingRows(findingId)

  return mapReliabilityFinding(findingId, rows)
}

export function mapReliabilityFinding(
  expectedFindingId: string,
  rows: ReliabilityFindingRow[]
): ReliabilityFindingDetail {
  if (rows.length === 0) {
    throw new ReliabilityFindingNotFoundError()
  }

  if (rows.length !== 1) {
    throw new ReliabilityFindingIntegrityError(
      "The persisted review contains duplicate finding rows."
    )
  }

  const row = rows[0]

  const findingId = requiredString(
    row.finding_id,
    "finding_id"
  )

  if (findingId !== expectedFindingId) {
    throw new ReliabilityFindingIntegrityError(
      "The persisted finding identity does not match the request."
    )
  }

  const result = requiredString(
    row.result,
    "result"
  )

  if (result !== "TRIGGERED") {
    throw new ReliabilityFindingIntegrityError(
      "A finding detail must represent a triggered evaluation."
    )
  }

  return {
    evaluationId: requiredString(
      row.evaluation_id,
      "evaluation_id"
    ),
    findingId,

    reviewId: requiredString(
      row.review_id,
      "review_id"
    ),
    monitoringRunId: requiredString(
      row.monitoring_run_id,
      "monitoring_run_id"
    ),
    jobName: requiredString(
      row.job_name,
      "job_name"
    ),
    environment: requiredString(
      row.environment,
      "environment"
    ),
    reviewedAt: toIsoTimestamp(
      row.reviewed_at,
      "reviewed_at"
    ),

    ruleId: requiredString(
      row.rule_id,
      "rule_id"
    ),
    result: "TRIGGERED",
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
    evidence: jsonObject(
      row.evidence_json,
      "evidence_json"
    ),
    reason: requiredString(
      row.reason,
      "reason"
    ),
  }
}

function requiredString(
  value: unknown,
  fieldName: string
): string {
  if (
    typeof value !== "string" ||
    value.trim() === ""
  ) {
    throw new ReliabilityFindingIntegrityError(
      `${fieldName} must be a non-empty string.`
    )
  }

  return value
}

function nullableString(
  value: unknown
): string | null {
  if (
    value === null ||
    value === undefined
  ) {
    return null
  }

  if (typeof value !== "string") {
    throw new ReliabilityFindingIntegrityError(
      "Expected nullable string value."
    )
  }

  return value
}

function jsonObject(
  value: unknown,
  fieldName: string
): Record<string, unknown> {
  let parsed: unknown = value

  if (typeof value === "string") {
    try {
      parsed = JSON.parse(value)
    } catch {
      throw new ReliabilityFindingIntegrityError(
        `${fieldName} is not valid JSON.`
      )
    }
  }

  if (
    parsed === null ||
    typeof parsed !== "object" ||
    Array.isArray(parsed)
  ) {
    throw new ReliabilityFindingIntegrityError(
      `${fieldName} must be a JSON object.`
    )
  }

  return parsed as Record<string, unknown>
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
    rawValue = (
      value as { value: unknown }
    ).value
  }

  if (
    typeof rawValue !== "string" &&
    !(rawValue instanceof Date)
  ) {
    throw new ReliabilityFindingIntegrityError(
      `${fieldName} is not a valid timestamp.`
    )
  }

  const date =
    rawValue instanceof Date
      ? rawValue
      : new Date(rawValue)

  if (Number.isNaN(date.getTime())) {
    throw new ReliabilityFindingIntegrityError(
      `${fieldName} is not a valid timestamp.`
    )
  }

  return date.toISOString()
}
