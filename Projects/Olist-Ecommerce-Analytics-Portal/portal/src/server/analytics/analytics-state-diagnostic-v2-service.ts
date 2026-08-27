import "server-only"

import {
  ANALYTICS_STATE_DIAGNOSTIC_STATES,
  type AnalyticsStateDiagnosticState,
  type AnalyticsStateDiagnosticV2,
} from "./analytics-state-diagnostic-v2"

import {
  BRAZIL_STATE_CODES,
  type BrazilStateCode,
} from "./analytics-state-summary"

import {
  fetchAnalyticsStateDiagnosticV2Rows,
  type AnalyticsStateDiagnosticV2Row,
} from "./analytics-state-diagnostic-v2-repository"

const EXPECTED_STATE_CODES =
  new Set<string>(BRAZIL_STATE_CODES)

const VALID_DIAGNOSTIC_STATES =
  new Set<string>(
    ANALYTICS_STATE_DIAGNOSTIC_STATES
  )

export class AnalyticsStateDiagnosticV2IntegrityError
  extends Error {
  constructor(message: string) {
    super(message)
    this.name =
      "AnalyticsStateDiagnosticV2IntegrityError"
  }
}

export async function getAnalyticsStateDiagnosticsV2(): Promise<
  AnalyticsStateDiagnosticV2[]
> {
  const rows =
    await fetchAnalyticsStateDiagnosticV2Rows()

  return mapAnalyticsStateDiagnosticsV2(rows)
}

export function mapAnalyticsStateDiagnosticsV2(
  rows: AnalyticsStateDiagnosticV2Row[]
): AnalyticsStateDiagnosticV2[] {
  if (
    rows.length !==
    BRAZIL_STATE_CODES.length
  ) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `Expected ${BRAZIL_STATE_CODES.length} Brazil states, received ${rows.length}.`
    )
  }

  const seenStateCodes = new Set<string>()

  const diagnostics = rows.map((row) => {
    const stateCode =
      parseStateCode(row.state_code)

    if (seenStateCodes.has(stateCode)) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Duplicate state_code: ${stateCode}`
      )
    }

    seenStateCodes.add(stateCode)

    const ciLowerPp =
      finiteNumber(
        row.ci_lower_pp,
        "ci_lower_pp"
      )

    const ciUpperPp =
      finiteNumber(
        row.ci_upper_pp,
        "ci_upper_pp"
      )

    if (ciLowerPp > ciUpperPp) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Invalid confidence interval for ${stateCode}.`
      )
    }

    return {
      stateCode,

      evidenceCount:
        nonNegativeInteger(
          row.evidence_count,
          "evidence_count"
        ),

      actualNegativeReviewRate:
        probability(
          row.actual_negative_review_rate,
          "actual_negative_review_rate"
        ),

      expectedNegativeReviewRate:
        probability(
          row.expected_negative_review_rate,
          "expected_negative_review_rate"
        ),

      residualPp:
        finiteNumber(
          row.residual_pp,
          "residual_pp"
        ),

      ciLowerPp,
      ciUpperPp,

      zScore:
        finiteNumber(
          row.z_score,
          "z_score"
        ),

      diagnosticState:
        parseDiagnosticState(
          row.diagnostic_state
        ),

      modelVersion:
        nonEmptyString(
          row.model_version,
          "model_version"
        ),

      generatedAt:
        timestampString(
          row.generated_at,
          "generated_at"
        ),
    }
  })

  for (const stateCode of BRAZIL_STATE_CODES) {
    if (!seenStateCodes.has(stateCode)) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Missing Brazil state: ${stateCode}`
      )
    }
  }

  return diagnostics
}

function parseStateCode(
  value: unknown
): BrazilStateCode {
  if (
    typeof value !== "string" ||
    !EXPECTED_STATE_CODES.has(value)
  ) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `Invalid Brazil state_code: ${String(value)}`
    )
  }

  return value as BrazilStateCode
}

function parseDiagnosticState(
  value: unknown
): AnalyticsStateDiagnosticState {
  if (
    typeof value !== "string" ||
    !VALID_DIAGNOSTIC_STATES.has(value)
  ) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `Invalid diagnostic_state: ${String(value)}`
    )
  }

  return value as AnalyticsStateDiagnosticState
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
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `${fieldName} must be a non-negative safe integer.`
    )
  }

  return numberValue
}

function probability(
  value: unknown,
  fieldName: string
): number {
  const numberValue =
    finiteNumber(value, fieldName)

  if (
    numberValue < 0 ||
    numberValue > 1
  ) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `${fieldName} must be between 0 and 1.`
    )
  }

  return numberValue
}

function finiteNumber(
  value: unknown,
  fieldName: string
): number {
  const numberValue = Number(value)

  if (!Number.isFinite(numberValue)) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `${fieldName} must be a finite number.`
    )
  }

  return numberValue
}

function nonEmptyString(
  value: unknown,
  fieldName: string
): string {
  if (
    typeof value !== "string" ||
    value.trim().length === 0
  ) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `${fieldName} must be a non-empty string.`
    )
  }

  return value
}

function timestampString(
  value: unknown,
  fieldName: string
): string {
  let rawValue: unknown = value

  if (
    value !== null &&
    typeof value === "object" &&
    "value" in value
  ) {
    rawValue =
      (value as { value: unknown }).value
  }

  if (
    typeof rawValue !== "string" &&
    !(rawValue instanceof Date)
  ) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `${fieldName} is not a valid timestamp.`
    )
  }

  const date = new Date(rawValue)

  if (Number.isNaN(date.getTime())) {
    throw new AnalyticsStateDiagnosticV2IntegrityError(
      `${fieldName} is not a valid timestamp.`
    )
  }

  return date.toISOString()
}