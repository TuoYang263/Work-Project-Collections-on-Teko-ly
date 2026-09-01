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

const RESIDUAL_TOLERANCE_PP = 0.001

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

    const evidenceCount =
      nonNegativeInteger(
        row.evidence_count,
        "evidence_count"
      )

    const actualNegativeReviewRate =
      probability(
        row.actual_negative_review_rate,
        "actual_negative_review_rate"
      )

    const expectedNegativeReviewRate =
      probability(
        row.expected_negative_review_rate,
        "expected_negative_review_rate"
      )

    const residualPp =
      finiteNumber(
        row.residual_pp,
        "residual_pp"
      )

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

    const expectedResidualPp =
      (
        actualNegativeReviewRate -
        expectedNegativeReviewRate
      ) * 100

    if (
      Math.abs(
        residualPp - expectedResidualPp
      ) > RESIDUAL_TOLERANCE_PP
    ) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Residual mismatch for ${stateCode}.`
      )
    }

    const diagnosticState =
      parseDiagnosticState(
        row.diagnostic_state
      )

    const expectedDiagnosticState =
      classifyDiagnosticState(
        evidenceCount,
        residualPp,
        ciLowerPp,
        ciUpperPp
      )

    if (
      diagnosticState !==
      expectedDiagnosticState
    ) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Diagnostic state mismatch for ${stateCode}: expected ${expectedDiagnosticState}, received ${diagnosticState}.`
      )
    }

    return {
      stateCode,
      evidenceCount,
      actualNegativeReviewRate,
      expectedNegativeReviewRate,
      residualPp,
      ciLowerPp,
      ciUpperPp,

      zScore:
        finiteNumber(
          row.z_score,
          "z_score"
        ),

      diagnosticState,

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

  const expectedModelVersion =
    diagnostics[0].modelVersion

  const expectedGeneratedAt =
    diagnostics[0].generatedAt

  for (const diagnostic of diagnostics) {
    if (
      diagnostic.modelVersion !==
      expectedModelVersion
    ) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Mixed model_version values detected: expected ${expectedModelVersion}, received ${diagnostic.modelVersion} for ${diagnostic.stateCode}.`
      )
    }

    if (
      diagnostic.generatedAt !==
      expectedGeneratedAt
    ) {
      throw new AnalyticsStateDiagnosticV2IntegrityError(
        `Mixed generated_at values detected: expected ${expectedGeneratedAt}, received ${diagnostic.generatedAt} for ${diagnostic.stateCode}.`
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

function classifyDiagnosticState(
  evidenceCount: number,
  residualPp: number,
  ciLowerPp: number,
  ciUpperPp: number
): AnalyticsStateDiagnosticState {
  if (evidenceCount < 100) {
    return "INSUFFICIENT_EVIDENCE"
  }

  if (
    residualPp >= 1 &&
    ciLowerPp > 0
  ) {
    return "WORSE_THAN_EXPECTED"
  }

  if (
    residualPp <= -1 &&
    ciUpperPp < 0
  ) {
    return "BETTER_THAN_EXPECTED"
  }

  return "AS_EXPECTED"
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