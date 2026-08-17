import "server-only"

import {
  BRAZIL_STATE_CODES,
  type AnalyticsStateSummary,
  type BrazilStateCode,
} from "./analytics-state-summary"

import {
  fetchAnalyticsStateSummaryRows,
  type AnalyticsStateSummaryRow,
} from "./analytics-state-summary-repository"

const EXPECTED_STATE_CODES =
  new Set<string>(BRAZIL_STATE_CODES)

export class AnalyticsStateSummaryIntegrityError extends Error {
  constructor(message: string) {
    super(message)
    this.name =
      "AnalyticsStateSummaryIntegrityError"
  }
}

export async function getAnalyticsStateSummaries(): Promise<
  AnalyticsStateSummary[]
> {
  const rows =
    await fetchAnalyticsStateSummaryRows()

  return mapAnalyticsStateSummaries(rows)
}

export function mapAnalyticsStateSummaries(
  rows: AnalyticsStateSummaryRow[]
): AnalyticsStateSummary[] {
  if (
    rows.length !==
    BRAZIL_STATE_CODES.length
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `Expected ${BRAZIL_STATE_CODES.length} Brazil states, received ${rows.length}.`
    )
  }

  const seenStateCodes = new Set<string>()

  const states = rows.map((row) => {
    const stateCode =
      parseStateCode(row.state_code)

    if (seenStateCodes.has(stateCode)) {
      throw new AnalyticsStateSummaryIntegrityError(
        `Duplicate state_code: ${stateCode}`
      )
    }

    seenStateCodes.add(stateCode)

    return {
      stateCode,
      orderCount: nonNegativeInteger(
        row.order_count,
        "order_count"
      ),
      gmv: nonNegativeNumber(
        row.gmv,
        "gmv"
      ),
      aov: nonNegativeNumber(
        row.aov,
        "aov"
      ),
    }
  })

  for (const stateCode of BRAZIL_STATE_CODES) {
    if (!seenStateCodes.has(stateCode)) {
      throw new AnalyticsStateSummaryIntegrityError(
        `Missing Brazil state: ${stateCode}`
      )
    }
  }

  return states
}

function parseStateCode(
  value: unknown
): BrazilStateCode {
  if (
    typeof value !== "string" ||
    !EXPECTED_STATE_CODES.has(value)
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `Invalid Brazil state_code: ${String(value)}`
    )
  }

  return value as BrazilStateCode
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
    throw new AnalyticsStateSummaryIntegrityError(
      `${fieldName} must be a non-negative safe integer.`
    )
  }

  return numberValue
}

function nonNegativeNumber(
  value: unknown,
  fieldName: string
): number {
  const numberValue = Number(value)

  if (
    !Number.isFinite(numberValue) ||
    numberValue < 0
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `${fieldName} must be a non-negative number.`
    )
  }

  return numberValue
}
