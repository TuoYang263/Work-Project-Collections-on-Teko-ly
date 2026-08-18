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

    const orderCount = nonNegativeInteger(
      row.order_count,
      "order_count"
    )

    const deliveryObservationCount =
      nonNegativeInteger(
        row.delivery_observation_count,
        "delivery_observation_count"
      )

    const lateDeliveryRate =
      nullableBoundedNumber(
        row.late_delivery_rate,
        "late_delivery_rate",
        0,
        1
      )

    const reviewedOrderCount =
      nonNegativeInteger(
        row.reviewed_order_count,
        "reviewed_order_count"
      )

    const averageReviewScore =
      nullableBoundedNumber(
        row.average_review_score,
        "average_review_score",
        1,
        5
      )

    validateObservationEvidence({
      stateCode,
      orderCount,
      deliveryObservationCount,
      lateDeliveryRate,
      reviewedOrderCount,
      averageReviewScore,
    })

    return {
      stateCode,
      orderCount,
      gmv: nonNegativeNumber(
        row.gmv,
        "gmv"
      ),
      aov: nonNegativeNumber(
        row.aov,
        "aov"
      ),
      deliveryObservationCount,
      lateDeliveryRate,
      reviewedOrderCount,
      averageReviewScore,
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

function validateObservationEvidence({
  stateCode,
  orderCount,
  deliveryObservationCount,
  lateDeliveryRate,
  reviewedOrderCount,
  averageReviewScore,
}: {
  stateCode: BrazilStateCode
  orderCount: number
  deliveryObservationCount: number
  lateDeliveryRate: number | null
  reviewedOrderCount: number
  averageReviewScore: number | null
}): void {
  if (deliveryObservationCount > orderCount) {
    throw new AnalyticsStateSummaryIntegrityError(
      `delivery_observation_count cannot exceed order_count for ${stateCode}.`
    )
  }

  if (reviewedOrderCount > orderCount) {
    throw new AnalyticsStateSummaryIntegrityError(
      `reviewed_order_count cannot exceed order_count for ${stateCode}.`
    )
  }

  if (
    deliveryObservationCount === 0 &&
    lateDeliveryRate !== null
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `late_delivery_rate must be null when delivery_observation_count is zero for ${stateCode}.`
    )
  }

  if (
    deliveryObservationCount > 0 &&
    lateDeliveryRate === null
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `late_delivery_rate is required when delivery_observation_count is positive for ${stateCode}.`
    )
  }

  if (
    reviewedOrderCount === 0 &&
    averageReviewScore !== null
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `average_review_score must be null when reviewed_order_count is zero for ${stateCode}.`
    )
  }

  if (
    reviewedOrderCount > 0 &&
    averageReviewScore === null
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `average_review_score is required when reviewed_order_count is positive for ${stateCode}.`
    )
  }
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

function nullableBoundedNumber(
  value: unknown,
  fieldName: string,
  min: number,
  max: number
): number | null {
  if (value === null || value === undefined) {
    return null
  }

  const numberValue = Number(value)

  if (
    !Number.isFinite(numberValue) ||
    numberValue < min ||
    numberValue > max
  ) {
    throw new AnalyticsStateSummaryIntegrityError(
      `${fieldName} must be null or a number between ${min} and ${max}.`
    )
  }

  return numberValue
}
