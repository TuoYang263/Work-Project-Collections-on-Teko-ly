import "server-only"

import type { AnalyticsSummary } from "./analytics-summary"
import {
  fetchAnalyticsSummaryRows,
  type AnalyticsSummaryRow,
} from "./analytics-summary-repository"

export class AnalyticsSummaryNotFoundError extends Error {
  constructor() {
    super("No analytics summary is available.")
    this.name = "AnalyticsSummaryNotFoundError"
  }
}

export class AnalyticsSummaryIntegrityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "AnalyticsSummaryIntegrityError"
  }
}

export async function getAnalyticsSummary(): Promise<AnalyticsSummary> {
  const rows = await fetchAnalyticsSummaryRows()
  return mapAnalyticsSummary(rows)
}

export function mapAnalyticsSummary(
  rows: AnalyticsSummaryRow[]
): AnalyticsSummary {
  if (rows.length === 0) {
    throw new AnalyticsSummaryNotFoundError()
  }

  if (rows.length !== 1) {
    throw new AnalyticsSummaryIntegrityError(
      "Analytics summary must contain exactly one row."
    )
  }

  const row = rows[0]

  return {
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
    firstOrderDate: toDateString(
      row.first_order_date,
      "first_order_date"
    ),
    lastOrderDate: toDateString(
      row.last_order_date,
      "last_order_date"
    ),
  }
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
    throw new AnalyticsSummaryIntegrityError(
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
    throw new AnalyticsSummaryIntegrityError(
      `${fieldName} must be a non-negative number.`
    )
  }

  return numberValue
}

function toDateString(
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

  if (typeof rawValue !== "string") {
    throw new AnalyticsSummaryIntegrityError(
      `${fieldName} is not a valid date.`
    )
  }

  const date = new Date(`${rawValue}T00:00:00Z`)

  if (Number.isNaN(date.getTime())) {
    throw new AnalyticsSummaryIntegrityError(
      `${fieldName} is not a valid date.`
    )
  }

  return rawValue
}
