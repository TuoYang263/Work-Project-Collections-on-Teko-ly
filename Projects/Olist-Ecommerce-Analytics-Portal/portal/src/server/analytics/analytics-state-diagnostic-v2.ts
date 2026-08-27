import type { BrazilStateCode } from "./analytics-state-summary"

export const ANALYTICS_STATE_DIAGNOSTIC_STATES = [
  "WORSE_THAN_EXPECTED",
  "BETTER_THAN_EXPECTED",
  "AS_EXPECTED",
  "INSUFFICIENT_EVIDENCE",
] as const

export type AnalyticsStateDiagnosticState =
  (typeof ANALYTICS_STATE_DIAGNOSTIC_STATES)[number]

export type AnalyticsStateDiagnosticV2 = {
  stateCode: BrazilStateCode
  evidenceCount: number

  actualNegativeReviewRate: number
  expectedNegativeReviewRate: number

  residualPp: number
  ciLowerPp: number
  ciUpperPp: number
  zScore: number

  diagnosticState: AnalyticsStateDiagnosticState

  modelVersion: string
  generatedAt: string
}