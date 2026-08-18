import type { AnalyticsStateSummary } from "./analytics-state-summary"
import {
  type BusinessDecisionModelResult,
  evaluateStateBusinessDecisions,
} from "./business-decision-v1"
import { mapStateSummariesToBusinessMetrics } from "./business-decision-v1-adapter"

export function buildBusinessDecisionModelV1(
  states: AnalyticsStateSummary[]
): BusinessDecisionModelResult {
  const metrics =
    mapStateSummariesToBusinessMetrics(states)

  return evaluateStateBusinessDecisions(metrics)
}