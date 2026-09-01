import type { AnalyticsStateSummary } from "./analytics-state-summary"
import type { StateBusinessMetrics } from "./business-decision-v1"

export function mapStateSummariesToBusinessMetrics(
  states: AnalyticsStateSummary[]
): StateBusinessMetrics[] {
  return states.map((state) => ({
    stateCode: state.stateCode,
    gmv: state.gmv,

    // v1 serving layer is currently a full-history snapshot.
    // Growth remains unavailable until a governed
    // current-vs-previous period contract exists.
    gmvGrowthRate: null,

    lateDeliveryRate:
      state.lateDeliveryRate,

    averageReviewScore:
      state.averageReviewScore,
  }))
}