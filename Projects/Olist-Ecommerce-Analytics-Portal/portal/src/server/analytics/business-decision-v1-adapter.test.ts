import { describe, expect, it } from "vitest"

import type { AnalyticsStateSummary } from "./analytics-state-summary"
import { mapStateSummariesToBusinessMetrics } from "./business-decision-v1-adapter"

describe("Business Decision Model v1 adapter", () => {
  it("maps governed state analytics evidence into decision metrics", () => {
    const states: AnalyticsStateSummary[] = [
      {
        stateCode: "SP",
        orderCount: 100,
        gmv: 25000,
        aov: 250,
        deliveryObservationCount: 90,
        lateDeliveryRate: 0.12,
        reviewedOrderCount: 80,
        averageReviewScore: 3.9,
      },
    ]

    expect(
      mapStateSummariesToBusinessMetrics(states)
    ).toEqual([
      {
        stateCode: "SP",
        gmv: 25000,
        gmvGrowthRate: null,
        lateDeliveryRate: 0.12,
        averageReviewScore: 3.9,
      },
    ])
  })

  it("preserves missing service evidence as null", () => {
    const states: AnalyticsStateSummary[] = [
      {
        stateCode: "AC",
        orderCount: 10,
        gmv: 1000,
        aov: 100,
        deliveryObservationCount: 0,
        lateDeliveryRate: null,
        reviewedOrderCount: 0,
        averageReviewScore: null,
      },
    ]

    const [result] =
      mapStateSummariesToBusinessMetrics(states)

    expect(result.lateDeliveryRate).toBeNull()
    expect(result.averageReviewScore).toBeNull()
    expect(result.gmvGrowthRate).toBeNull()
  })
})