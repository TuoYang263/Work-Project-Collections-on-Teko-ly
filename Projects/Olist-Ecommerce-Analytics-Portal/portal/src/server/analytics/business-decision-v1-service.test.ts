import { describe, expect, it } from "vitest"

import type { AnalyticsStateSummary } from "./analytics-state-summary"
import { buildBusinessDecisionModelV1 } from "./business-decision-v1-service"

const states: AnalyticsStateSummary[] = [
  {
    stateCode: "AC",
    orderCount: 10,
    gmv: 100,
    aov: 10,
    deliveryObservationCount: 9,
    lateDeliveryRate: 0.02,
    reviewedOrderCount: 8,
    averageReviewScore: 4.5,
  },
  {
    stateCode: "AL",
    orderCount: 20,
    gmv: 200,
    aov: 10,
    deliveryObservationCount: 18,
    lateDeliveryRate: 0.03,
    reviewedOrderCount: 17,
    averageReviewScore: 4.4,
  },
  {
    stateCode: "AP",
    orderCount: 30,
    gmv: 300,
    aov: 10,
    deliveryObservationCount: 28,
    lateDeliveryRate: 0.04,
    reviewedOrderCount: 25,
    averageReviewScore: 4.3,
  },
  {
    stateCode: "AM",
    orderCount: 40,
    gmv: 400,
    aov: 10,
    deliveryObservationCount: 37,
    lateDeliveryRate: 0.05,
    reviewedOrderCount: 35,
    averageReviewScore: 4.2,
  },
  {
    stateCode: "BA",
    orderCount: 50,
    gmv: 500,
    aov: 10,
    deliveryObservationCount: 45,
    lateDeliveryRate: 0.06,
    reviewedOrderCount: 42,
    averageReviewScore: 4.1,
  },
  {
    stateCode: "CE",
    orderCount: 60,
    gmv: 600,
    aov: 10,
    deliveryObservationCount: 55,
    lateDeliveryRate: 0.07,
    reviewedOrderCount: 50,
    averageReviewScore: 4.0,
  },
  {
    stateCode: "DF",
    orderCount: 70,
    gmv: 700,
    aov: 10,
    deliveryObservationCount: 65,
    lateDeliveryRate: 0.08,
    reviewedOrderCount: 60,
    averageReviewScore: 3.9,
  },
  {
    stateCode: "ES",
    orderCount: 80,
    gmv: 800,
    aov: 10,
    deliveryObservationCount: 70,
    lateDeliveryRate: 0.25,
    reviewedOrderCount: 65,
    averageReviewScore: 3.0,
  },
]

describe("Business Decision Model v1 service", () => {
  it("builds decisions from governed analytics state summaries", () => {
    const result =
      buildBusinessDecisionModelV1(states)

    expect(result.states).toHaveLength(8)

    const riskyHighValueState =
      result.states.find(
        (state) => state.stateCode === "ES"
      )

    expect(
      riskyHighValueState?.decision
    ).toBe("RECOVER_SERVICE")

    expect(
      riskyHighValueState?.priority
    ).toBe("P1")
  })

  it("does not invent growth evidence", () => {
    const result =
      buildBusinessDecisionModelV1(states)

    expect(
      result.thresholds.highGrowthRate
    ).toBeNull()

    for (const state of result.states) {
      expect(
        state.evidence.gmvGrowthRate
      ).toBeNull()

      expect(
        state.reasonCodes
      ).not.toContain("HIGH_GROWTH")
    }
  })
})