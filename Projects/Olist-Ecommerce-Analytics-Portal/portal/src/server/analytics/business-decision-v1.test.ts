import { describe, expect, it } from "vitest"

import {
  evaluateStateBusinessDecisions,
  type StateBusinessMetrics,
} from "./business-decision-v1"

const states: StateBusinessMetrics[] = [
  {
    stateCode: "AA",
    gmv: 100,
    gmvGrowthRate: 0.01,
    lateDeliveryRate: 0.02,
    averageReviewScore: 4.7,
  },
  {
    stateCode: "BB",
    gmv: 200,
    gmvGrowthRate: 0.02,
    lateDeliveryRate: 0.2,
    averageReviewScore: 3.0,
  },
  {
    stateCode: "CC",
    gmv: 300,
    gmvGrowthRate: 0.03,
    lateDeliveryRate: 0.03,
    averageReviewScore: 4.6,
  },
  {
    stateCode: "DD",
    gmv: 400,
    gmvGrowthRate: 0.04,
    lateDeliveryRate: 0.04,
    averageReviewScore: 4.5,
  },
  {
    stateCode: "EE",
    gmv: 500,
    gmvGrowthRate: 0.3,
    lateDeliveryRate: 0.05,
    averageReviewScore: 4.4,
  },
  {
    stateCode: "FF",
    gmv: 600,
    gmvGrowthRate: 0.05,
    lateDeliveryRate: 0.06,
    averageReviewScore: 4.3,
  },
  {
    stateCode: "GG",
    gmv: 700,
    gmvGrowthRate: 0.06,
    lateDeliveryRate: 0.07,
    averageReviewScore: 4.2,
  },
  {
    stateCode: "HH",
    gmv: 800,
    gmvGrowthRate: 0.2,
    lateDeliveryRate: 0.25,
    averageReviewScore: 2.8,
  },
]

describe("Business Decision Model v1", () => {
  it("calculates peer-relative percentile thresholds", () => {
    const result = evaluateStateBusinessDecisions(states)

    expect(result.thresholds.highValueGmv).toBeCloseTo(625)
    expect(result.thresholds.highGrowthRate).toBeCloseTo(0.095)
    expect(
      result.thresholds.highLateDeliveryRate,
    ).toBeCloseTo(0.1025)
    expect(result.thresholds.lowReviewScore).toBeCloseTo(3.9)
  })

  it("classifies high-value states with service risk as RECOVER_SERVICE", () => {
    const result = evaluateStateBusinessDecisions(states)
    const state = result.states.find(
      (item) => item.stateCode === "HH",
    )

    expect(state?.decision).toBe("RECOVER_SERVICE")
    expect(state?.priority).toBe("P1")
    expect(state?.reasonCodes).toContain("HIGH_VALUE")
    expect(state?.reasonCodes).toContain(
      "HIGH_LATE_DELIVERY",
    )
    expect(state?.reasonCodes).toContain("LOW_REVIEW_SCORE")
  })

  it("classifies high-value healthy states as PROTECT_VALUE", () => {
    const result = evaluateStateBusinessDecisions(states)
    const state = result.states.find(
      (item) => item.stateCode === "GG",
    )

    expect(state?.decision).toBe("PROTECT_VALUE")
    expect(state?.priority).toBe("P1")
    expect(state?.reasonCodes).toContain("HIGH_VALUE")
  })

  it("classifies high-growth healthy states as EXPAND", () => {
    const result = evaluateStateBusinessDecisions(states)
    const state = result.states.find(
      (item) => item.stateCode === "EE",
    )

    expect(state?.decision).toBe("EXPAND")
    expect(state?.priority).toBe("P2")
    expect(state?.reasonCodes).toContain("HIGH_GROWTH")
  })

  it("classifies lower-value states with service risk as INVESTIGATE", () => {
    const result = evaluateStateBusinessDecisions(states)
    const state = result.states.find(
      (item) => item.stateCode === "BB",
    )

    expect(state?.decision).toBe("INVESTIGATE")
    expect(state?.priority).toBe("P2")
    expect(state?.reasonCodes).toContain(
      "HIGH_LATE_DELIVERY",
    )
  })

  it("classifies states without strong signals as MONITOR", () => {
    const result = evaluateStateBusinessDecisions(states)
    const state = result.states.find(
      (item) => item.stateCode === "CC",
    )

    expect(state?.decision).toBe("MONITOR")
    expect(state?.priority).toBe("P3")
  })

  it("does not treat missing service evidence as healthy", () => {
    const result = evaluateStateBusinessDecisions([
      ...states,
      {
        stateCode: "II",
        gmv: 900,
        gmvGrowthRate: 0.01,
        lateDeliveryRate: null,
        averageReviewScore: null,
      },
    ])

    const state = result.states.find(
      (item) => item.stateCode === "II",
    )

    expect(state?.decision).toBe("MONITOR")
    expect(state?.reasonCodes).toContain(
      "MISSING_SERVICE_EVIDENCE",
    )
    expect(state?.decision).not.toBe("PROTECT_VALUE")
    expect(state?.decision).not.toBe("EXPAND")
  })

  it("returns an empty result for an empty input", () => {
    const result = evaluateStateBusinessDecisions([])

    expect(result.states).toEqual([])
    expect(result.thresholds).toEqual({
      highValueGmv: 0,
      highGrowthRate: null,
      highLateDeliveryRate: null,
      lowReviewScore: null,
    })
  })
})