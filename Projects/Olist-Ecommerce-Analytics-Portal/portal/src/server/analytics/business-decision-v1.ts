export type BusinessDecision =
  | "RECOVER_SERVICE"
  | "PROTECT_VALUE"
  | "EXPAND"
  | "INVESTIGATE"
  | "MONITOR"

export type BusinessPriority = "P1" | "P2" | "P3"

export type BusinessReasonCode =
  | "HIGH_VALUE"
  | "HIGH_GROWTH"
  | "HIGH_LATE_DELIVERY"
  | "LOW_REVIEW_SCORE"
  | "MISSING_SERVICE_EVIDENCE"

export type StateBusinessMetrics = {
  stateCode: string
  gmv: number
  gmvGrowthRate: number | null
  lateDeliveryRate: number | null
  averageReviewScore: number | null
}

export type BusinessDecisionThresholds = {
  highValueGmv: number
  highGrowthRate: number | null
  highLateDeliveryRate: number | null
  lowReviewScore: number | null
}

export type StateBusinessDecision = {
  stateCode: string
  decision: BusinessDecision
  priority: BusinessPriority
  reasonCodes: BusinessReasonCode[]
  evidence: {
    gmv: number
    gmvGrowthRate: number | null
    lateDeliveryRate: number | null
    averageReviewScore: number | null
  }
}

export type BusinessDecisionModelResult = {
  thresholds: BusinessDecisionThresholds
  states: StateBusinessDecision[]
}

function percentile(values: number[], p: number): number | null {
  if (values.length === 0) {
    return null
  }

  const sorted = [...values].sort((a, b) => a - b)

  if (sorted.length === 1) {
    return sorted[0]
  }

  const position = (sorted.length - 1) * p
  const lowerIndex = Math.floor(position)
  const upperIndex = Math.ceil(position)

  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex]
  }

  const weight = position - lowerIndex

  return (
    sorted[lowerIndex] +
    (sorted[upperIndex] - sorted[lowerIndex]) * weight
  )
}

function usableNumbers(
  values: Array<number | null>,
): number[] {
  return values.filter(
    (value): value is number =>
      value !== null && Number.isFinite(value),
  )
}

export function calculateBusinessDecisionThresholds(
  states: StateBusinessMetrics[],
): BusinessDecisionThresholds {
  const gmvValues = usableNumbers(
    states.map((state) => state.gmv),
  )

  if (gmvValues.length === 0) {
    throw new Error(
      "Business Decision Model v1 requires at least one valid GMV value.",
    )
  }

  const highValueGmv = percentile(gmvValues, 0.75)

  if (highValueGmv === null) {
    throw new Error(
      "Business Decision Model v1 could not calculate the GMV threshold.",
    )
  }

  return {
    highValueGmv,
    highGrowthRate: percentile(
      usableNumbers(
        states.map((state) => state.gmvGrowthRate),
      ),
      0.75,
    ),
    highLateDeliveryRate: percentile(
      usableNumbers(
        states.map((state) => state.lateDeliveryRate),
      ),
      0.75,
    ),
    lowReviewScore: percentile(
      usableNumbers(
        states.map((state) => state.averageReviewScore),
      ),
      0.25,
    ),
  }
}

function evaluateState(
  state: StateBusinessMetrics,
  thresholds: BusinessDecisionThresholds,
): StateBusinessDecision {
  const highValue = state.gmv >= thresholds.highValueGmv

  const highGrowth =
    state.gmvGrowthRate !== null &&
    thresholds.highGrowthRate !== null &&
    state.gmvGrowthRate >= thresholds.highGrowthRate

  const highLateDelivery =
    state.lateDeliveryRate !== null &&
    thresholds.highLateDeliveryRate !== null &&
    state.lateDeliveryRate >=
      thresholds.highLateDeliveryRate

  const lowReviewScore =
    state.averageReviewScore !== null &&
    thresholds.lowReviewScore !== null &&
    state.averageReviewScore <= thresholds.lowReviewScore

  const serviceRisk =
    highLateDelivery || lowReviewScore

  const hasServiceEvidence =
    state.lateDeliveryRate !== null &&
    state.averageReviewScore !== null &&
    thresholds.highLateDeliveryRate !== null &&
    thresholds.lowReviewScore !== null

  const healthyService =
    hasServiceEvidence && !serviceRisk

  const reasonCodes: BusinessReasonCode[] = []

  if (highValue) {
    reasonCodes.push("HIGH_VALUE")
  }

  if (highGrowth) {
    reasonCodes.push("HIGH_GROWTH")
  }

  if (highLateDelivery) {
    reasonCodes.push("HIGH_LATE_DELIVERY")
  }

  if (lowReviewScore) {
    reasonCodes.push("LOW_REVIEW_SCORE")
  }

  if (!hasServiceEvidence) {
    reasonCodes.push("MISSING_SERVICE_EVIDENCE")
  }

  let decision: BusinessDecision
  let priority: BusinessPriority

  if (highValue && serviceRisk) {
    decision = "RECOVER_SERVICE"
    priority = "P1"
  } else if (highValue && healthyService) {
    decision = "PROTECT_VALUE"
    priority = "P1"
  } else if (highGrowth && healthyService) {
    decision = "EXPAND"
    priority = "P2"
  } else if (serviceRisk) {
    decision = "INVESTIGATE"
    priority = "P2"
  } else {
    decision = "MONITOR"
    priority = "P3"
  }

  return {
    stateCode: state.stateCode,
    decision,
    priority,
    reasonCodes,
    evidence: {
      gmv: state.gmv,
      gmvGrowthRate: state.gmvGrowthRate,
      lateDeliveryRate: state.lateDeliveryRate,
      averageReviewScore: state.averageReviewScore,
    },
  }
}

export function evaluateStateBusinessDecisions(
  states: StateBusinessMetrics[],
): BusinessDecisionModelResult {
  if (states.length === 0) {
    return {
      thresholds: {
        highValueGmv: 0,
        highGrowthRate: null,
        highLateDeliveryRate: null,
        lowReviewScore: null,
      },
      states: [],
    }
  }

  const thresholds =
    calculateBusinessDecisionThresholds(states)

  return {
    thresholds,
    states: states.map((state) =>
      evaluateState(state, thresholds),
    ),
  }
}
