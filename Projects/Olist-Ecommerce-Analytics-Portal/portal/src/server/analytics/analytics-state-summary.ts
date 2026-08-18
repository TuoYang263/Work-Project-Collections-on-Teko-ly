export const BRAZIL_STATE_CODES = [
  "AC",
  "AL",
  "AP",
  "AM",
  "BA",
  "CE",
  "DF",
  "ES",
  "GO",
  "MA",
  "MT",
  "MS",
  "MG",
  "PA",
  "PB",
  "PR",
  "PE",
  "PI",
  "RJ",
  "RN",
  "RS",
  "RO",
  "RR",
  "SC",
  "SP",
  "SE",
  "TO",
] as const

export type BrazilStateCode =
  (typeof BRAZIL_STATE_CODES)[number]

export type AnalyticsStateSummary = {
  stateCode: BrazilStateCode
  orderCount: number
  gmv: number
  aov: number
  deliveryObservationCount: number
  lateDeliveryRate: number | null
  reviewedOrderCount: number
  averageReviewScore: number | null
}