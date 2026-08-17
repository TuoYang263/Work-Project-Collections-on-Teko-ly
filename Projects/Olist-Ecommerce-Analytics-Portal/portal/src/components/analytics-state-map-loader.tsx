"use client"

import dynamic from "next/dynamic"

import type {
  AnalyticsStateSummary,
  BrazilStateCode,
} from "@/server/analytics/analytics-state-summary"

type StateSelection = {
  stateCode: BrazilStateCode
  stateName: string
}

const AnalyticsStateMap = dynamic(
  () =>
    import(
      "@/components/analytics-state-map"
    ).then(
      (module) =>
        module.AnalyticsStateMap
    ),
  {
    ssr: false,

    loading: () => (
      <div className="flex h-[520px] items-center justify-center rounded-lg border text-sm text-muted-foreground">
        Loading Brazil state map...
      </div>
    ),
  }
)

export function AnalyticsStateMapLoader({
  states,
  selectedStateCode,
  onStateSelect,
}: {
  states: AnalyticsStateSummary[]
  selectedStateCode: BrazilStateCode | null
  onStateSelect: (
    selection: StateSelection | null
  ) => void
}) {
  return (
    <AnalyticsStateMap
      states={states}
      selectedStateCode={
        selectedStateCode
      }
      onStateSelect={onStateSelect}
    />
  )
}
