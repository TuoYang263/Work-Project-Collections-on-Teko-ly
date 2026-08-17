"use client"

import { useMemo, useState } from "react"

import { AnalyticsStateMapLoader } from "@/components/analytics-state-map-loader"
import { AnalyticsSummaryPanel } from "@/components/analytics-summary-panel"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type {
  AnalyticsStateSummary,
  BrazilStateCode,
} from "@/server/analytics/analytics-state-summary"
import type { AnalyticsSummary } from "@/server/analytics/analytics-summary"

type AnalyticsDashboardProps = {
  data: AnalyticsSummary
  states: AnalyticsStateSummary[]
}

type StateSelection = {
  stateCode: BrazilStateCode
  stateName: string
}

export function AnalyticsDashboard({
  data,
  states,
}: AnalyticsDashboardProps) {
  const [selection, setSelection] =
    useState<StateSelection | null>(null)

  const selectedState = useMemo(() => {
    if (!selection) {
      return null
    }

    return (
      states.find(
        (state) =>
          state.stateCode ===
          selection.stateCode
      ) ?? null
    )
  }, [selection, states])

  const displayedSummary =
    useMemo<AnalyticsSummary>(() => {
      if (!selectedState) {
        return data
      }

      return {
        ...data,
        orderCount: selectedState.orderCount,
        gmv: selectedState.gmv,
        aov: selectedState.aov,
      }
    }, [data, selectedState])

  const scopeLabel =
    selection && selectedState
      ? `${selection.stateName} (${selection.stateCode})`
      : "All Brazil"

  return (
    <div className="space-y-6">
      <AnalyticsSummaryPanel
        data={displayedSummary}
        scopeLabel={scopeLabel}
      />

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle>
                Orders by customer state
              </CardTitle>

              <p className="mt-1 text-sm text-muted-foreground">
                {selection
                  ? `Selected: ${scopeLabel}`
                  : "All Brazil"}
                {" · "}
                Click a state to filter the KPI cards.
              </p>
            </div>

            {selection && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() =>
                  setSelection(null)
                }
              >
                Reset to All Brazil
              </Button>
            )}
          </div>
        </CardHeader>

        <CardContent>
          <AnalyticsStateMapLoader
            states={states}
            selectedStateCode={
              selection?.stateCode ?? null
            }
            onStateSelect={setSelection}
          />
        </CardContent>
      </Card>
    </div>
  )
}
