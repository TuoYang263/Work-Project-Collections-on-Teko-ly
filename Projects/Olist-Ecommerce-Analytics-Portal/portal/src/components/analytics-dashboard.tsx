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
import type {
  BusinessDecision,
  BusinessDecisionModelResult,
  BusinessReasonCode,
} from "@/server/analytics/business-decision-v1"

type AnalyticsDashboardProps = {
  data: AnalyticsSummary
  states: AnalyticsStateSummary[]
  decisions: BusinessDecisionModelResult
}

type StateSelection = {
  stateCode: BrazilStateCode
  stateName: string
}

function formatDecision(
  decision: BusinessDecision
): string {
  const labels: Record<
    BusinessDecision,
    string
  > = {
    RECOVER_SERVICE: "Recover Service",
    PROTECT_VALUE: "Protect Value",
    EXPAND: "Expand",
    INVESTIGATE: "Investigate",
    MONITOR: "Monitor",
  }

  return labels[decision]
}

function formatReason(
  reason: BusinessReasonCode
): string {
  const labels: Record<
    BusinessReasonCode,
    string
  > = {
    HIGH_VALUE: "High-value market",
    HIGH_GROWTH: "High growth",
    HIGH_LATE_DELIVERY:
      "High late-delivery rate",
    LOW_REVIEW_SCORE: "Low review score",
    MISSING_SERVICE_EVIDENCE:
      "Missing service evidence",
  }

  return labels[reason]
}

function describeDecision(
  decision: BusinessDecision
): string {
  const descriptions: Record<
    BusinessDecision,
    string
  > = {
    RECOVER_SERVICE:
      "This is a high-value market with peer-relative service-risk signals. Service recovery should be prioritized.",

    PROTECT_VALUE:
      "This is a high-value market without current peer-relative service-risk triggers. Protect current performance.",

    EXPAND:
      "Growth and service-health evidence support potential expansion.",

    INVESTIGATE:
      "Service-risk signals are present, but the market is below the current high-value threshold. Investigate before allocating major intervention resources.",

    MONITOR:
      "No current high-value or service-risk trigger is present. Continue monitoring.",
  }

  return descriptions[decision]
}

export function AnalyticsDashboard({
  data,
  states,
  decisions,
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

  const selectedDecision = useMemo(() => {
    if (!selection) {
      return null
    }

    return (
      decisions.states.find(
        (state) =>
          state.stateCode ===
          selection.stateCode
      ) ?? null
    )
  }, [decisions.states, selection])

  const decisionCounts = useMemo(
    () => ({
      recoverService: decisions.states.filter(
        (state) =>
          state.decision ===
          "RECOVER_SERVICE"
      ).length,

      protectValue: decisions.states.filter(
        (state) =>
          state.decision ===
          "PROTECT_VALUE"
      ).length,

      investigate: decisions.states.filter(
        (state) =>
          state.decision ===
          "INVESTIGATE"
      ).length,

      monitor: decisions.states.filter(
        (state) =>
          state.decision === "MONITOR"
      ).length,
    }),
    [decisions.states]
  )

  const priorityCounts = useMemo(
    () => ({
      P1: decisions.states.filter(
        (state) => state.priority === "P1"
      ).length,

      P2: decisions.states.filter(
        (state) => state.priority === "P2"
      ).length,

      P3: decisions.states.filter(
        (state) => state.priority === "P3"
      ).length,
    }),
    [decisions.states]
  )

  const topIntervention = useMemo(
    () =>
      decisions.states
        .filter(
          (state) =>
            state.decision ===
            "RECOVER_SERVICE"
        )
        .sort(
          (left, right) =>
            right.evidence.gmv -
            left.evidence.gmv
        )[0] ?? null,
    [decisions.states]
  )

  const displayedSummary =
    useMemo<AnalyticsSummary>(() => {
      if (!selectedState) {
        return data
      }

      return {
        ...data,
        orderCount:
          selectedState.orderCount,
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
          <CardTitle>
            Business actions
          </CardTitle>

          <p className="mt-1 text-sm text-muted-foreground">
            Recommended actions derived from
            peer-relative market value and
            service health.
          </p>
        </CardHeader>

        <CardContent>
          {selectedDecision ? (
            <div className="space-y-4">
              <div>
                <div className="text-lg font-semibold">
                  {scopeLabel}
                </div>

                <div className="mt-1 font-medium">
                  {formatDecision(
                    selectedDecision.decision
                  )}
                  {" · "}
                  {selectedDecision.priority}
                </div>
              </div>

              <div>
                <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Why this action?
                </div>

                <p className="mt-1 text-sm">
                  {describeDecision(
                    selectedDecision.decision
                  )}
                </p>
              </div>

              {selectedDecision.reasonCodes
                .length > 0 && (
                <div>
                  <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Evidence signals
                  </div>

                  <p className="mt-1 text-sm text-muted-foreground">
                    {selectedDecision.reasonCodes
                      .map(formatReason)
                      .join(" · ")}
                  </p>
                </div>
              )}

              <div className="grid gap-4 sm:grid-cols-3">
                <div>
                  <div className="text-xs text-muted-foreground">
                    GMV
                  </div>

                  <div className="font-medium">
                    R${" "}
                    {selectedDecision.evidence.gmv.toLocaleString(
                      undefined,
                      {
                        maximumFractionDigits: 2,
                      }
                    )}
                  </div>
                </div>

                <div>
                  <div className="text-xs text-muted-foreground">
                    Late delivery
                  </div>

                  <div className="font-medium">
                    {selectedDecision.evidence
                      .lateDeliveryRate !== null
                      ? `${(
                          selectedDecision
                            .evidence
                            .lateDeliveryRate *
                          100
                        ).toFixed(1)}%`
                      : "No evidence"}
                  </div>
                </div>

                <div>
                  <div className="text-xs text-muted-foreground">
                    Review score
                  </div>

                  <div className="font-medium">
                    {selectedDecision.evidence
                      .averageReviewScore !==
                    null
                      ? selectedDecision.evidence
                          .averageReviewScore
                          .toFixed(2)
                      : "No evidence"}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <div>
                  <div className="text-2xl font-semibold">
                    {
                      decisionCounts.recoverService
                    }
                  </div>

                  <div className="text-sm font-medium">
                    Recover Service
                  </div>

                  <div className="text-xs text-muted-foreground">
                    High-value markets with
                    service risk
                  </div>
                </div>

                <div>
                  <div className="text-2xl font-semibold">
                    {decisionCounts.protectValue}
                  </div>

                  <div className="text-sm font-medium">
                    Protect Value
                  </div>

                  <div className="text-xs text-muted-foreground">
                    High-value markets currently
                    healthy
                  </div>
                </div>

                <div>
                  <div className="text-2xl font-semibold">
                    {decisionCounts.investigate}
                  </div>

                  <div className="text-sm font-medium">
                    Investigate
                  </div>

                  <div className="text-xs text-muted-foreground">
                    Service-risk signals
                    requiring review
                  </div>
                </div>

                <div>
                  <div className="text-2xl font-semibold">
                    {decisionCounts.monitor}
                  </div>

                  <div className="text-sm font-medium">
                    Monitor
                  </div>

                  <div className="text-xs text-muted-foreground">
                    No strong intervention
                    signal
                  </div>
                </div>
              </div>

              {topIntervention && (
                <div className="border-t pt-4">
                  <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Top intervention
                  </div>

                  <div className="mt-1 font-semibold">
                    {
                      topIntervention.stateCode
                    }
                    {" · "}
                    {formatDecision(
                      topIntervention.decision
                    )}
                    {" · "}
                    {topIntervention.priority}
                  </div>

                  <p className="mt-1 text-sm text-muted-foreground">
                    High-value market with
                    current peer-relative
                    service-risk signals.
                  </p>
                </div>
              )}

              <div className="border-t pt-4 text-xs text-muted-foreground">
                Peer thresholds: high value ≥
                R${" "}
                {decisions.thresholds.highValueGmv.toLocaleString(
                  undefined,
                  {
                    maximumFractionDigits: 0,
                  }
                )}
                {" · "}
                late delivery ≥{" "}
                {decisions.thresholds
                  .highLateDeliveryRate !==
                null
                  ? `${(
                      decisions.thresholds
                        .highLateDeliveryRate *
                      100
                    ).toFixed(1)}%`
                  : "n/a"}
                {" · "}
                review score ≤{" "}
                {decisions.thresholds
                  .lowReviewScore !== null
                  ? decisions.thresholds.lowReviewScore.toFixed(
                      2
                    )
                  : "n/a"}
                {" · "}
                Priority mix: P1{" "}
                {priorityCounts.P1}, P2{" "}
                {priorityCounts.P2}, P3{" "}
                {priorityCounts.P3}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle>
                Business action by customer
                state
              </CardTitle>

              <p className="mt-1 text-sm text-muted-foreground">
                {selection
                  ? `Selected: ${scopeLabel}`
                  : "All Brazil"}
                {" · "}
                Click a state to inspect its
                recommended action and KPI
                evidence.
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
            decisions={decisions.states}
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