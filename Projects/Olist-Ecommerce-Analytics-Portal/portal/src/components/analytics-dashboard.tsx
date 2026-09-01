"use client"

import { useEffect, useMemo, useRef, useState } from "react"

import { AnalyticsStateMapLoader } from "@/components/analytics-state-map-loader"
import { AnalyticsSummaryPanel } from "@/components/analytics-summary-panel"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { AnalyticsStateDiagnosticV2 } from "@/server/analytics/analytics-state-diagnostic-v2"
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
  diagnostics: AnalyticsStateDiagnosticV2[]
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

function formatDiagnosticState(
  state: AnalyticsStateDiagnosticV2["diagnosticState"]
): string {
  const labels: Record<
    AnalyticsStateDiagnosticV2["diagnosticState"],
    string
  > = {
    WORSE_THAN_EXPECTED: "Worse than expected",
    BETTER_THAN_EXPECTED: "Better than expected",
    AS_EXPECTED: "As expected",
    INSUFFICIENT_EVIDENCE: "Insufficient evidence",
  }

  return labels[state]
}

function describeDiagnosticState(
  state: AnalyticsStateDiagnosticV2["diagnosticState"]
): string {
  const descriptions: Record<
    AnalyticsStateDiagnosticV2["diagnosticState"],
    string
  > = {
    WORSE_THAN_EXPECTED:
      "Negative-review risk is materially higher than expected for this order and delivery mix.",

    BETTER_THAN_EXPECTED:
      "Negative-review risk is materially lower than expected for this order and delivery mix.",

    AS_EXPECTED:
      "Observed negative-review risk is broadly consistent with what we would expect for this order and delivery mix.",

    INSUFFICIENT_EVIDENCE:
      "There are too few eligible orders to make a strong state-level diagnostic judgment.",
  }

  return descriptions[state]
}

export function AnalyticsDashboard({
  data,
  states,
  diagnostics,
  decisions,
}: AnalyticsDashboardProps) {
  const [selection, setSelection] =
    useState<StateSelection | null>(null)

  const detailsRef =
    useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!selection) {
      return
    }

    const frame = requestAnimationFrame(() => {
      detailsRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      })
    })

    return () => {
      cancelAnimationFrame(frame)
    }
  }, [selection])

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

  const selectedDiagnostic = useMemo(() => {
    if (!selection) {
      return null
    }

    return (
      diagnostics.find(
        (diagnostic) =>
          diagnostic.stateCode ===
          selection.stateCode
      ) ?? null
    )
  }, [diagnostics, selection])

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

      <div
        ref={detailsRef}
        className={
          selectedDiagnostic
            ? "grid scroll-mt-24 gap-6 xl:grid-cols-2 xl:items-stretch"
            : "space-y-6 scroll-mt-24"
        }
      >
        <Card className="h-full">
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
            <div className="space-y-4">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="text-sm font-medium">
                    27 customer-state markets
                  </div>

                  <p className="mt-1 text-xs text-muted-foreground">
                    Current action mix across Brazil.
                  </p>
                </div>

                {topIntervention && (
                  <div className="min-w-[220px] rounded-lg border bg-muted/20 px-3 py-2.5">
                    <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                      Top intervention
                    </div>

                    <div className="mt-1 text-sm font-semibold">
                      {topIntervention.stateCode}
                      {" · "}
                      {formatDecision(
                        topIntervention.decision
                      )}
                      {" · "}
                      {topIntervention.priority}
                    </div>
                  </div>
                )}
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-lg border bg-background px-3 py-3">
                  <div className="text-2xl font-semibold tracking-tight">
                    {decisionCounts.recoverService}
                  </div>

                  <div className="mt-1 text-sm font-medium">
                    Recover Service
                  </div>

                  <div className="mt-0.5 text-xs text-muted-foreground">
                    High-value + service risk
                  </div>
                </div>

                <div className="rounded-lg border bg-background px-3 py-3">
                  <div className="text-2xl font-semibold tracking-tight">
                    {decisionCounts.protectValue}
                  </div>

                  <div className="mt-1 text-sm font-medium">
                    Protect Value
                  </div>

                  <div className="mt-0.5 text-xs text-muted-foreground">
                    High-value + healthy
                  </div>
                </div>

                <div className="rounded-lg border bg-background px-3 py-3">
                  <div className="text-2xl font-semibold tracking-tight">
                    {decisionCounts.investigate}
                  </div>

                  <div className="mt-1 text-sm font-medium">
                    Investigate
                  </div>

                  <div className="mt-0.5 text-xs text-muted-foreground">
                    Service-risk signal
                  </div>
                </div>

                <div className="rounded-lg border bg-background px-3 py-3">
                  <div className="text-2xl font-semibold tracking-tight">
                    {decisionCounts.monitor}
                  </div>

                  <div className="mt-1 text-sm font-medium">
                    Monitor
                  </div>

                  <div className="mt-0.5 text-xs text-muted-foreground">
                    No strong intervention
                  </div>
                </div>
              </div>

              <details className="border-t pt-3 text-xs text-muted-foreground">
                <summary className="cursor-pointer font-medium text-foreground">
                  Decision model details
                </summary>

                <div className="mt-2">
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
              </details>
            </div>
          )}
        </CardContent>
      </Card>

      {selectedDiagnostic && (
        <Card className="h-full">
          <CardHeader>
            <CardTitle>
              Review risk vs expected
            </CardTitle>

            <p className="mt-1 text-sm text-muted-foreground">
              Business action shows what to do.
              This diagnostic shows whether
              negative-review risk for{" "}
              {scopeLabel} is unusually high or
              low after accounting for the order
              and delivery mix.
            </p>
          </CardHeader>

          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="text-lg font-semibold">
                  {formatDiagnosticState(
                    selectedDiagnostic.diagnosticState
                  )}
                </div>

                <p className="mt-1 text-sm text-muted-foreground">
                  {describeDiagnosticState(
                    selectedDiagnostic.diagnosticState
                  )}
                </p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <div>
                  <div className="text-xs text-muted-foreground">
                    Observed negative reviews
                  </div>

                  <div className="font-medium">
                    {(
                      selectedDiagnostic
                        .actualNegativeReviewRate * 100
                    ).toFixed(2)}
                    %
                  </div>
                </div>

                <div>
                  <div className="text-xs text-muted-foreground">
                    Expected negative reviews
                  </div>

                  <div className="font-medium">
                    {(
                      selectedDiagnostic
                        .expectedNegativeReviewRate * 100
                    ).toFixed(2)}
                    %
                  </div>
                </div>

                <div>
                  <div className="text-xs text-muted-foreground">
                    Difference
                  </div>

                  <div className="font-medium">
                    {selectedDiagnostic.residualPp >= 0
                      ? "+"
                      : ""}
                    {selectedDiagnostic.residualPp.toFixed(
                      2
                    )}{" "}
                    pp
                  </div>
                </div>

                <div>
                  <div className="text-xs text-muted-foreground">
                    Orders evaluated
                  </div>

                  <div className="font-medium">
                    {selectedDiagnostic.evidenceCount.toLocaleString()}{" "}
                    orders
                  </div>
                </div>
              </div>

              <details className="border-t pt-4 text-xs text-muted-foreground">
                <summary className="cursor-pointer font-medium text-foreground">
                  Method details
                </summary>

                <div className="mt-2">
                  95% diagnostic interval:{" "}
                  {selectedDiagnostic.ciLowerPp.toFixed(2)}
                  {" to "}
                  {selectedDiagnostic.ciUpperPp.toFixed(2)}
                  {" pp"}
                  {" · "}
                  Model version:{" "}
                  {selectedDiagnostic.modelVersion}
                </div>
              </details>
            </div>
          </CardContent>
        </Card>
      )}

      </div>

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