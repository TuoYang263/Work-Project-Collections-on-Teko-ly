import { connection } from "next/server"
import type { ReactNode } from "react"

import { AnalyticsDashboard } from "@/components/analytics-dashboard"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { AnalyticsStateSummary } from "@/server/analytics/analytics-state-summary"
import {
  AnalyticsStateSummaryIntegrityError,
  getAnalyticsStateSummaries,
} from "@/server/analytics/analytics-state-summary-service"
import type { AnalyticsSummary } from "@/server/analytics/analytics-summary"
import {
  AnalyticsSummaryIntegrityError,
  AnalyticsSummaryNotFoundError,
  getAnalyticsSummary,
} from "@/server/analytics/analytics-summary-service"

type AnalyticsPageResult =
  | {
      status: "available"
      data: AnalyticsSummary
      states: AnalyticsStateSummary[]
    }
  | {
      status: "not-found"
    }
  | {
      status: "integrity-error"
    }
  | {
      status: "unavailable"
    }

export default async function AnalyticsPage() {
  await connection()

  const result =
    await loadAnalyticsPageResult()

  if (result.status === "available") {
    return (
      <AnalyticsPageShell>
        <AnalyticsDashboard
          data={result.data}
          states={result.states}
        />
      </AnalyticsPageShell>
    )
  }

  const content = {
    "not-found": {
      title: "No analytics data yet",
      message:
        "No analytical KPI summary is available.",
    },

    "integrity-error": {
      title: "Analytics data issue",
      message:
        "The analytical data is inconsistent and cannot be shown safely.",
    },

    unavailable: {
      title: "Analytics unavailable",
      message:
        "The analytical data could not be loaded.",
    },
  }[result.status]

  return (
    <AnalyticsPageShell>
      <Card>
        <CardHeader>
          <CardTitle>
            {content.title}
          </CardTitle>
        </CardHeader>

        <CardContent>
          <p className="text-sm text-muted-foreground">
            {content.message}
          </p>
        </CardContent>
      </Card>
    </AnalyticsPageShell>
  )
}

async function loadAnalyticsPageResult(): Promise<AnalyticsPageResult> {
  try {
    const [data, states] =
      await Promise.all([
        getAnalyticsSummary(),
        getAnalyticsStateSummaries(),
      ])

    return {
      status: "available",
      data,
      states,
    }
  } catch (error) {
    if (
      error instanceof
      AnalyticsSummaryNotFoundError
    ) {
      return {
        status: "not-found",
      }
    }

    if (
      error instanceof
        AnalyticsSummaryIntegrityError ||
      error instanceof
        AnalyticsStateSummaryIntegrityError
    ) {
      console.error(
        "Analytics integrity error:",
        error
      )

      return {
        status: "integrity-error",
      }
    }

    console.error(
      "Analytics unavailable:",
      error
    )

    return {
      status: "unavailable",
    }
  }
}

function AnalyticsPageShell({
  children,
}: {
  children: ReactNode
}) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">
          Analytics
        </h1>

        <p className="mt-2 text-muted-foreground">
          Business performance and
          geospatial analytics.
        </p>
      </div>

      {children}
    </div>
  )
}
