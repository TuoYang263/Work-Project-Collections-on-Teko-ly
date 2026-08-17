import { connection } from "next/server"
import type { ReactNode } from "react"

import { AnalyticsSummaryPanel } from "@/components/analytics-summary-panel"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { AnalyticsSummary } from "@/server/analytics/analytics-summary"
import {
  getAnalyticsSummary,
  AnalyticsSummaryIntegrityError,
  AnalyticsSummaryNotFoundError,
} from "@/server/analytics/analytics-summary-service"

type AnalyticsPageResult =
  | {
      status: "available"
      data: AnalyticsSummary
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

  const result = await loadAnalyticsPageResult()

  if (result.status === "available") {
    return (
      <AnalyticsPageShell>
        <AnalyticsSummaryPanel data={result.data} />
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
        "The analytical KPI summary is inconsistent and cannot be shown safely.",
    },
    unavailable: {
      title: "Analytics unavailable",
      message:
        "The analytical KPI summary could not be loaded.",
    },
  }[result.status]

  return (
    <AnalyticsPageShell>
      <Card>
        <CardHeader>
          <CardTitle>{content.title}</CardTitle>
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
    const data = await getAnalyticsSummary()

    return {
      status: "available",
      data,
    }
  } catch (error) {
    if (error instanceof AnalyticsSummaryNotFoundError) {
      return {
        status: "not-found",
      }
    }

    if (error instanceof AnalyticsSummaryIntegrityError) {
      console.error(
        "Analytics summary integrity error:",
        error
      )

      return {
        status: "integrity-error",
      }
    }

    console.error(
      "Analytics summary unavailable:",
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
          Business performance and geospatial analytics.
        </p>
      </div>

      {children}
    </div>
  )
}
