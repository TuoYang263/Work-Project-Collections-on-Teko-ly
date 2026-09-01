import { connection } from "next/server"
import type { ReactNode } from "react"

import { ReliabilityOverviewPanel } from "@/components/reliability-overview-panel"
import { ReliabilityStateCard } from "@/components/reliability-state-card"
import type { ReliabilityOverview } from "@/server/reliability/reliability-overview"
import {
  getReliabilityOverview,
  ReliabilityReviewIntegrityError,
  ReliabilityReviewNotFoundError,
} from "@/server/reliability/reliability-overview-service"

type ReliabilityPageResult =
  | {
      status: "available"
      data: ReliabilityOverview
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

export default async function ReliabilityPage() {
  await connection()

  const result = await loadReliabilityPageResult()

  if (result.status === "available") {
    return (
      <ReliabilityPageState>
        <ReliabilityOverviewPanel data={result.data} />
      </ReliabilityPageState>
    )
  }

  return (
    <ReliabilityPageState>
      <ReliabilityStateCard state={result.status} />
    </ReliabilityPageState>
  )
}

async function loadReliabilityPageResult(): Promise<ReliabilityPageResult> {
  try {
    const data = await getReliabilityOverview()

    return {
      status: "available",
      data,
    }
  } catch (error) {
    if (error instanceof ReliabilityReviewNotFoundError) {
      return {
        status: "not-found",
      }
    }

    if (error instanceof ReliabilityReviewIntegrityError) {
      console.error(
        "Reliability review integrity error:",
        error
      )

      return {
        status: "integrity-error",
      }
    }

    console.error(
      "Reliability review unavailable:",
      error
    )

    return {
      status: "unavailable",
    }
  }
}

function ReliabilityPageState({
  children,
}: {
  children: ReactNode
}) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">
          Reliability
        </h1>

        <p className="mt-2 text-muted-foreground">
          Pipeline quality and deterministic findings.
        </p>
      </div>

      {children}
    </div>
  )
}
