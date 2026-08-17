import Link from "next/link"
import { connection } from "next/server"
import type { ReactNode } from "react"

import { ReliabilityFindingDetailView } from "@/components/reliability-finding-detail"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { ReliabilityFindingDetail } from "@/server/reliability/reliability-finding"
import {
  getReliabilityFinding,
  ReliabilityFindingIntegrityError,
  ReliabilityFindingNotFoundError,
} from "@/server/reliability/reliability-finding-service"

type FindingPageProps = {
  params: Promise<{
    findingId: string
  }>
}

type FindingPageResult =
  | {
      status: "available"
      finding: ReliabilityFindingDetail
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

export default async function FindingPage({
  params,
}: FindingPageProps) {
  await connection()

  const { findingId: encodedFindingId } = await params

  const findingId = decodeURIComponent(
    encodedFindingId
  )

  const result = await loadFindingPageResult(
    findingId
  )

  if (result.status === "available") {
    return (
      <FindingPageShell>
        <ReliabilityFindingDetailView
          finding={result.finding}
        />
      </FindingPageShell>
    )
  }

  const message = {
    "not-found":
      "The requested reliability finding does not exist.",
    "integrity-error":
      "The persisted finding data is inconsistent and cannot be shown safely.",
    unavailable:
      "The reliability finding could not be loaded.",
  }[result.status]

  return (
    <FindingPageShell>
      <Card>
        <CardHeader>
          <CardTitle>Finding unavailable</CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {message}
          </p>

          <Link
            href="/reliability"
            className="text-sm underline"
          >
            Back to reliability
          </Link>
        </CardContent>
      </Card>
    </FindingPageShell>
  )
}

async function loadFindingPageResult(
  findingId: string
): Promise<FindingPageResult> {
  try {
    const finding =
      await getReliabilityFinding(findingId)

    return {
      status: "available",
      finding,
    }
  } catch (error) {
    if (
      error instanceof ReliabilityFindingNotFoundError
    ) {
      return {
        status: "not-found",
      }
    }

    if (
      error instanceof ReliabilityFindingIntegrityError
    ) {
      console.error(
        "Reliability finding integrity error:",
        error
      )

      return {
        status: "integrity-error",
      }
    }

    console.error(
      "Reliability finding unavailable:",
      error
    )

    return {
      status: "unavailable",
    }
  }
}

function FindingPageShell({
  children,
}: {
  children: ReactNode
}) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">
          Finding
        </h1>

        <p className="mt-2 text-muted-foreground">
          Deterministic review evidence and context.
        </p>
      </div>

      {children}
    </div>
  )
}
