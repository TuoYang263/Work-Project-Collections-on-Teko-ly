import { NextResponse } from "next/server"

import {
  getReliabilityOverview,
  ReliabilityReviewIntegrityError,
  ReliabilityReviewNotFoundError,
} from "@/server/reliability/reliability-overview-service"

export const runtime = "nodejs"

export async function GET() {
  try {
    const data = await getReliabilityOverview()

    return NextResponse.json({
      data,
      meta: {
        generatedAt: new Date().toISOString(),
      },
    })
  } catch (error) {
    if (error instanceof ReliabilityReviewNotFoundError) {
      return NextResponse.json(
        {
          error: {
            code: "RELIABILITY_REVIEW_NOT_FOUND",
            message:
              "No persisted reliability review exists for this pipeline.",
          },
        },
        { status: 404 }
      )
    }

    if (error instanceof ReliabilityReviewIntegrityError) {
      console.error(
        "Reliability review integrity error:",
        error
      )

      return NextResponse.json(
        {
          error: {
            code: "RELIABILITY_REVIEW_INTEGRITY_ERROR",
            message:
              "The persisted reliability review is inconsistent.",
          },
        },
        { status: 500 }
      )
    }

    console.error(
      "Reliability review source unavailable:",
      error
    )

    return NextResponse.json(
      {
        error: {
          code: "RELIABILITY_REVIEW_UNAVAILABLE",
          message:
            "The reliability review source is unavailable.",
        },
      },
      { status: 503 }
    )
  }
}
