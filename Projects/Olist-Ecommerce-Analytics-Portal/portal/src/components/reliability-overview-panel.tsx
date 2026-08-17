import Link from "next/link"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import type { ReliabilityOverview } from "@/server/reliability/reliability-overview"

type ReliabilityOverviewPanelProps = {
  data: ReliabilityOverview
}

export function ReliabilityOverviewPanel({
  data,
}: ReliabilityOverviewPanelProps) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle>Latest deterministic review</CardTitle>
              <CardDescription>
                {data.jobName} · {data.environment}
              </CardDescription>
            </div>

            <Badge variant="outline">
              {formatTimestamp(data.reviewedAt)}
            </Badge>
          </div>
        </CardHeader>

        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Metric
              label="Total evaluations"
              value={data.summary.total}
            />
            <Metric
              label="PASS"
              value={data.summary.pass}
            />
            <Metric
              label="TRIGGERED"
              value={data.summary.triggered}
            />
            <Metric
              label="NOT EVALUATED"
              value={data.summary.notEvaluated}
            />
          </div>

          <Separator className="my-6" />

          <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-2">
            <div>
              <span className="font-medium text-foreground">
                Monitoring run:
              </span>{" "}
              {data.monitoringRunId}
            </div>

            <div>
              <span className="font-medium text-foreground">
                Review:
              </span>{" "}
              {data.reviewId}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Triggered findings</CardTitle>
          <CardDescription>
            Deterministic findings from the latest persisted review.
          </CardDescription>
        </CardHeader>

        <CardContent>
          {data.findings.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No triggered findings in this review.
            </p>
          ) : (
            <div className="space-y-4">
              {data.findings.map((finding, index) => (
                <div key={finding.evaluationId}>
                  {index > 0 && <Separator className="mb-4" />}

                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      {finding.severity && (
                        <Badge variant="outline">
                          {finding.severity}
                        </Badge>
                      )}

                      <Badge variant="secondary">
                        {finding.ruleId}
                      </Badge>
                    </div>

                    <div>
                      <p className="text-sm font-medium">
                        {finding.entityId ?? finding.entityType}
                      </p>

                      <p className="mt-1 text-sm text-muted-foreground">
                        {finding.reason}
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <p className="text-xs text-muted-foreground">
                        Evidence source: {finding.evidenceSource}
                      </p>

                      <Link
                        href={`/findings/${encodeURIComponent(
                          finding.findingId
                        )}`}
                        className="text-sm font-medium underline-offset-4 hover:underline"
                      >
                        View details
                      </Link>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

type MetricProps = {
  label: string
  value: number
}

function Metric({ label, value }: MetricProps) {
  return (
    <div className="rounded-lg border p-4">
      <p className="text-xs font-medium tracking-wide text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold">
        {value}
      </p>
    </div>
  )
}

function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(value))
}
