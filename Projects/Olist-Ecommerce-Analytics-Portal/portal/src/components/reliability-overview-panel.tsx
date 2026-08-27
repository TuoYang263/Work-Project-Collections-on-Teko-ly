import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  ListChecks,
} from "lucide-react"
import Link from "next/link"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { ReliabilityOverview } from "@/server/reliability/reliability-overview"

type ReliabilityOverviewPanelProps = {
  data: ReliabilityOverview
}

export function ReliabilityOverviewPanel({
  data,
}: ReliabilityOverviewPanelProps) {
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border bg-muted/20 px-4 py-3">
        <div>
          <div className="text-sm font-medium">
            Latest deterministic review
          </div>

          <div className="mt-0.5 text-xs text-muted-foreground">
            {data.jobName} · {data.environment}
          </div>
        </div>

        <div className="text-right">
          <div className="text-sm font-medium">
            {formatTimestamp(data.reviewedAt)}
          </div>

          <div className="mt-0.5 text-xs text-muted-foreground">
            Review {shortId(data.reviewId)}
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Total evaluations"
          value={data.summary.total}
          icon={ListChecks}
          tone="neutral"
        />

        <MetricCard
          label="Pass"
          value={data.summary.pass}
          icon={CheckCircle2}
          tone="success"
        />

        <MetricCard
          label="Triggered"
          value={data.summary.triggered}
          icon={AlertTriangle}
          tone="attention"
        />

        <MetricCard
          label="Not evaluated"
          value={data.summary.notEvaluated}
          icon={CircleDashed}
          tone="neutral"
        />
      </div>

      <Card>
        <CardHeader className="gap-1">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>
                Triggered findings
              </CardTitle>

              <CardDescription>
                Deterministic findings from the latest persisted review.
              </CardDescription>
            </div>

            <Badge
              variant={
                data.findings.length > 0
                  ? "outline"
                  : "secondary"
              }
            >
              {data.findings.length} finding
              {data.findings.length === 1 ? "" : "s"}
            </Badge>
          </div>
        </CardHeader>

        <CardContent>
          {data.findings.length === 0 ? (
            <div className="rounded-lg border border-dashed px-4 py-6 text-sm text-muted-foreground">
              No triggered findings in this review.
            </div>
          ) : (
            <div className="divide-y">
              {data.findings.map((finding) => (
                <article
                  key={finding.evaluationId}
                  className="grid gap-4 py-4 first:pt-0 last:pb-0 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      {finding.severity && (
                        <Badge variant="outline">
                          {finding.severity}
                        </Badge>
                      )}

                      <Badge variant="secondary">
                        {finding.ruleId}
                      </Badge>

                      <span className="text-xs text-muted-foreground">
                        {finding.evidenceSource}
                      </span>
                    </div>

                    <div className="mt-3">
                      <p className="truncate text-sm font-semibold">
                        {finding.entityId ??
                          finding.entityType}
                      </p>

                      <p className="mt-1 max-w-4xl text-sm leading-6 text-muted-foreground">
                        {finding.reason}
                      </p>
                    </div>
                  </div>

                  <Link
                    href={`/findings/${encodeURIComponent(
                      finding.findingId
                    )}`}
                    className="inline-flex h-9 items-center justify-center rounded-md border bg-background px-3 text-sm font-medium shadow-sm transition-colors hover:bg-muted"
                  >
                    View details
                  </Link>
                </article>
              ))}
            </div>
          )}

          <details className="mt-5 border-t pt-4 text-xs text-muted-foreground">
            <summary className="cursor-pointer font-medium text-foreground">
              Review metadata
            </summary>

            <div className="mt-2 grid gap-2 md:grid-cols-2">
              <div className="break-all">
                <span className="font-medium text-foreground">
                  Monitoring run:
                </span>{" "}
                {data.monitoringRunId}
              </div>

              <div className="break-all">
                <span className="font-medium text-foreground">
                  Review:
                </span>{" "}
                {data.reviewId}
              </div>
            </div>
          </details>
        </CardContent>
      </Card>
    </div>
  )
}

type MetricTone =
  | "neutral"
  | "success"
  | "attention"

type MetricCardProps = {
  label: string
  value: number
  icon: typeof ListChecks
  tone: MetricTone
}

function MetricCard({
  label,
  value,
  icon: Icon,
  tone,
}: MetricCardProps) {
  const toneClass = {
    neutral:
      "bg-muted/50 text-muted-foreground",
    success:
      "bg-emerald-500/10 text-emerald-700",
    attention:
      "bg-amber-500/10 text-amber-700",
  }[tone]

  return (
    <Card className="shadow-none">
      <CardContent className="px-4 py-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
              {label}
            </div>

            <div className="mt-2 text-3xl font-semibold tracking-tight">
              {value}
            </div>
          </div>

          <div
            className={`flex size-9 items-center justify-center rounded-lg ${toneClass}`}
          >
            <Icon className="size-4" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function shortId(value: string): string {
  if (value.length <= 12) {
    return value
  }

  return `${value.slice(0, 8)}…${value.slice(-4)}`
}

function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(new Date(value))
}
