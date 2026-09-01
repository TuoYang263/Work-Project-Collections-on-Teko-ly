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
import type { ReliabilityFindingDetail } from "@/server/reliability/reliability-finding"

type ReliabilityFindingDetailProps = {
  finding: ReliabilityFindingDetail
}

export function ReliabilityFindingDetailView({
  finding,
}: ReliabilityFindingDetailProps) {
  return (
    <div className="space-y-6">
      <Link
        href="/reliability"
        className="text-sm text-muted-foreground hover:text-foreground"
      >
        ← Back to reliability
      </Link>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle>{finding.ruleId}</CardTitle>

              <CardDescription>
                Deterministic reliability finding
              </CardDescription>
            </div>

            <div className="flex gap-2">
              {finding.severity && (
                <Badge variant="outline">
                  {finding.severity}
                </Badge>
              )}

              <Badge variant="secondary">
                {finding.result}
              </Badge>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          <div>
            <p className="text-sm font-medium">
              Entity
            </p>

            <p className="mt-1 text-sm text-muted-foreground">
              {finding.entityId ?? finding.entityType}
            </p>
          </div>

          <div>
            <p className="text-sm font-medium">
              Reason
            </p>

            <p className="mt-1 text-sm text-muted-foreground">
              {finding.reason}
            </p>
          </div>

          <Separator />

          <div className="grid gap-4 text-sm md:grid-cols-2">
            <Metadata
              label="Evidence source"
              value={finding.evidenceSource}
            />

            <Metadata
              label="Environment"
              value={finding.environment}
            />

            <Metadata
              label="Monitoring run"
              value={finding.monitoringRunId}
            />

            <Metadata
              label="Review"
              value={finding.reviewId}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Evidence</CardTitle>

          <CardDescription>
            Evidence persisted with the deterministic evaluation.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <pre className="overflow-x-auto rounded-lg border bg-muted/30 p-4 text-xs">
            {JSON.stringify(
              finding.evidence,
              null,
              2
            )}
          </pre>
        </CardContent>
      </Card>
    </div>
  )
}

function Metadata({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div>
      <p className="font-medium">
        {label}
      </p>

      <p className="mt-1 break-all text-muted-foreground">
        {value}
      </p>
    </div>
  )
}
