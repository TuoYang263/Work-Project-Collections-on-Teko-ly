import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { AnalyticsSummary } from "@/server/analytics/analytics-summary"

type AnalyticsSummaryPanelProps = {
  data: AnalyticsSummary
  scopeLabel?: string
}

export function AnalyticsSummaryPanel({
  data,
  scopeLabel = "All Brazil",
}: AnalyticsSummaryPanelProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-muted/25 px-4 py-3">
        <div className="text-sm">
          <span className="font-medium text-foreground">
            Data period
          </span>
          <span className="text-muted-foreground">
            {" · "}
            Current analytical mart
          </span>
        </div>

        <div className="text-sm font-medium text-foreground">
          {formatDate(data.firstOrderDate)}
          {" — "}
          {formatDate(data.lastOrderDate)}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard
          label="Orders"
          value={formatInteger(data.orderCount)}
          description={scopeLabel}
        />

        <MetricCard
          label="GMV"
          value={formatCompactBRL(data.gmv)}
          description={`Gross merchandise value · ${scopeLabel}`}
        />

        <MetricCard
          label="AOV"
          value={formatBRL(data.aov)}
          description={`Average order value · ${scopeLabel}`}
        />
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  description,
}: {
  label: string
  value: string
  description: string
}) {
  return (
    <Card className="gap-3 py-4 shadow-sm">
      <CardHeader className="gap-1 px-4 pb-0">
        <CardDescription className="text-xs font-medium uppercase tracking-wide">
          {label}
        </CardDescription>

        <CardTitle className="text-2xl font-semibold tracking-tight lg:text-3xl">
          {value}
        </CardTitle>
      </CardHeader>

      <CardContent className="px-4 pb-0">
        <p className="text-xs text-muted-foreground">
          {description}
        </p>
      </CardContent>
    </Card>
  )
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("en-US").format(value)
}

function formatBRL(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "BRL",
  }).format(value)
}

function formatCompactBRL(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "BRL",
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value)
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeZone: "UTC",
  }).format(
    new Date(`${value}T00:00:00Z`)
  )
}
