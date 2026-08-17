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
}

export function AnalyticsSummaryPanel({
  data,
}: AnalyticsSummaryPanelProps) {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard
          label="Orders"
          value={formatInteger(data.orderCount)}
          description="All orders"
        />

        <MetricCard
          label="GMV"
          value={formatCompactBRL(data.gmv)}
          description="Gross merchandise value"
        />

        <MetricCard
          label="AOV"
          value={formatBRL(data.aov)}
          description="Average order value"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Data period</CardTitle>
          <CardDescription>
            Order purchase dates represented in the current analytical mart.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <p className="text-lg font-medium">
            {formatDate(data.firstOrderDate)}
            {" — "}
            {formatDate(data.lastOrderDate)}
          </p>
        </CardContent>
      </Card>
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
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl">
          {value}
        </CardTitle>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-muted-foreground">
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
