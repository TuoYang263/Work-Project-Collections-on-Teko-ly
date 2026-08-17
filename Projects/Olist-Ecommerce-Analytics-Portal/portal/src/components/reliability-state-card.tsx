import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

type ReliabilityState =
  | "not-found"
  | "integrity-error"
  | "unavailable"

type ReliabilityStateCardProps = {
  state: ReliabilityState
}

const stateContent: Record<
  ReliabilityState,
  {
    label: string
    title: string
    description: string
  }
> = {
  "not-found": {
    label: "No review yet",
    title: "Reliability review",
    description:
      "No persisted deterministic review exists for this pipeline environment yet.",
  },

  "integrity-error": {
    label: "Data issue",
    title: "Reliability review",
    description:
      "The persisted review data is inconsistent and cannot be shown safely.",
  },

  unavailable: {
    label: "Unavailable",
    title: "Reliability review",
    description:
      "The current reliability review could not be loaded.",
  },
}

export function ReliabilityStateCard({
  state,
}: ReliabilityStateCardProps) {
  const content = stateContent[state]

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>{content.title}</CardTitle>
            <CardDescription>
              Deterministic pipeline quality review
            </CardDescription>
          </div>

          <Badge variant="outline">
            {content.label}
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        <p className="text-sm text-muted-foreground">
          {content.description}
        </p>
      </CardContent>
    </Card>
  )
}
