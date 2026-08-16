import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

import type {
  ControlStateOverview,
  PipelineState,
} from "@/server/overview/control-state";

export type ControlStateCardState =
  | {
      status: "available";
      data: ControlStateOverview;
    }
  | {
      status: "not-initialized";
    }
  | {
      status: "unavailable";
    };

interface ControlStateCardProps {
  result: ControlStateCardState;
}

export function ControlStateCard({
  result,
}: ControlStateCardProps) {
  if (result.status === "not-initialized") {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle>Pipeline Control</CardTitle>
              <CardDescription>
                Window controller and watermark state
              </CardDescription>
            </div>

            <Badge variant="outline">Not initialized</Badge>
          </div>
        </CardHeader>

        <CardContent>
          <p className="text-sm text-muted-foreground">
            No control state exists for the configured pipeline environment.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (result.status === "unavailable") {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle>Pipeline Control</CardTitle>
              <CardDescription>
                Window controller and watermark state
              </CardDescription>
            </div>

            <Badge variant="destructive">Unavailable</Badge>
          </div>
        </CardHeader>

        <CardContent>
          <p className="text-sm text-muted-foreground">
            The current pipeline state could not be loaded.
          </p>
        </CardContent>
      </Card>
    );
  }

  const { data } = result;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>Pipeline Control</CardTitle>
            <CardDescription>
              {data.pipelineName}
            </CardDescription>
          </div>

          <PipelineStateBadge state={data.state} />
        </div>
      </CardHeader>

      <CardContent className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Detail
            label="Environment"
            value={data.environment}
          />

          <Detail
            label="Control version"
            value={String(data.controlVersion)}
          />

          <Detail
            label="Updated at"
            value={formatUtc(data.updatedAt)}
          />

          <Detail
            label="Active attempt"
            value={
              data.activeAttempt
                ? `#${data.activeAttempt.attemptNumber}`
                : "None"
            }
          />
        </div>

        <Separator />

        <div className="grid gap-5 lg:grid-cols-2">
          <div>
            <div className="text-sm font-medium">
              Last successful window
            </div>

            {data.lastSuccessfulWindow ? (
              <div className="mt-2 text-sm text-muted-foreground">
                <div>
                  {formatUtc(data.lastSuccessfulWindow.start)}
                </div>
                <div>→</div>
                <div>
                  {formatUtc(data.lastSuccessfulWindow.end)}
                </div>
              </div>
            ) : (
              <p className="mt-2 text-sm text-muted-foreground">
                No successful window yet.
              </p>
            )}
          </div>

          <div>
            <div className="text-sm font-medium">
              Last error
            </div>

            {data.lastError ? (
              <div className="mt-2 space-y-1 text-sm text-muted-foreground">
                <div>
                  {data.lastError.code ?? "No error code"}
                </div>
                <div>
                  {data.lastError.message ?? "No error message"}
                </div>
              </div>
            ) : (
              <p className="mt-2 text-sm text-muted-foreground">
                None
              </p>
            )}
          </div>
        </div>

        {data.activeAttempt ? (
          <>
            <Separator />

            <div>
              <div className="text-sm font-medium">
                Current attempt
              </div>

              <div className="mt-2 grid gap-3 text-sm text-muted-foreground sm:grid-cols-2">
                <div>
                  Attempt ID: {data.activeAttempt.attemptId}
                </div>

                <div>
                  Attempt number: {data.activeAttempt.attemptNumber}
                </div>

                <div>
                  Window start:{" "}
                  {formatUtc(data.activeAttempt.windowStart)}
                </div>

                <div>
                  Window end:{" "}
                  {formatUtc(data.activeAttempt.windowEnd)}
                </div>

                <div>
                  Retry of:{" "}
                  {data.activeAttempt.retryOfAttemptId ?? "No"}
                </div>
              </div>
            </div>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}

function PipelineStateBadge({
  state,
}: {
  state: PipelineState;
}) {
  switch (state) {
    case "FAILED":
    case "QUARANTINED":
      return <Badge variant="destructive">{state}</Badge>;

    case "WAITING_RETRY":
      return <Badge variant="outline">{state}</Badge>;

    case "RUNNING":
      return <Badge>{state}</Badge>;

    case "IDLE":
      return <Badge variant="secondary">{state}</Badge>;
  }
}

function Detail({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </div>

      <div className="mt-1 text-sm font-medium">
        {value}
      </div>
    </div>
  );
}

function formatUtc(value: string): string {
  const date = new Date(value);

  return (
    new Intl.DateTimeFormat("en-GB", {
      dateStyle: "medium",
      timeStyle: "short",
      timeZone: "UTC",
    }).format(date) + " UTC"
  );
}
