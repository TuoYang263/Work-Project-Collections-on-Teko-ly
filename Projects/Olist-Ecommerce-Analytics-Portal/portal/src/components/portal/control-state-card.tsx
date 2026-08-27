import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

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
              <CardTitle>Pipeline control</CardTitle>
              <CardDescription>
                Window controller and watermark state
              </CardDescription>
            </div>

            <Badge variant="outline">
              Not initialized
            </Badge>
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
              <CardTitle>Pipeline control</CardTitle>
              <CardDescription>
                Window controller and watermark state
              </CardDescription>
            </div>

            <Badge variant="destructive">
              Unavailable
            </Badge>
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
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <OverviewMetric
          label="Pipeline state"
          value={formatPipelineState(data.state)}
          badge={<PipelineStateBadge state={data.state} />}
        />

        <OverviewMetric
          label="Environment"
          value={data.environment}
          description={data.pipelineName}
        />

        <OverviewMetric
          label="Active attempt"
          value={
            data.activeAttempt
              ? `#${data.activeAttempt.attemptNumber}`
              : "None"
          }
          description={
            data.activeAttempt
              ? "Processing window in progress"
              : "No active processing attempt"
          }
        />

        <OverviewMetric
          label="Control version"
          value={String(data.controlVersion)}
          description={`Updated ${formatUtc(data.updatedAt)}`}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card className="h-full">
          <CardHeader>
            <CardTitle>
              Processing window
            </CardTitle>
            <CardDescription>
              Latest successful watermark and active work.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-5">
            <section>
              <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
                Last successful window
              </div>

              {data.lastSuccessfulWindow ? (
                <div className="mt-2 rounded-lg border bg-muted/20 px-3 py-3">
                  <div className="text-sm font-medium">
                    {formatUtc(
                      data.lastSuccessfulWindow.start
                    )}
                  </div>

                  <div className="my-1 text-xs text-muted-foreground">
                    to
                  </div>

                  <div className="text-sm font-medium">
                    {formatUtc(
                      data.lastSuccessfulWindow.end
                    )}
                  </div>
                </div>
              ) : (
                <div className="mt-2 rounded-lg border border-dashed px-3 py-4 text-sm text-muted-foreground">
                  No successful window yet.
                </div>
              )}
            </section>

            {data.activeAttempt ? (
              <section className="border-t pt-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
                    Current attempt
                  </div>

                  <Badge variant="outline">
                    Attempt #{data.activeAttempt.attemptNumber}
                  </Badge>
                </div>

                <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
                  <Detail
                    label="Attempt ID"
                    value={data.activeAttempt.attemptId}
                  />
                  <Detail
                    label="Retry of"
                    value={
                      data.activeAttempt.retryOfAttemptId ??
                      "No"
                    }
                  />
                  <Detail
                    label="Window start"
                    value={formatUtc(
                      data.activeAttempt.windowStart
                    )}
                  />
                  <Detail
                    label="Window end"
                    value={formatUtc(
                      data.activeAttempt.windowEnd
                    )}
                  />
                </div>
              </section>
            ) : (
              <section className="border-t pt-4">
                <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
                  Current attempt
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  No active attempt.
                </p>
              </section>
            )}
          </CardContent>
        </Card>

        <Card className="h-full">
          <CardHeader>
            <CardTitle>
              Operational health
            </CardTitle>
            <CardDescription>
              Controller freshness and most recent error evidence.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-5">
            <div className="grid gap-4 sm:grid-cols-2">
              <Detail
                label="Controller updated"
                value={formatUtc(data.updatedAt)}
              />

              <Detail
                label="Pipeline"
                value={data.pipelineName}
              />
            </div>

            <section className="border-t pt-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
                  Last error
                </div>

                <Badge
                  variant={
                    data.lastError
                      ? "destructive"
                      : "outline"
                  }
                >
                  {data.lastError ? "Attention" : "Clear"}
                </Badge>
              </div>

              {data.lastError ? (
                <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-3">
                  <div className="text-sm font-medium">
                    {data.lastError.code ??
                      "No error code"}
                  </div>

                  <p className="mt-1 text-sm text-muted-foreground">
                    {data.lastError.message ??
                      "No error message"}
                  </p>
                </div>
              ) : (
                <div className="mt-3 rounded-lg border bg-muted/20 px-3 py-4 text-sm text-muted-foreground">
                  No current controller error.
                </div>
              )}
            </section>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function OverviewMetric({
  label,
  value,
  description,
  badge,
}: {
  label: string;
  value: string;
  description?: string;
  badge?: ReactNode;
}) {
  return (
    <Card className="shadow-none">
      <CardContent className="px-4 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
              {label}
            </div>

            <div className="mt-2 truncate text-2xl font-semibold tracking-tight">
              {value}
            </div>
          </div>

          {badge}
        </div>

        {description ? (
          <p className="mt-3 text-xs text-muted-foreground">
            {description}
          </p>
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
      return (
        <Badge variant="destructive">
          {state}
        </Badge>
      );

    case "WAITING_RETRY":
      return (
        <Badge variant="outline">
          Waiting
        </Badge>
      );

    case "RUNNING":
      return <Badge>Running</Badge>;

    case "IDLE":
      return (
        <Badge variant="secondary">
          Idle
        </Badge>
      );
  }
}

function formatPipelineState(
  state: PipelineState,
): string {
  return {
    IDLE: "Idle",
    RUNNING: "Running",
    FAILED: "Failed",
    WAITING_RETRY: "Waiting retry",
    QUARANTINED: "Quarantined",
  }[state];
}

function Detail({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-muted-foreground">
        {label}
      </div>

      <div className="mt-1 break-words text-sm font-medium">
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
