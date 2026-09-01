import "server-only";

export const PIPELINE_STATES = [
  "IDLE",
  "RUNNING",
  "FAILED",
  "WAITING_RETRY",
  "QUARANTINED",
] as const;

export type PipelineState = (typeof PIPELINE_STATES)[number];

export interface ControlStateOverview {
  pipelineName: string;
  environment: string;
  state: PipelineState;
  controlVersion: number;

  lastSuccessfulWindow: {
    start: string;
    end: string;
  } | null;

  activeAttempt: {
    attemptId: string;
    attemptNumber: number;
    windowStart: string;
    windowEnd: string;
    retryOfAttemptId: string | null;
  } | null;

  lastError: {
    code: string | null;
    message: string | null;
  } | null;

  updatedAt: string;
}
