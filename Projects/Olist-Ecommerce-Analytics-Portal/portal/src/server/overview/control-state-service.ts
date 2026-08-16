import "server-only";

import {
  PIPELINE_STATES,
  type ControlStateOverview,
  type PipelineState,
} from "@/server/overview/control-state";

import {
  loadControlState,
  type ControlStateRow,
} from "@/server/overview/control-state-repository";

export async function getControlStateOverview(): Promise<ControlStateOverview> {
  const row = await loadControlState();

  return mapControlState(row);
}

function mapControlState(row: ControlStateRow): ControlStateOverview {
  const state = parseState(row.state);

  const lastSuccessfulWindow =
    row.last_successful_window_start != null &&
    row.last_successful_window_end != null
      ? {
          start: toIsoString(row.last_successful_window_start),
          end: toIsoString(row.last_successful_window_end),
        }
      : null;

  const activeAttempt =
    row.active_attempt_id != null &&
    row.active_attempt_number != null &&
    row.active_window_start != null &&
    row.active_window_end != null
      ? {
          attemptId: row.active_attempt_id,
          attemptNumber: row.active_attempt_number,
          windowStart: toIsoString(row.active_window_start),
          windowEnd: toIsoString(row.active_window_end),
          retryOfAttemptId: row.active_retry_of_attempt_id,
        }
      : null;

  const lastError =
    row.last_error_code != null || row.last_error_message != null
      ? {
          code: row.last_error_code,
          message: row.last_error_message,
        }
      : null;

  return {
    pipelineName: row.pipeline_name,
    environment: row.environment,
    state,
    controlVersion: row.control_version,
    lastSuccessfulWindow,
    activeAttempt,
    lastError,
    updatedAt: toIsoString(row.updated_at),
  };
}

function parseState(value: string): PipelineState {
  if (
    PIPELINE_STATES.includes(
      value as PipelineState,
    )
  ) {
    return value as PipelineState;
  }

  throw new Error(`Unknown pipeline state: ${value}`);
}

function toIsoString(value: unknown): string {
  if (value instanceof Date) {
    return value.toISOString();
  }

  if (
    typeof value === "object" &&
    value !== null &&
    "value" in value &&
    typeof value.value === "string"
  ) {
    return new Date(value.value).toISOString();
  }

  if (typeof value === "string") {
    return new Date(value).toISOString();
  }

  throw new Error("Unsupported BigQuery timestamp value.");
}
