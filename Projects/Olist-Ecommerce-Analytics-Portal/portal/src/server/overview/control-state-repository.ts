import "server-only";

import { getBigQueryClient } from "@/server/bigquery/client";

export interface ControlStateRow {
  pipeline_name: string;
  environment: string;
  state: string;

  last_successful_window_start: unknown;
  last_successful_window_end: unknown;

  active_window_start: unknown;
  active_window_end: unknown;
  active_attempt_id: string | null;
  active_attempt_number: number | null;
  active_retry_of_attempt_id: string | null;

  control_version: number;

  last_error_code: string | null;
  last_error_message: string | null;

  updated_at: unknown;
}

export class ControlStateNotInitializedError extends Error {}

export class DuplicateControlStateError extends Error {}

export async function loadControlState(): Promise<ControlStateRow> {
  const projectId = requireEnv("GCP_PROJECT_ID");
  const datasetId = process.env.CONTROL_DATASET_ID ?? "olist_control";
  const pipelineName =
    process.env.CONTROL_PIPELINE_NAME ?? "olist-dbt-build-job";
  const environment =
    process.env.CONTROL_PIPELINE_ENVIRONMENT ?? "prod";

  const tableFqn =
    `\`${projectId}.${datasetId}.pipeline_control_state\``;

  const query = `
    SELECT
      pipeline_name,
      environment,
      state,
      last_successful_window_start,
      last_successful_window_end,
      active_window_start,
      active_window_end,
      active_attempt_id,
      active_attempt_number,
      active_retry_of_attempt_id,
      control_version,
      last_error_code,
      last_error_message,
      updated_at
    FROM ${tableFqn}
    WHERE pipeline_name = @pipeline_name
      AND environment = @environment
    LIMIT 2
  `;

  const bigQueryClient = getBigQueryClient(projectId);
  const [rows] = await bigQueryClient.query({
    query,
    params: {
      pipeline_name: pipelineName,
      environment,
    },
  });

  if (rows.length === 0) {
    throw new ControlStateNotInitializedError(
      "Pipeline control state is not initialized.",
    );
  }

  if (rows.length > 1) {
    throw new DuplicateControlStateError(
      "More than one control state row exists for the pipeline environment.",
    );
  }

  return rows[0] as ControlStateRow;
}

function requireEnv(name: string): string {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return value;
}