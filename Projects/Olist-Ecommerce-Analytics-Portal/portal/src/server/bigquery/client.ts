import "server-only";

import { BigQuery } from "@google-cloud/bigquery";

let client: BigQuery | null = null;

export function getBigQueryClient(projectId: string): BigQuery {
  if (client === null) {
    client = new BigQuery({
      projectId,
    });
  }

  return client;
}
