import "server-only";

import { BigQuery } from "@google-cloud/bigquery";

let client: BigQuery | null = null;

type ServiceAccountCredentials = {
  client_email: string;
  private_key: string;
};

export function getBigQueryClient(
  projectId: string
): BigQuery {
  if (client === null) {
    client = new BigQuery({
      projectId,
      credentials:
        loadServiceAccountCredentials(),
    });
  }

  return client;
}

function loadServiceAccountCredentials():
  | ServiceAccountCredentials
  | undefined {
  const rawCredentials =
    process.env.GCP_SERVICE_ACCOUNT_JSON;

  if (!rawCredentials) {
    // Local development can continue to use
    // Google Application Default Credentials.
    return undefined;
  }

  let parsed: unknown;

  try {
    parsed = JSON.parse(rawCredentials);
  } catch {
    throw new Error(
      "GCP_SERVICE_ACCOUNT_JSON is not valid JSON."
    );
  }

  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !("client_email" in parsed) ||
    !("private_key" in parsed) ||
    typeof parsed.client_email !== "string" ||
    typeof parsed.private_key !== "string" ||
    !parsed.client_email.trim() ||
    !parsed.private_key.trim()
  ) {
    throw new Error(
      "GCP_SERVICE_ACCOUNT_JSON is missing required credentials."
    );
  }

  return {
    client_email: parsed.client_email,
    private_key: parsed.private_key,
  };
}
