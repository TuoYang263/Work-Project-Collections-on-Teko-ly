import { connection } from "next/server";

import {
  ControlStateCard,
  type ControlStateCardState,
} from "@/components/portal/control-state-card";

import { ControlStateNotInitializedError } from "@/server/overview/control-state-repository";
import { getControlStateOverview } from "@/server/overview/control-state-service";

export default async function OverviewPage() {
  await connection();

  const controlState = await loadControlState();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Overview
        </h1>

        <p className="mt-1 text-sm text-muted-foreground">
          Current operational state of the Olist data pipeline.
        </p>
      </div>

      <ControlStateCard result={controlState} />
    </div>
  );
}

async function loadControlState(): Promise<ControlStateCardState> {
  try {
    const data = await getControlStateOverview();

    return {
      status: "available",
      data,
    };
  } catch (error) {
    if (error instanceof ControlStateNotInitializedError) {
      return {
        status: "not-initialized",
      };
    }

    console.error(
      "Failed to load control state for overview.",
      error,
    );

    return {
      status: "unavailable",
    };
  }
}
