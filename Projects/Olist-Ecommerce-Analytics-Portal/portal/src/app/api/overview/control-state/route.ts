import {
  ControlStateNotInitializedError,
  DuplicateControlStateError,
} from "@/server/overview/control-state-repository";

import { getControlStateOverview } from "@/server/overview/control-state-service";

export async function GET() {
  try {
    const data = await getControlStateOverview();

    return Response.json({
      data,
      meta: {
        generatedAt: new Date().toISOString(),
      },
    });
  } catch (error) {
    if (error instanceof ControlStateNotInitializedError) {
      return Response.json(
        {
          error: {
            code: "CONTROL_STATE_NOT_INITIALIZED",
            message: "Pipeline control state is not initialized.",
          },
        },
        { status: 503 },
      );
    }

    if (error instanceof DuplicateControlStateError) {
      console.error(error);

      return Response.json(
        {
          error: {
            code: "CONTROL_STATE_INTEGRITY_ERROR",
            message: "Pipeline control state is unavailable.",
          },
        },
        { status: 500 },
      );
    }

    console.error("Failed to load pipeline control state.", error);

    return Response.json(
      {
        error: {
          code: "CONTROL_STATE_UNAVAILABLE",
          message: "Pipeline control state is unavailable.",
        },
      },
      { status: 503 },
    );
  }
}
