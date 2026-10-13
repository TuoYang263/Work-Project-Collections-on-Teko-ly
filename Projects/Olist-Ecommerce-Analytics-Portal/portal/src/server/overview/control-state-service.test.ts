import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("server-only", () => ({}));

vi.mock("./control-state-repository", () => ({
  loadControlState: vi.fn(),
}));

import { loadControlState } from "./control-state-repository";
import { getControlStateOverview } from "./control-state-service";

const mockedLoadControlState = vi.mocked(loadControlState);

describe("getControlStateOverview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("maps the persisted cycle id into the overview model", async () => {
    mockedLoadControlState.mockResolvedValue({
      pipeline_name: "olist-dbt-build-job",
      environment: "prod",
      state: "IDLE",
      cycle_id: 3,

      last_successful_window_start: "2018-06-01T00:00:00Z",
      last_successful_window_end: "2018-07-01T00:00:00Z",

      active_window_start: null,
      active_window_end: null,
      active_attempt_id: null,
      active_attempt_number: null,
      active_retry_of_attempt_id: null,

      control_version: 148,

      last_error_code: null,
      last_error_message: null,

      updated_at: "2026-09-08T19:02:00Z",
    });

    const result = await getControlStateOverview();

    expect(result.cycleId).toBe(3);
    expect(result.controlVersion).toBe(148);
    expect(result.lastSuccessfulWindow).toEqual({
      start: "2018-06-01T00:00:00.000Z",
      end: "2018-07-01T00:00:00.000Z",
    });
  });
});