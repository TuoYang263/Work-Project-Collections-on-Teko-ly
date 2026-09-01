import {
  describe,
  expect,
  it,
  vi,
} from "vitest"

vi.mock("server-only", () => ({}))

vi.mock(
  "./analytics-state-diagnostic-v2-repository",
  () => ({
    fetchAnalyticsStateDiagnosticV2Rows:
      vi.fn(),
  }),
)

import {
  AnalyticsStateDiagnosticV2IntegrityError,
  mapAnalyticsStateDiagnosticsV2,
} from "./analytics-state-diagnostic-v2-service"

import { BRAZIL_STATE_CODES } from "./analytics-state-summary"

import type { AnalyticsStateDiagnosticV2Row } from "./analytics-state-diagnostic-v2-repository"

const MODEL_VERSION =
  "business_decision_v2_logit_001"

const GENERATED_AT =
  "2026-08-17T10:00:00.000Z"

function buildValidRows(): AnalyticsStateDiagnosticV2Row[] {
  return BRAZIL_STATE_CODES.map(
    (stateCode) => ({
      state_code: stateCode,
      evidence_count: 200,
      actual_negative_review_rate: 0.12,
      expected_negative_review_rate: 0.11,
      residual_pp: 1,
      ci_lower_pp: 0.2,
      ci_upper_pp: 1.8,
      z_score: 2,
      diagnostic_state:
        "WORSE_THAN_EXPECTED",
      model_version: MODEL_VERSION,
      generated_at: GENERATED_AT,
    }),
  )
}

describe(
  "Analytics state diagnostic v2 integrity",
  () => {
    it("accepts a coherent 27-state snapshot", () => {
      const result =
        mapAnalyticsStateDiagnosticsV2(
          buildValidRows(),
        )

      expect(result).toHaveLength(
        BRAZIL_STATE_CODES.length,
      )

      expect(
        new Set(
          result.map(
            (row) => row.modelVersion,
          ),
        ).size,
      ).toBe(1)

      expect(
        new Set(
          result.map(
            (row) => row.generatedAt,
          ),
        ).size,
      ).toBe(1)
    })

    it("rejects a residual that does not match actual minus expected", () => {
      const rows = buildValidRows()

      rows[0] = {
        ...rows[0],
        residual_pp: 0.5,
      }

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        AnalyticsStateDiagnosticV2IntegrityError,
      )

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow("Residual mismatch")
    })

    it("rejects a diagnostic state that does not match the frozen rule", () => {
      const rows = buildValidRows()

      rows[0] = {
        ...rows[0],
        diagnostic_state: "AS_EXPECTED",
      }

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        AnalyticsStateDiagnosticV2IntegrityError,
      )

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        "Diagnostic state mismatch",
      )
    })

    it("rejects mixed model versions in one snapshot", () => {
      const rows = buildValidRows()

      rows[0] = {
        ...rows[0],
        model_version:
          "business_decision_v2_logit_002",
      }

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        AnalyticsStateDiagnosticV2IntegrityError,
      )

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        "Mixed model_version values detected",
      )
    })

    it("rejects mixed generated_at values in one snapshot", () => {
      const rows = buildValidRows()

      rows[0] = {
        ...rows[0],
        generated_at:
          "2026-08-17T11:00:00.000Z",
      }

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        AnalyticsStateDiagnosticV2IntegrityError,
      )

      expect(() =>
        mapAnalyticsStateDiagnosticsV2(
          rows,
        ),
      ).toThrow(
        "Mixed generated_at values detected",
      )
    })
  },
)
