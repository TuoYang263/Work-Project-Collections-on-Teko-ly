import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest"

vi.mock("server-only", () => ({}))

vi.mock(
  "./reliability-finding-repository",
  () => ({
    fetchReliabilityFindingRows: vi.fn(),
  }),
)

import { fetchReliabilityFindingRows } from "./reliability-finding-repository"
import {
  getReliabilityFinding,
  ReliabilityFindingNotFoundError,
} from "./reliability-finding-service"

const mockedFetchReliabilityFindingRows =
  vi.mocked(fetchReliabilityFindingRows)

const validFindingId =
  "20260810T030139Z_35356a7d:M9-R006:model:model.olist_ecommerce_analytics.fct_order_payments"

describe("Reliability finding service input validation", () => {
  beforeEach(() => {
    mockedFetchReliabilityFindingRows.mockReset()
  })

  it("allows a valid finding ID to reach the repository", async () => {
    mockedFetchReliabilityFindingRows.mockResolvedValue([])

    await expect(
      getReliabilityFinding(validFindingId),
    ).rejects.toBeInstanceOf(
      ReliabilityFindingNotFoundError,
    )

    expect(
      mockedFetchReliabilityFindingRows,
    ).toHaveBeenCalledOnce()

    expect(
      mockedFetchReliabilityFindingRows,
    ).toHaveBeenCalledWith(validFindingId)
  })

  it("rejects an empty finding ID before repository access", async () => {
    await expect(
      getReliabilityFinding(""),
    ).rejects.toBeInstanceOf(
      ReliabilityFindingNotFoundError,
    )

    expect(
      mockedFetchReliabilityFindingRows,
    ).not.toHaveBeenCalled()
  })

  it("rejects invalid characters before repository access", async () => {
    await expect(
      getReliabilityFinding(
        "this is not a valid finding",
      ),
    ).rejects.toBeInstanceOf(
      ReliabilityFindingNotFoundError,
    )

    expect(
      mockedFetchReliabilityFindingRows,
    ).not.toHaveBeenCalled()
  })

  it("rejects an excessively long finding ID before repository access", async () => {
    const excessivelyLongFindingId =
      `A${"a".repeat(512)}`

    await expect(
      getReliabilityFinding(
        excessivelyLongFindingId,
      ),
    ).rejects.toBeInstanceOf(
      ReliabilityFindingNotFoundError,
    )

    expect(
      mockedFetchReliabilityFindingRows,
    ).not.toHaveBeenCalled()
  })
})