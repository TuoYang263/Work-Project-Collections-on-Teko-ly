export type ReliabilityFindingDetail = {
  evaluationId: string
  findingId: string

  reviewId: string
  monitoringRunId: string
  jobName: string
  environment: string
  reviewedAt: string

  ruleId: string
  result: "TRIGGERED"
  severity: string | null

  entityType: string
  entityId: string | null

  evidenceSource: string
  evidence: Record<string, unknown>
  reason: string
}
