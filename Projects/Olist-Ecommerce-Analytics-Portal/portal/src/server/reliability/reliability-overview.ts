export type ReliabilityFinding = {
  evaluationId: string
  findingId: string
  ruleId: string
  severity: string | null
  entityType: string
  entityId: string | null
  evidenceSource: string
  reason: string
}

export type ReliabilityOverview = {
  reviewId: string
  monitoringRunId: string
  jobName: string
  environment: string
  reviewedAt: string
  summary: {
    total: number
    pass: number
    triggered: number
    notEvaluated: number
  }
  findings: ReliabilityFinding[]
}
