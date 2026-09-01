CREATE TABLE IF NOT EXISTS
  `balmy-nuance-468118-g4.olist_analytics.analytics_state_diagnostics_v2`
(
  state_code STRING NOT NULL,
  evidence_count INT64 NOT NULL,

  actual_negative_review_rate FLOAT64 NOT NULL,
  expected_negative_review_rate FLOAT64 NOT NULL,

  residual_pp FLOAT64 NOT NULL,
  ci_lower_pp FLOAT64 NOT NULL,
  ci_upper_pp FLOAT64 NOT NULL,
  z_score FLOAT64 NOT NULL,

  diagnostic_state STRING NOT NULL,

  model_version STRING NOT NULL,
  generated_at TIMESTAMP NOT NULL
);