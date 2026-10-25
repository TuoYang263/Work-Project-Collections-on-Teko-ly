# NYC V2 — Amazon Bedrock Workshop applicability review

**Review date:** 2026-10-20
**Input:** `Build Chatbots with Amazon Bedrock Knowledge Bases and Agents.zip` (39 PDFs, 103 pages)
**Status:** Research review / engineering recommendations; NOT a deployed feature and NOT a superseding ADR.

## 1. What the supplied workshop actually covers

The tutorial builds a *mortgage application chatbot*, not an incident-response system. Its sections include:

- **Prerequisites and Prompt Engineering:** Bedrock Playground, experimenting with prompts/models, account/event environment.
- **Knowledge Bases:** connect documents, prepare a knowledge base, sync and test retrieval over mortgage policies and requirements.
- **Bedrock Agents:** agent-based question answering and tool/action interactions.
- **Bedrock Flows / Prompt Management:** model prompts and reusable templates; compose a flow with a Lambda affordability calculation, a Condition branch, prompts, and an Agent node; test the flow.
- **Bedrock Guardrails:** content filters, denied topics, word filters, sensitive information/PII, contextual grounding, testing and deployment.
- **Multi-Agent Collaboration:** intent classification agent, technical specialist, policy specialist, supervisor agent, test cases.
- **Own-account deployment and cleanup:** account infrastructure steps and resource teardown.

The ZIP contains image-based PDF printouts, not machine-readable lab source code, CloudFormation templates, notebooks, or executable sample scripts. Any URLs printed inside screenshots may need to be opened manually; the workshop landing page itself uses JavaScript and could not be directly text-scraped.

Workshop landing page: https://catalog.us-east-1.prod.workshops.aws/workshops/1e5b6626-f63c-41d7-adcb-4a3cdfd279ac/en-US

## 2. 2026 compatibility verdict

| Subject | 2026 status / caveat | NYC V2 choice |
| --- | --- | --- |
| Bedrock Knowledge Bases and RAG | Still supported and actively evolving | Later use for ADRs, quality contracts, runbooks, postmortems. Start with local files/keyword retrieval first. |
| Prompt Management | Supported: variables, variants, versioning | Pin and test prompts; first use Git versioning and structured output tests. |
| Bedrock Flows | Supported predefined AI-workflow orchestration | Borrow explicit branching and audit trail; do not replace Airflow's deterministic data orchestration. Optional later. |
| Bedrock Guardrails | Supported, including PII filters, topic filters, contextual grounding | Defense in depth on model input/output, NOT IAM/policy authorization or a kill switch. |
| Bedrock Agents (Classic) | Maintenance mode; not open to new customers from 2026-07-30 | Do not start new NYC V2 implementation on Classic Agents. |
| Classic multi-agent supervisor | Concept remains good, Classic creation paths outdated for new customers | Defer supervisor; AgentCore code-defined multi-agent or MCP agent-as-tool, only after evaluation. |
| AgentCore | Current target for new governed agents | Future managed deployment option; begin with one local read-only Investigator and tool contracts. |
| Model choices and click paths | Screenshots use older model families and console UI | Recheck model/region/access/pricing at build time. |
| Regions | AgentCore Runtime available in Frankfurt in current support matrix, but new Runtime V2 announced Sep 2026 in five initial regions excluding Frankfurt | Don't assume V2 in eu-central-1; verify feature-specific availability before deployment. |

**Important:** Workshop participation using an AWS-facilitated event account does not prove the user's separate AWS account qualifies as a pre-July-2026 Classic customer.

## 3. Transferable design patterns for NYC V2

### A. Static runbook RAG (later)

Put approved technical docs, quality contracts, ADRs and resolved incidents in a retrievable corpus. Retrieval responses must cite document identity and versions. Use direct, deterministic API reads for current Airflow runs, S3 manifests, check results and logs; don't embed every transient operational event into a vector database. Treat a retrieved document as untrusted data, not an executable instruction.

### B. Evidence-first tools / strict MCP target boundaries (first priority for AI MVP)

Read-only tools with strict schemas, quotas, scoping, authorization and redaction:
- `get_run_evidence(dag_id, run_id)`
- `get_task_attempts(dag_id, run_id, task_id?)`
- `get_stage_logs(dag_id, run_id, task_id, attempt, max_lines)`
- `get_raw_manifest(data_month, bucket, manifest_key)`
- `get_quality_results(dag_id, run_id)`
- `compare_runs(dag_id, run_id_a, run_id_b)`
- `get_runbook(error_category)`

Use deterministic extraction/classification first; AI may formulate hypotheses with evidence refs, but no mutation tools on initial version.

### C. Guardrails and safe shutdown

PII/secret redaction before an LLM call, prompt-injection handling for retrieved logs/docs, blocked unsafe remediation recommendations, explicit allowlists and per-run scoping. Bedrock Guardrails are text/model filters; `ApplyGuardrail` can be applied independently, but doesn't authorize tool calls and is not a complete prompt-injection defense. Some contextual grounding features have documented usage limitations. AgentCore Gateway/Policy IAM enforcement and the ADR v1.3 independent persistent kill switch remain separate, mandatory controls.

### D. Failure triage and branching

Borrow the workshop Flow pattern (Lambda deterministic calculator -> condition -> LLM/agent explanation):

`Run selected -> collect deterministic Airflow evidence -> normalize facts / classify error -> branch on first proven failure -> retrieve targeted runbook -> generate evidence-cited diagnosis -> operator review`.

Do not use an LLM to decide `sha256`, S3 size validity, DAG state or Quality Gate pass/fail.

### E. Supervisor, deferred

Future specialists could be `Ingestion/S3 Investigator`, `Airflow/Compute Investigator`, and `Data Quality/Delta Investigator`, with a supervisor that routes only cross-domain cases. Adopt only if benchmarked against a single agent for accuracy, citation faithfulness, tool-count, latency, cost and policy compliance. AgentCore migration docs state that custom orchestration and multi-agent may require code-defined agents, not a one-click replica of Classic supervisors.

## 4. Starter evaluation scenarios from real NYC V2 engineering

1. S3 versioned read `AccessDenied` when `s3:GetObjectVersion` permission is absent (prior observed integration case).
2. Raw S3 manifest checksum mismatch, bad Parquet content, or content length mismatch.
3. Airflow harmless task fails on attempt 1 and succeeds on retry; outcome `SUCCEEDED` but attempt history must be disclosed.
4. Airflow harmless task fails twice; outcome `FAILED`, first failing stage `harmless_task`.
5. Quality Gate reports invalid/empty Gold result; diagnosis must not claim successful publication.
6. Run evidence is missing or disagrees with Airflow; return `UNKNOWN`/`EVIDENCE_CONFLICT`, never fabricate evidence.
7. Investigator is `PAUSED` or `STOPPED`; tools must refuse further invocations independently of the language model.

## 5. Phased work after Unit 3D.14

**Phase 0 — finish real NYC pipeline first:** worker AWS credential delivery and refresh design; controlled real DAG execution; source-to-S3-to-Bronze and XCom evidence; avoid the default 2023 multi-month Full Refresh and legacy BigQuery export until isolated.

**Phase A — cost-first local investigator:** deterministic `get_*` tools, structured stage evidence, Streamlit diagnostics page, bounded single read-only investigator, kill switch, fault-injection tests. Does not require Bedrock Agents Classic or paid AgentCore deployment.

**Phase B — Bedrock inference and RAG:** use currently available Bedrock model in permitted region; restrict tokens, costs and invocations; retrieval on small approved runbook corpus; citations; prompt and output regression tests. Verify model/guardrail/region support and storage cost.

**Phase C — optional AgentCore:** Runtime + Gateway + Identity + Policy + Observability; scoped MCP targets; role-based authorization; cloud evidence ingestion path; evaluate whether managed infrastructure actually improves isolation, operations and maintainability.

**Phase D — optional supervisor:** only after single-agent evaluation data demonstrates a concrete gain.

## 6. Cost & security considerations

- Previous NYC V2 AWS budget alert is **not a hard spending cap**.
- Cloud vector databases, model inference, AgentCore runtime requests, Gateway requests, logs/traces, RAG embeddings/reranking can each be billed separately. Avoid keeping provisioned vector infrastructure running for casual prototyping.
- Avoid workshop CloudFormation stacks or sample permissions in a real account until reviewing resource count, region, IAM and teardown plan. An AWS-facilitated lab account is different from the user's own AWS billable account.
- Do not pass raw secrets, tokens, full customer-like data or unbounded logs to a model. Separate data plane health from model availability. `STOP` for the investigator must not stop ingestion.

## 7. Current AWS documentation checked

- Agents Classic maintenance mode: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html
- AgentCore regions: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html
- New Runtime V2 rollout (Sep 2026): https://aws.amazon.com/about-aws/whats-new/2026/09/new-agentcore-runtime-generally-available/
- AgentCore IAM: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html
- AgentCore Gateway: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-target-http-runtime.html
- Bedrock Knowledge Base retrieval: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-retrieval.html
- Guardrail independent API: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
- Guardrail grounding limitations: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html
- KB Guardrails coverage caveat: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve-generate.html
- Bedrock Flows nodes: https://docs.aws.amazon.com/bedrock/latest/userguide/flows-nodes.html
- Bedrock Prompt Management: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html
- AgentCore pricing: https://aws.amazon.com/bedrock/agentcore/pricing/

**ADR relationship:** This review supports the existing NYC V2 observability/AI Investigator ADR v1.3 direction; it does not supersede or silently change the accepted single-investigator, read-only, operator-approved, kill-switch-first sequencing.
