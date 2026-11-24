# NYC V2 — R3 Redshift Serverless Deployment & Teardown Runbook

**Date:** 2026-11-24
**Status:** PRE-DEPLOYMENT / **NOT APPROVED TO APPLY**
**Scope:** `Projects/NYC-Taxi-Pipeline/infra/redshift_serverless/`
**Branch:** `feature/nyc-v2`
**Related ADR:** `airflow_pipeline/docs/nyc_v2_serving_and_cost_control_ADR.md` (SRV-01)

> **Safety contract:** This document is not authorization to create, query, or destroy AWS resources. `terraform apply`, SQL/COPY, and any destructive operation each require an explicit go/no-go decision. Do not run them while simply following read-only checks.

## 1. Purpose and verified baseline

R3 creates a **real, private, cost-governed Redshift Serverless serving environment** for already-certified NYC V2 Gold snapshots. It is not a permanently running compute tier, and creating infrastructure must **never** be triggered by a dashboard visit or the current offline Cost Router (`cw-v1.0`).

Current certified source release: `airflow-a521894129ca813bd8977d28` (three-month 2023-01–03 baseline). The existing S3 Serving release is immutable. R3 must not recompute Bronze/Silver/Gold or overwrite the source release.

Evidence as of 2026-11-24:

| Item | Observed outcome |
|---|---|
| Terraform `fmt`, `validate`, no-apply `plan` | PASS; **15 add / 0 change / 0 destroy** |
| Deployment identity | `nyc-v2` profile + Provider `assume_role` to `NYCV2TerraformDeployRole` |
| Terraform backend | Existing S3 bucket, separate `nyc-v2` backend profile, `use_lockfile=true` |
| IAM baseline | Four deployment-managed policies attached; S3 COPY permissions boundary separately created |
| EC2 dry runs | CreateSubnet, CreateSecurityGroup, CreateRouteTable, CreateVpcEndpoint: PASS |
| AWS inventory | No Redshift namespaces, workgroups, usage limits, or VPC endpoints reported in the last preflight |
| AWS Budget | $10 USD monthly, notifications >$5 / >$8 / >$10, 1 verified recipient, no service filter |
| Live Redshift / R4 SQL | **Not created or tested** |

### Planned resources (15)

- Redshift: namespace (1), workgroup (1), compute usage limit (1).
- IAM: COPY role (1), its restricted inline S3 policy (1).
- Networking: three private subnets (3), route table (1), route associations (3), security group (1), HTTPS egress rule to S3 managed prefix list (1), S3 Gateway Endpoint (1).

Expected: Frankfurt (`eu-central-1`); AWS account `438336772967`; VPC `vpc-0c9ec662214c94fca`; **4 base RPUs / 4 max RPUs**, daily **2 RPU-hours** with `deactivate`; `publicly_accessible=false`; Enhanced VPC Routing; **no NAT Gateway**.

## 2. Pre-apply go/no-go checklist (read-only)

**All boxes must be affirmatively reviewed before requesting deployment approval:**

- [ ] Operator explicitly authorizes a specific **paid R3 experiment**, date/session, scope and budget. Policy attachment and a successful Plan **do not** constitute approval.
- [ ] Current Frankfurt AWS pricing, expected active compute window, RMS storage, managed secret charges, taxes and account-level budget headroom checked. $10 is an **alert threshold, not a hard spending cap**.
- [ ] `NYCV2TerraformDeployRole` can still be assumed; ensure operator is in **the intended AWS account and Region**.
- [ ] `NYCV2RedshiftCopyBoundary` exists and latest local `iam.tf` contains `permissions_boundary` on `aws_iam_role.redshift_copy`.
- [ ] Current release manifests/Parquet still exist in the approved S3 prefix. No unreviewed `serving_release_id` override.
- [ ] All three subnet CIDRs are still unoccupied; default VPC subnet routing and egress rules reviewed. New subnets may temporarily inherit the VPC main route table **until their private-route-table associations are created**; no public IP assignment is configured.
- [ ] EC2 route-association and security-group-egress IAM permissions were **not** fully dry-run tested. Redshift/IAM create and full teardown permissions likewise remain **unproven at runtime**. An AccessDenied during `apply` remains possible.
- [ ] Redshift COPY role trust (`aws:SourceArn`) and restricted S3 Gateway Endpoint policy are understood as **not yet runtime COPY-verified**.
- [ ] Administrator bootstrap for R4 is planned separately: successful Data API authentication is **not equivalent to database DDL/COPY privileges**.
- [ ] Decide beforehand whether R3 resources will be removed after acceptance or retained until R4. This is an **explicit lifecycle decision**, not a router-triggered action.

**Do not add AdministratorAccess or blindly relax tag conditions if an IAM action fails.** Diagnose the exact action/resource/condition, apply a scoped remedy, and review the new plan.

### Read-only local and AWS checks

From `infra/redshift_serverless/`:

```bash
aws sts get-caller-identity --profile nyc-v2-deploy --region eu-central-1
terraform fmt -check
terraform validate
terraform plan -input=false -lock-timeout=30s -no-color

aws redshift-serverless list-namespaces --profile nyc-v2-deploy --region eu-central-1 --output json
aws redshift-serverless list-workgroups --profile nyc-v2-deploy --region eu-central-1 --output json
aws redshift-serverless list-usage-limits --profile nyc-v2-deploy --region eu-central-1 --output json
```

Expected before first deployment: `Plan: 15 to add, 0 to change, 0 to destroy`, inventories empty, correct STS role ARN. If actual results differ, **STOP** and review. `terraform plan` does not prove create/delete API permissions.

## 3. Approved deployment procedure — **DO NOT EXECUTE UNTIL EXPLICIT APPROVAL**

1. Capture a time-stamped approval record including maximum planned spending, intended first SQL/COPY session, and emergency cleanup owner.
2. Log in with the existing `nyc-v2` AWS login profile; Provider uses its own `assume_role` block. **Do not change `backend.tf` to `nyc-v2-deploy`.**
3. Create a fresh saved plan **after approval**:

   ```bash
   terraform plan -input=false -lock-timeout=30s -out=deployment.tfplan
   terraform show -no-color deployment.tfplan
   ```

   Expected **15/0/0**, exactly the reviewed resources, no unexpected public access/NAT. Saved plans may contain sensitive values; they belong in local ignored files, never Git, email or public logs.
4. Only after operator reconfirms the reviewed plan, apply **that exact saved plan**:

   ```bash
   terraform apply -input=false deployment.tfplan
   ```

   **No unattended retries.** Save errors and inspect state before taking further action.
5. Immediately perform the **post-apply gate below**. **Do not start Data API SQL, COPY, a dashboard, or automated polling jobs until it passes.**

## 4. Mandatory post-apply gate — before SQL/COPY

Run these only if a deployment has been explicitly approved and performed:

```bash
terraform state list

aws redshift-serverless get-workgroup \
  --workgroup-name nyc-v2-serving-wg \
  --profile nyc-v2-deploy --region eu-central-1 \
  --query 'workgroup.{Name:workgroupName,Status:status,Base:baseCapacity,Max:maxCapacity,Public:publiclyAccessible,EnhancedVpcRouting:enhancedVpcRouting,Arn:workgroupArn}' \
  --output json

aws redshift-serverless get-namespace \
  --namespace-name nyc-v2-serving \
  --profile nyc-v2-deploy --region eu-central-1 \
  --query 'namespace.{Name:namespaceName,Status:status,SecretArn:adminPasswordSecretArn}' \
  --output json

aws redshift-serverless list-usage-limits \
  --profile nyc-v2-deploy --region eu-central-1 --output json
```

**Go/no-go conditions:**

- Workgroup is `AVAILABLE`, belongs to the intended namespace/account, with Base/Max **4/4**, private access **false** for public access, and Enhanced VPC Routing **true**.
- Verify the **actual** usage-limit response contains a limit attached to this workgroup's `workgroupArn` with `usageType=serverless-compute`, `period=daily`, `amount=2`, `breachAction=deactivate`. Do not infer limit existence from `terraform plan` or an incomplete `apply`.
- The three new subnets route through the project private route table; **no `0.0.0.0/0` IGW/NAT route** in that dedicated route table; S3 Gateway Endpoint is attached to it. Verify Security Group ingress and egress rules. A live S3 COPY must still be separately tested.
- Review the Secrets Manager administrator secret reference, without outputting a secret value.

**CRITICAL:** Terraform creates the workgroup **before** its usage limit. If workgroup creation succeeds but the limit fails, the environment is *not approved for querying*. Suspend all SQL/COPY, record which resources exist, and prioritize approved remediation/teardown. Usage limit, AWS Budget and 4 RPU capacity are **not** absolute billing ceilings; AWS billing data can lag.

## 5. Failed or partial `apply` — incident response

1. **Stop** new queries, COPY, retry loops and automated executors. Record UTC time and the first failing AWS API/action; do not paste secrets or full sensitive state into an issue.
2. Do **not** delete S3 Terraform State, `.tflock` manually, or recreate a second stack. Do not apply the same saved plan without a new state review.
3. Use `terraform state list` and a fresh `terraform plan` to compare known resources; inspect Redshift namespaces/workgroups/usage limits and tagged EC2/IAM resources through read-only calls or AWS Console.
4. If the workgroup exists but the usage limit is missing, treat this as **cost-sensitive**; no SQL. If the role cannot clean up, arrange a narrow, audited emergency permission/administrator cleanup path **before** any further spend.
5. Classify failure: (a) IAM permission/condition; (b) AWS quota/unsupported parameter; (c) network/endpoint; (d) namespace/workgroup service-linked role; (e) provider state/dependency. Do not assume `terraform validate` or `--dry-run` validated all of these.
6. Choose exactly one **approved** recovery: repair with a fresh reviewed plan, or generate a destroy plan for resources tracked by State. Manual console deletes/import/state reconciliation are **last-resort, documented** operations; never silently detach or delete resources unrelated to this module.
7. Capture final AWS inventory and subsequent billing evidence. If any managed storage, snapshot, secret or workgroup remains, cleanup is **not complete**.

## 6. R4 access and serving acceptance (separately approved)

A deployed workgroup alone does **not** make R4 complete. Before even `SELECT 1`, verify usage limit and explicit query approval. Test Data API IAM auth and database permissions as separate steps, then execute release-scoped DDL/COPY through an approved bootstrap mechanism. Validate six Gold tables: row counts, field order/types, timestamp parity, representative metrics and release identity. Keep existing serving revision on failure; do not make a dashboard visit trigger SQL or provisioning.

Current three-month S3 release is a **baseline**; future 12-month/multi-year benchmarks are separate, measured, approved experiments. Processing scale and Serving scale differ.

## 7. Approved teardown — **DESTRUCTIVE / REQUIRES SEPARATE APPROVAL**

Before destruction, decide whether warehouse tables need a backup. **Deleting the namespace destroys live warehouse data**, and deleting the workgroup disables SQL access; a retained manual snapshot may continue to incur charges. Immutable source S3 Serving releases are outside this Terraform stack and must remain untouched.

After an explicit **teardown** decision, review a destroy plan:

```bash
terraform plan -destroy -input=false -lock-timeout=30s -out=teardown.tfplan
terraform show -no-color teardown.tfplan
```

Confirm the plan refers **only to NYC V2's module-owned resources** and does not touch existing S3 buckets, source releases, other VPC subnets, shared IAM roles or global budget configuration. Then, and only with explicit final confirmation:

```bash
terraform apply -input=false teardown.tfplan
```

`terraform destroy`/destroy plans can fail because of IAM resource conditions, service dependencies, protection settings, role-policy cleanup, or an existing Redshift snapshot. **Do not remove IAM constraints broadly just to force deletion.** Escalate to the documented partial-failure process and use least-privilege cleanup.

### Post-teardown acceptance (read-only)

```bash
terraform state list
aws redshift-serverless list-namespaces --profile nyc-v2-deploy --region eu-central-1 --output json
aws redshift-serverless list-workgroups --profile nyc-v2-deploy --region eu-central-1 --output json
aws redshift-serverless list-usage-limits --profile nyc-v2-deploy --region eu-central-1 --output json
aws redshift-serverless list-snapshots --profile nyc-v2-deploy --region eu-central-1 --output json
aws ec2 describe-vpc-endpoints --filters Name=vpc-id,Values=vpc-0c9ec662214c94fca --profile nyc-v2-deploy --region eu-central-1 --output json
```

Also confirm project subnets, project route-table associations, security group/rules, COPY IAM Role and managed admin secret in Console/CLI (some inventory calls may need extra read-only permissions). Confirm **no unexpected retained snapshots or project-specific paid resources**. `terraform state list` should not list managed resources from this module (provider data sources can be listed separately).

Deletion of the Redshift namespace normally removes its **Redshift-managed** administrator secret; still verify no unexpected secret remains. A remote S3 Terraform state object, its previous versions and source Serving release objects may remain by design. Do **not** delete the shared raw/state S3 bucket to remove Redshift.

Review actual charges in AWS Cost Explorer/Billing after billing data updates. Immediate `$0.00` on the dashboard is **not** proof that no charges accrued.

## 8. Evidence to preserve without leaking secrets

- Approval record: purpose, time window, max planned spend, operator and cleanup decision.
- Git revision, `terraform version`, `terraform providers`, reviewed Plan summary, `terraform state list` resource names.
- Role ARN, Workgroup config, actual Usage Limit JSON, route table/SG/endpoint inspection, timing of readiness.
- R4 COPY / SQL row-count evidence and actual RPU usage, if approved.
- Final teardown inventory and delayed AWS billing check.

Keep AWS credentials, Secrets Manager values, sensitive Terraform State and saved plan bytes **out of Git and public logs**. Never share `.terraform/`, `.tfstate`, `terraform.tfvars` or private credentials as review artifacts.

## 9. Known open risks — blocking unconditional deployment approval

1. **IAM lifecycle authorization remains partly unverified**: EC2 route associations, egress-rule API, Redshift create/delete, IAM Role lifecycle and Secrets Manager integration have not been exercised against real resources.
2. **Usage Limit timing gap**: the Workgroup may exist before limit creation; post-apply gate is mandatory.
3. **R4 bootstrap not validated**: private Data API access is plausible, but database DDL permissions, trust policy and real S3 COPY path need runtime proof.
4. **Networking details**: the Security Group has no ingress, HTTPS egress is S3-only, and first-created subnets temporarily inherit the main VPC route table before association. Validate actual routing after apply.
5. **Full teardown untested**: clean removal, snapshots, administrator secret and billing are not yet verified. Do not claim zero ongoing cost until AWS inventory and delayed billing evidence support it.

## 10. Deployment Readiness Evidence — 2026-11-24

### 10.1 Verified baseline

| Verification Item | Result |
|---|---|
| Terraform CLI | v1.17.2 |
| AWS Provider | v6.73.0 |
| Minimum Terraform version | >= 1.11.0, < 2.0.0 |
| Native S3 state locking compatibility | PASS |
| Terraform formatting | PASS |
| Terraform configuration validation | PASS |
| Historical Terraform Plan | 15 add, 0 change, 0 destroy |
| Deployment Role identity | PASS |
| Default Network ACL review | PASS |
| Main VPC route table review | PASS |
| IAM lifecycle policy static review | PASS |
| IAM lifecycle policy attachments | 4/4 verified |
| COPY Role permissions boundary review | PASS |
| Unexpected inline policies | None |
| Redshift Service-linked Role | Created and verified |
| Existing Redshift snapshots | None |

### 10.2 IAM lifecycle verification

The following customer-managed policies were inspected
and confirmed attached to `NYCV2TerraformDeployRole`:

- `NYCV2TerraformRedshiftLifecycle`
- `NYCV2TerraformNetworkLifecycle`
- `NYCV2TerraformCopyRoleLifecycle`
- `NYCV2TerraformAdminSecretLifecycle`

The following permissions boundary was also inspected:

- `NYCV2RedshiftCopyBoundary`

No unexpected inline policies were found on the
deployment role.

Static policy verification does not guarantee successful
runtime authorization for all AWS API operations.

### 10.3 Account-level service dependency

The Redshift Service-linked Role was successfully
bootstrapped and verified on 2026-11-24.

- Role: `AWSServiceRoleForRedshift`
- Trusted service: `redshift.amazonaws.com`
- Managed policy: `AmazonRedshiftServiceLinkedRolePolicy`
- Creation timestamp: `2026-11-24T18:41:43Z`

This role is an account-level AWS service dependency.

It is not managed by the temporary NYC V2 Terraform
stack and must not be included in routine stack teardown.

No additional permissions were granted to the Terraform
Deployment Role during this bootstrap operation.

### 10.4 Network verification

The existing VPC main route table and default Network ACL
were inspected.

- Default Network ACL permits IPv4 inbound and outbound
  traffic.
- Existing main route table contains an Internet Gateway
  route.
- Dedicated private route table will be created by Terraform.
- No NAT Gateway is planned.
- Redshift public access remains disabled.
- Enhanced VPC Routing remains enabled.
- Security Group is designed for S3 HTTPS egress only.

Actual routing, Security Group rules and S3 COPY
connectivity must be verified after deployment.

### 10.5 Budget and cost governance

Budget status observed on 2026-11-24:

| Metric | Value |
|---|---:|
| Monthly AWS Budget | USD 10.00 |
| Observed actual spend | USD 0.002 |
| Observed forecast spend | USD 0.002 |
| Remaining reported budget | USD 9.998 |
| Proposed first experiment budget | USD 2.00 |
| Redshift base capacity | 4 RPU |
| Redshift maximum capacity | 4 RPU |
| Daily compute usage limit | 2 RPU-hours |
| Usage limit breach action | deactivate |

The proposed experiment budget requires explicit operator
approval before deployment.

AWS Budget notifications and Redshift usage limits are
cost governance mechanisms, not absolute billing ceilings.

Compute, managed storage, Secrets Manager, taxes and
potential residual resource charges must be considered
separately.

Billing information may be delayed.

### 10.6 Remaining runtime risks

The following items remain unverified until controlled
deployment and runtime acceptance:

- Actual IAM create, update and delete authorization.
- Actual private networking and S3 COPY connectivity.
- COPY Role assumption and trust policy compatibility.
- Usage limit creation and enforcement.
- Managed administrator secret lifecycle.
- Redshift Data API authentication and database permissions.
- Partial deployment failure and recovery.
- Complete resource teardown.
- Residual resource and billing reconciliation.

### 10.7 Deployment authorization status

**Status: PRE-DEPLOYMENT / NOT APPROVED TO APPLY**

- Infrastructure as Code: verified.
- Static security and IAM review: completed.
- Account-level service dependency: verified.
- Budget headroom: verified against reported billing data.
- Runtime deployment: not executed.
- SQL and COPY execution: not executed.
- Teardown validation: not executed.

The successful completion of pre-deployment checks does
not constitute authorization to create paid AWS resources.

Explicit operator approval remains mandatory before
running `terraform apply`, SQL/COPY operations or
destructive infrastructure operations.

## 11. Reference docs

- [AWS: Redshift Serverless billing](https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-billing.html)
- [AWS: Redshift Serverless usage limits](https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-workgroup-max-rpu.html)
- [AWS: deleting workgroups](https://docs.aws.amazon.com/redshift/latest/mgmt/serverless_delete-workgroup.html)
- [AWS: deleting namespaces](https://docs.aws.amazon.com/redshift/latest/mgmt/serverless-console-namespace-delete.html)
- [AWS: Redshift-managed Secrets Manager administrator passwords](https://docs.aws.amazon.com/redshift/latest/mgmt/redshift-secrets-manager-integration.html)
- [AWS: Terraform S3 state backend practices](https://docs.aws.amazon.com/prescriptive-guidance/latest/terraform-aws-provider-best-practices/backend.html)
- [AWS: Redshift enhanced VPC routing](https://docs.aws.amazon.com/redshift/latest/mgmt/enhanced-vpc-enabling-cluster.html)
