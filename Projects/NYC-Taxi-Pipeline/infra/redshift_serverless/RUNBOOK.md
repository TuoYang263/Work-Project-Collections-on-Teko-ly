# NYC V2 — R3 Redshift Serverless Deployment & Teardown Runbook

**Date:** 2026-11-24
**Status:** R3 LIFECYCLE VERIFIED / STACK DESTROYED / NO ACTIVE DEPLOYMENT
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
| Live Redshift / R4 SQL | R3 infrastructure deployed and destroyed; R4 SQL/COPY not tested |

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

## 9. Historical pre-deployment risks

1. **IAM lifecycle authorization remains partly unverified**: EC2 route associations, egress-rule API, Redshift create/delete, IAM Role lifecycle and Secrets Manager integration have not been exercised against real resources.
2. **Usage Limit timing gap**: the Workgroup may exist before limit creation; post-apply gate is mandatory.
3. **R4 bootstrap not validated**: private Data API access is plausible, but database DDL permissions, trust policy and real S3 COPY path need runtime proof.
4. **Networking details**: the Security Group has no ingress, HTTPS egress is S3-only, and first-created subnets temporarily inherit the main VPC route table before association. Validate actual routing after apply.
5. **Full teardown untested**: clean removal, snapshots, administrator secret and billing are not yet verified. Do not claim zero ongoing cost until AWS inventory and delayed billing evidence support it.

## 10. Historical Deployment Readiness Evidence — 2026-11-24

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


## 12. R3 Runtime Lifecycle Acceptance — 2026-11-24

**Status: PASS — infrastructure lifecycle completed.**

The historical pre-deployment evidence in Sections 9 and 10
is superseded by the runtime observations below.
No live R3 stack remains. Future deployments require fresh
operator approval.

### 12.1 Deployment and recovery evidence

- Tested Git revision: `66efea0`.
- Terraform CLI: v1.17.2.
- AWS Provider: v6.73.0.
- Initial saved plan: 15 add, 0 change, 0 destroy.
- Initial apply: 13 resources successfully created.
- Workgroup creation failed because the deployment identity
  lacked `ec2:DescribeAccountAttributes`.
- Added the required read-only permission.
- Verified partial state and actual AWS inventory.
- Generated and reviewed a fresh recovery plan.
- Recovery apply: 2 add, 0 change, 0 destroy.
- Workgroup creation completed in 67 seconds.
- Usage limit creation completed in 1 second.
- Final managed resource count: 15.
- Recovery apply exit code: 0.

### 12.2 Post-deployment runtime acceptance

| Verification | Observed result |
|---|---|
| Namespace | AVAILABLE |
| Workgroup | AVAILABLE |
| Database | nyc_v2 |
| Base capacity | 4 RPU |
| Maximum capacity | 4 RPU |
| Public access | Disabled |
| Enhanced VPC Routing | Enabled |
| Daily compute usage limit | 2 RPU-hours |
| Breach action | deactivate |
| Usage limit binding | Correct Workgroup ARN |
| Managed administrator secret reference | Present |
| Default COPY role association | Verified |
| Security Group ingress | None |
| Security Group egress | S3 HTTPS only |
| Private subnet associations | 3/3 |
| Internet/NAT default route | None |
| S3 Gateway Endpoint | Available |

The actual AWS configuration matched the intended
infrastructure design.

No SQL or COPY operations were executed.
Usage limit configuration was verified, but actual
limit enforcement was not exercised.

### 12.3 Teardown and recovery evidence

- Initial destroy plan: 0 add, 0 change, 15 destroy.
- Initial teardown successfully removed 11 resources.
- Three private subnets and one security group remained.
- The provider failed during its ENI dependency inspection
  because `ec2:DescribeNetworkInterfaces` was missing.
- Added the required read-only permission.
- Verified zero ENI dependencies for the remaining resources.
- Generated a fresh destroy plan containing exactly four
  expected resource addresses and resource IDs.
- Verified the saved plan SHA256 before execution.
- Recovery teardown: 0 add, 0 change, 4 destroy.
- Final apply exit code: 0.
- Final Terraform managed resource count: 0.

### 12.4 Runtime IAM remediation

| Incident | Missing IAM action | Remediation |
|---|---|---|
| Workgroup creation | ec2:DescribeAccountAttributes | Region-scoped read permission |
| Subnet and Security Group deletion | ec2:DescribeNetworkInterfaces | Region-scoped read permission |

Both actions are now included in the ReadNetworking
statement of `NYCV2TerraformNetworkLifecycle`.

The permissions are restricted to `eu-central-1`.

No AdministratorAccess was attached to the deployment role.
No ENI detach/delete permission was added.
No Terraform state removal or forced resource recreation
was required.

These permissions are part of the required deployment
lifecycle baseline for future R4 sessions.

### 12.5 Final residual resource audit

| Resource | Final observed state |
|---|---|
| Terraform managed resources | 0 |
| Redshift namespaces | 0 |
| Redshift workgroups | 0 |
| Redshift usage limits | 0 |
| Dedicated private subnets | 0 |
| Dedicated security group | 0 |
| Dedicated private route table | 0 |
| S3 Gateway Endpoint | Absent |
| Redshift-managed VPC Endpoint | Absent |
| Redshift-managed ENIs | 0 |
| Managed administrator secret | ResourceNotFoundException |
| Redshift snapshots | 0 |
| Redshift recovery points | 0 |
| Temporary COPY role | NoSuchEntity |

Read-only AWS inventory checks independently confirmed
the absence of the above experiment resources.

### 12.6 Preserved assets

| Asset | Observed result |
|---|---|
| Certified S3 serving release | Preserved |
| Serving release ID | airflow-a521894129ca813bd8977d28 |
| S3 release object count | 14 |
| S3 release total bytes | 14047189 |
| Remote Terraform state | Preserved |
| State object size | 1371 bytes |
| State server-side encryption | AES256 |

The S3 preservation check verified object count and total
size, not a new per-object content-integrity verification.

Existing shared infrastructure and account-level
deployment dependencies are outside the teardown scope.

### 12.7 Remaining validation and next stage

R3 infrastructure lifecycle acceptance is complete.

The following items remain outside the completed R3 scope:

- Delayed AWS billing reconciliation against the planned
  USD 2 experiment budget.
- Actual Redshift usage limit enforcement.
- Authorization for Redshift Data API queries.
- Database DDL and COPY permissions.
- Runtime COPY role assumption and S3 connectivity.
- Six Gold serving tables reconciliation.
- Serving API and dashboard integration.

R4 requires a fresh infrastructure deployment, current
budget verification, explicit approval, and a new
post-deployment quality gate.

The historical R3 deployment is not authorization to
recreate infrastructure or execute SQL/COPY.
