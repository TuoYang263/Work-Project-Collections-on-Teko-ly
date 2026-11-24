# NYC V2 — Redshift Serverless Infrastructure

Phase: R3 — Infrastructure as Code

Region: eu-central-1

Scope:
- Redshift Serverless Namespace and Workgroup
- IAM and S3 read access
- Private networking
- Cost and usage guardrails
- Reproducible deployment and teardown

Existing R1/R2 serving releases are reused.

Safety:
- No automatic terraform apply
- No paid AWS resource creation without approval
- No Terraform State or credentials in Git
- Review cost and security before deployment
