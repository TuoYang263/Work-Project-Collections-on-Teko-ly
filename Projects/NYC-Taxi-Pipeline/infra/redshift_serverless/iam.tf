# Identify the AWS account used by this Terraform deployment.
data "aws_caller_identity" "current" {}

# Trust policy: who is allowed to assume the COPY role?
data "aws_iam_policy_document" "redshift_copy_trust" {
  statement {
    sid     = "AllowRedshiftAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type = "Service"
      identifiers = [
        "redshift-serverless.amazonaws.com",
        "redshift.amazonaws.com"
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }

    # Restrict role assumption to Redshift Serverless workgroups.
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"

      values = [
        "arn:aws:redshift-serverless:${var.aws_region}:${data.aws_caller_identity.current.account_id}:workgroup/*"
      ]
    }
  }
}

# IAM role for Redshift COPY operations.
resource "aws_iam_role" "redshift_copy" {
  name                 = "nyc-v2-redshift-copy"
  description          = "Read verified NYC V2 serving releases"
  assume_role_policy   = data.aws_iam_policy_document.redshift_copy_trust.json
  permissions_boundary = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/NYCV2RedshiftCopyBoundary"

  tags = {
    Name      = "nyc-v2-redshift-copy"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}

# Least-privilege S3 permissions for Redshift COPY.
data "aws_iam_policy_document" "redshift_copy_s3" {

  # Read only objects within the approved serving release.
  statement {
    sid     = "ReadVerifiedServingRelease"
    effect  = "Allow"
    actions = ["s3:GetObject"]

    resources = [
      "arn:aws:s3:::${var.serving_bucket_name}/serving/releases/${var.serving_release_id}/*"
    ]
  }

  # Redshift COPY also requires bucket listing permission.
  statement {
    sid     = "ListServingBucket"
    effect  = "Allow"
    actions = ["s3:ListBucket"]

    resources = [
      "arn:aws:s3:::${var.serving_bucket_name}"
    ]
  }
}

# Attach S3 permissions to the existing Redshift COPY role.
resource "aws_iam_role_policy" "redshift_copy_s3" {
  name   = "nyc-v2-redshift-copy-s3"
  role   = aws_iam_role.redshift_copy.id
  policy = data.aws_iam_policy_document.redshift_copy_s3.json
}
