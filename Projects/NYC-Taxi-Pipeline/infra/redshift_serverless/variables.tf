variable "aws_region" {
  description = "AWS Region for NYC V2 Redshift Serverless."
  type        = string
  default     = "eu-central-1"
  nullable    = false

  validation {
    condition     = var.aws_region == "eu-central-1"
    error_message = "NYC V2 R3 only supports eu-central-1."
  }
}

variable "vpc_id" {
  description = "Existing VPC ID for NYC V2 Redshift."
  type        = string
  nullable    = false

  validation {
    condition     = can(regex("^vpc-([0-9a-f]{8}|[0-9a-f]{17})$", var.vpc_id))
    error_message = "A valid AWS VPC ID is required."
  }
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDRs by Availability Zone."
  type        = map(string)

  default = {
    "eu-central-1a" = "172.31.48.0/24"
    "eu-central-1b" = "172.31.49.0/24"
    "eu-central-1c" = "172.31.50.0/24"
  }

  validation {
    condition = (
      toset(keys(var.private_subnet_cidrs)) == toset([
        "eu-central-1a",
        "eu-central-1b",
        "eu-central-1c"
      ]) &&
      alltrue([
        for cidr in values(var.private_subnet_cidrs) :
        can(cidrhost(cidr, 0))
      ])
    )

    error_message = "Exactly three valid subnet CIDRs are required for eu-central-1a/b/c."
  }
}

variable "serving_bucket_name" {
  description = "Existing S3 bucket containing verified NYC V2 serving releases."
  type        = string
  nullable    = false

  validation {
    condition = (
      length(var.serving_bucket_name) >= 3 &&
      length(var.serving_bucket_name) <= 63 &&
      can(regex("^[a-z0-9][a-z0-9.-]*[a-z0-9]$", var.serving_bucket_name))
    )
    error_message = "A valid existing S3 bucket name is required."
  }
}

variable "serving_release_id" {
  description = "Verified NYC V2 serving release authorized for Redshift COPY."
  type        = string
  default     = "airflow-a521894129ca813bd8977d28"
  nullable    = false

  validation {
    condition = can(regex(
      "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
      var.serving_release_id
    ))
    error_message = "Invalid serving release ID."
  }
}

# Redshift Serverless namespace configuration.
variable "redshift_namespace_name" {
  description = "Namespace for NYC V2 serving data."
  type        = string
  default     = "nyc-v2-serving"
  nullable    = false
}

# Redshift Serverless workgroup configuration.
variable "redshift_workgroup_name" {
  description = "Compute workgroup for NYC V2 serving."
  type        = string
  default     = "nyc-v2-serving-wg"
  nullable    = false
}

# Start with the minimum supported capacity in Frankfurt.
variable "redshift_base_capacity" {
  description = "Base Redshift Processing Units (RPUs)."
  type        = number
  default     = 4
  nullable    = false

  validation {
    condition     = var.redshift_base_capacity == 4
    error_message = "NYC V2 R3 currently permits only 4 base RPUs."
  }
}

# Maximum Redshift Serverless compute capacity.
variable "redshift_max_capacity" {
  description = "Maximum RPUs allowed for the workgroup."
  type        = number
  default     = 4
  nullable    = false

  validation {
    condition     = var.redshift_max_capacity == 4
    error_message = "NYC V2 R3 permits a maximum of 4 RPUs."
  }
}

# Daily compute usage guardrail.
variable "redshift_daily_rpu_hours" {
  description = "Daily Redshift compute usage limit in RPU-hours."
  type        = number
  default     = 2
  nullable    = false

  validation {
    condition     = var.redshift_daily_rpu_hours == 2
    error_message = "NYC V2 R3 currently permits 2 RPU-hours per day."
  }
}
