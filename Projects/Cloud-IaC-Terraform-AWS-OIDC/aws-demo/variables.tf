variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1" # Stockholm (close to Finland)
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name"
  type        = string
}

variable "iam_role_name" {
  description = "Name for the IAM role"
  type        = string
  default     = "terraform-demo-role"
}