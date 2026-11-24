provider "aws" {
  region  = var.aws_region
  profile = "nyc-v2"

  assume_role {
    role_arn     = "arn:aws:iam::438336772967:role/NYCV2TerraformDeployRole"
    session_name = "nyc-v2-terraform"
  }

  allowed_account_ids = [
    "438336772967"
  ]
}
