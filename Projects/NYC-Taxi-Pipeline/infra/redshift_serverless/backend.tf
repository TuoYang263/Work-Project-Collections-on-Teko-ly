terraform {
  backend "s3" {
    bucket  = "nyc-v2-raw-438336772967-eu-central-1-an"
    key     = "terraform-state/redshift-serverless/terraform.tfstate"
    region  = "eu-central-1"
    profile = "nyc-v2"

    encrypt      = true
    use_lockfile = true
  }
}
