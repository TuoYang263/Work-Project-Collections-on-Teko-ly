terraform {
  backend "s3" {
    bucket         = "tf-state-438336772967-eu-north-1"
    key            = "aws-demo/terraform.tfstate"
    region         = "eu-north-1"
    dynamodb_table = "tf-lock-eu-north-1"
    encrypt        = true
  }
}