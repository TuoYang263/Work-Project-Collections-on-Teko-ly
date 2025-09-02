terraform {
  required_version = ">=1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# S3 bucket (no ACL argument to stay compatible with provider v5)
resource "aws_s3_bucket" "demo" {
  bucket        = var.bucket_name
  force_destroy = true
}

# Block all public access explicility
resource "aws_s3_bucket_public_access_block" "demo" {
  bucket                  = aws_s3_bucket.demo.id
  block_public_acls       = true
  block_public_policy     = true
  restrict_public_buckets = true
  ignore_public_acls      = true
}

# IAM role (example: assumble by EC2)
resource "aws_iam_role" "demo_role" {
  name = var.iam_role_name
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ec2.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

# Attach AmazonS3ReadOnlyAccessDemo access for demo
resource "aws_iam_role_policy_attachment" "s3_readonly" {
  role       = aws_iam_role.demo_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}