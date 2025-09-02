output "bucket_name" {
  value = aws_s3_bucket.demo.bucket
}

output "iam_role_arn" {
  value = aws_iam_role.demo_role.arn
}