resource "aws_vpc_endpoint" "redshift_s3" {
  vpc_id            = var.vpc_id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"

  route_table_ids = [
    aws_route_table.redshift_private.id
  ]

  # Only allow read operations needed for verified serving releases.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "ReadServingReleaseObjects"
        Effect    = "Allow"
        Principal = "*"
        Action    = ["s3:GetObject"]
        Resource = [
          "arn:aws:s3:::${var.serving_bucket_name}/serving/releases/*"
        ]
      },
      {
        Sid       = "InspectServingBucket"
        Effect    = "Allow"
        Principal = "*"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.serving_bucket_name}"
        ]
      }
    ]
  })

  tags = {
    Name      = "nyc-v2-redshift-s3-endpoint"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}
