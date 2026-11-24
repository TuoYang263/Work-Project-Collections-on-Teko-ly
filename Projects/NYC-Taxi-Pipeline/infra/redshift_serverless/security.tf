# Dedicated Security Group for NYC V2 Redshift Serverless.
resource "aws_security_group" "redshift" {
  name_prefix = "nyc-v2-redshift-"
  description = "Private Redshift Serverless access"
  vpc_id      = var.vpc_id

  # No inbound rules at this stage.
  # Database access will initially use the Redshift Data API.

  tags = {
    Name      = "nyc-v2-redshift-sg"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}

# Allow HTTPS egress to the regional S3 service.
resource "aws_vpc_security_group_egress_rule" "redshift_s3" {
  security_group_id = aws_security_group.redshift.id

  description    = "HTTPS to S3 Gateway Endpoint"
  ip_protocol    = "tcp"
  from_port      = 443
  to_port        = 443
  prefix_list_id = aws_vpc_endpoint.redshift_s3.prefix_list_id
}
