# Redshift Serverless namespace for NYC V2 Serving.
resource "aws_redshiftserverless_namespace" "serving" {
  namespace_name = var.redshift_namespace_name
  db_name        = "nyc_v2"

  # Let AWS manage the database administrator password.
  manage_admin_password = true

  # Attach the dedicated S3 COPY role.
  iam_roles = [
    aws_iam_role.redshift_copy.arn
  ]

  default_iam_role_arn = aws_iam_role.redshift_copy.arn

  depends_on = [
    aws_iam_role_policy.redshift_copy_s3
  ]

  tags = {
    Name      = "nyc-v2-serving"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}

# Redshift Serverless compute and network configuration.
resource "aws_redshiftserverless_workgroup" "serving" {
  workgroup_name = var.redshift_workgroup_name
  namespace_name = aws_redshiftserverless_namespace.serving.namespace_name

  base_capacity = var.redshift_base_capacity
  max_capacity  = var.redshift_max_capacity

  publicly_accessible  = false
  enhanced_vpc_routing = true

  subnet_ids = [
    for subnet in aws_subnet.redshift_private : subnet.id
  ]

  security_group_ids = [
    aws_security_group.redshift.id
  ]

  depends_on = [
    aws_route_table_association.redshift_private,
    aws_vpc_endpoint.redshift_s3,
    aws_vpc_security_group_egress_rule.redshift_s3
  ]

  tags = {
    Name      = "nyc-v2-serving-wg"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}

# Daily compute usage limit for NYC V2 Redshift Serverless.
resource "aws_redshiftserverless_usage_limit" "serving_daily" {
  resource_arn  = aws_redshiftserverless_workgroup.serving.arn
  usage_type    = "serverless-compute"
  period        = "daily"
  amount        = var.redshift_daily_rpu_hours
  breach_action = "deactivate"
}
