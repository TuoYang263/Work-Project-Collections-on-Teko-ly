resource "aws_subnet" "redshift_private" {
  for_each = var.private_subnet_cidrs

  vpc_id                  = var.vpc_id
  availability_zone       = each.key
  cidr_block              = each.value
  map_public_ip_on_launch = false

  tags = {
    Name      = "nyc-v2-redshift-private-${each.key}"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}

# Dedicated private route table.
# No default route to Internet Gateway or NAT Gateway.
resource "aws_route_table" "redshift_private" {
  vpc_id = var.vpc_id

  tags = {
    Name      = "nyc-v2-redshift-private-rt"
    Project   = "nyc-v2"
    ManagedBy = "Terraform"
  }
}

# Associate all three Redshift subnets with the private route table.
resource "aws_route_table_association" "redshift_private" {
  for_each = aws_subnet.redshift_private

  subnet_id      = each.value.id
  route_table_id = aws_route_table.redshift_private.id
}
