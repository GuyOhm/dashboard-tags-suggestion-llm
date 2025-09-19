locals {
  name = "${var.project}"
}

# ECR repository
resource "aws_ecr_repository" "app" {
  name = "${local.name}"
  image_scanning_configuration { scan_on_push = true }
}

# Security group (22 optional; 8501 for Streamlit)
resource "aws_security_group" "app" {
  name        = "${local.name}-sg"
  description = "SG for Streamlit app"
  vpc_id      = data.aws_vpc.default.id

  dynamic "ingress" {
    for_each = length(var.allow_ssh_cidr) > 0 ? [1] : []
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.allow_ssh_cidr
      description = "SSH"
    }
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = var.allow_http_cidr
    description = "Streamlit"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ARM64 Amazon Linux 2023 AMI
data "aws_ami" "al2023_arm" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }
}

# IAM role for EC2 to pull from ECR and use SSM
data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "ec2_role" {
  name               = "${local.name}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}
# ECR pull policy
resource "aws_iam_role_policy" "ec2_ecr_pull" {
  name = "${local.name}-ecr-pull"
  role = aws_iam_role.ec2_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Action = [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      Resource = "*"
    }]
  })
}
# Attach SSM managed policy for remote commands
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${local.name}-instance-profile"
  role = aws_iam_role.ec2_role.name
}

# EC2 instance (t4g.micro - Graviton/ARM)
resource "aws_instance" "app" {
  ami                         = data.aws_ami.al2023_arm.id
  instance_type               = "t4g.micro"
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true

  user_data = <<-EOF
    #!/bin/bash
    set -e
    dnf update -y
    dnf install -y docker awscli
    systemctl enable --now docker
    usermod -aG docker ec2-user

    # Login to ECR
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.app.repository_url}

    # Pull and run app
    docker pull ${aws_ecr_repository.app.repository_url}:latest || true
    docker stop app || true
    docker rm app || true
    docker run -d --name app -p 8501:8501 \
      ${aws_ecr_repository.app.repository_url}:latest || true
  EOF

  tags = { Name = local.name }
}