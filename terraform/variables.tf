variable "aws_region" {
  type    = string
  default = "eu-west-1"
}

variable "project" {
  type    = string
  default = "dashboard-tags-suggestion"
}

variable "allow_ssh_cidr" {
  type    = list(string)
  default = []
}

variable "allow_http_cidr" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}