These notebooks target to be run on edge device using Boto3 with AWS services.

## Setup device

Install the latest Boto 3 release via pip:
`pip install boto3`

Install jupyter notebook via pip:
`pip install notebook`

## AWS configuration 

Before you can begin using Boto 3, you should set up authentication credentials. You can follow the instructions in [AWS configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration).

## Troubleshooting

* ClientError: An error occurred (AccessDenied) when calling the CreateBucket operation: Access Denied

   -  Add `AdministratorAccess` policy to your role from [IAM console](https://console.aws.amazon.com/iam/home)
