import config


def get_boto_client(service_name):
    import boto3
    client = boto3.client(
        service_name,
        aws_access_key_id=config.aws_access_key,
        aws_secret_access_key=config.aws_secrete_key,
        aws_session_token=config.aws_session_token,
        region_name=config.aws_region,
    )
    return client
