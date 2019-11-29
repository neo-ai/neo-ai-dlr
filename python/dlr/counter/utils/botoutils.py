from .. import config


class SNS(object):
    def send(self, message):
        print(message)
        client = get_boto_client('sns')
        response = client.publish(
            TopicArn=config.sns_config_arn,
            Message=message,
            Subject='string',
        )
        print(response)


def get_boto_client(service_name):
    import boto3
    try:
        client = boto3.client(
            service_name,
            aws_access_key_id=config.aws_access_key,
            aws_secret_access_key=config.aws_secrete_key,
            aws_session_token=config.aws_session_token,
            region_name=config.aws_region,
        )
    except Exception as e:
        print(e)
    return client

