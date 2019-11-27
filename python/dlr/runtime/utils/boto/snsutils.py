from utils.boto import botoutils
import config


def publish():
    client = botoutils.get_boto_client('sns')
    response = client.publish(
        TopicArn=config.sns_config_arn,
        Message="""{
            'default':
             {'test data',
            'key': 'value'
        }""",
        Subject='string',
    )
