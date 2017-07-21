import os
import json
import boto3
from utils import *

SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']

def lambda_handler(event, context):
    client = boto3.client('sns')
    for record in event['Records']:
        if record['eventName'] == 'INSERT':
            jsonobj = unmarshalDynamoJson(record['dynamodb']['NewImage'])
            response = client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=json.dumps(jsonobj)
            )
            print(response)

if __name__ == '__main__':
    with open('test_event.json') as data_file:    
        event = json.load(data_file)
    lambda_handler(event, 0)