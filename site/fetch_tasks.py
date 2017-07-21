import json
import boto3
from utils import *

def lambda_handler(event, context):
    client = boto3.client('sns')
    for record in event['Records']:
        if record['eventName'] == 'INSERT':
            jsonobj = unmarshalDynamoJson(record['dynamodb']['NewImage'])
            response = client.publish(
                TopicArn='arn:aws:sns:us-west-2:185174815983:kmeansservice',
                Message=json.dumps(jsonobj)
            )
            print(response)

if __name__ == '__main__':
    with open('test_event.json') as data_file:    
        event = json.load(data_file)
    lambda_handler(event, 0)