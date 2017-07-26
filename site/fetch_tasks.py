import os
import json
import boto3
import gochariots
from utils import *

SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']

def lambda_handler(event, context):
    gochariots.setHost(os.environ['GOCHARIOTS_HOST'])

    cnt = 0
    client = boto3.client('sns')
    for record in event['Records']:
        if record['eventName'] == 'INSERT':
            cnt += 1
            jsonobj = unmarshalDynamoJson(record['dynamodb']['NewImage'])

            payload = {'action': 'fetch_tasks', 'entry': jsonobj}
            record = gochariots.Record(jsonobj['seed'])
            record.add('kmeans', json.dumps(payload))
            record.setHash(jsonobj['hash'])
            response = gochariots.post(record)

            hash = gochariots.getHash(record)[0]
            jsonobj['hash'] = hash
            response = client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=json.dumps(jsonobj)
            )
            print(jsonobj['task_id'], jsonobj['task_status'])
        else:
            print('skip', record['eventName'])
    print(cnt, 'tasks dispatched')
    return cnt

if __name__ == '__main__':
    with open('test_event.json') as data_file:    
        event = json.load(data_file)
    lambda_handler(event, 0)