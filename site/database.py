"""
Wrapper function for DynamoDB.

Author: Angad Gill, Wei-Tsung Lin
"""
import time
import uuid
import boto3
from config import DYNAMO_DBNAME

def dynamo_no_context_add_tasks(tasks):
    """
    Add tasks to DynamoDB.

    Parameters
    ----------
    tasks: list(dict)
        List of all task objects.

    Returns
    -------
    dict
        response from MongoDB.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(DYNAMO_DBNAME)
    with table.batch_writer() as batch:
        for task in tasks:
            batch.put_item(Item=task)

def dynamo_no_context_update_task(job_id, task_id, aic, bic, labels, elapsed_time, elapsed_read_time,
                                 elapsed_processing_time):
    """
    Update task object on MongoDB.
    This does not use context object from Flask.

    Parameters
    ----------
    job_id: str
    task_id: int
    aic: float
    bic: float
    labels: list(int)
    elapsed_time: str
        Epoch time converted to str
    elapsed_read_time: str
        Epoch time converted to str
    elapsed_processing_time: str
        Epoch time converted to str

    Returns
    -------
    dict
        response from MongoDB.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(DYNAMO_DBNAME)
    response = table.update_item(
        Key={
            'job_id': job_id,
            'task_id': task_id
        },
        UpdateExpression="set task_status = :task_status, aic = :aic, bic = :bic, labels = :labels, elapsed_time = :elapsed_time, elapsed_read_time = :elapsed_read_time, elapsed_processing_time = :elapsed_processing_time",
        ExpressionAttributeValues={
            ':task_status': 'done',
            ':aic': str(aic),
            ':bic': str(bic),
            ':labels': labels,
            ':elapsed_time': elapsed_time,
            ':elapsed_read_time': elapsed_read_time,
            ':elapsed_processing_time': elapsed_processing_time
        }
    )
    return response

def dynamo_no_context_update_task_status(job_id, task_id, status):
    """
    Update 'task_status' value in task object.
    This does not use context object from Flask.

    Parameters
    ----------
    job_id: str
    task_id: int
    status: str

    Returns
    -------
    dict
        response from MongoDB.
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(DYNAMO_DBNAME)
    response = table.update_item(
        Key={
            'job_id': job_id,
            'task_id': task_id
        },
        UpdateExpression="set task_status = :task_status",
        ExpressionAttributeValues={
            ':task_status': status
        }
    )
    return response