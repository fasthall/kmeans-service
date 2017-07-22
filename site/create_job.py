"""
Author: Angad Gill, Wei-Tsung Lin
"""
import uuid
import sys
import csv
import time
from utils import read_csv, col_select, float_to_str
from database import dynamo_no_context_add_tasks

def lambda_handler(event, context):
    if 'n_init' not in event:
        return 'n_init needed'
    if 'n_exp' not in event:
        return 'n_exp needed'
    if 'max_k' not in event:
        return 'max_k needed'
    if 'covars' not in event:
        return 'covars needed'
    if 'columns' not in event:
        return 'columns needed'
    if 'scale' not in event:
        return 'scale needed'
    if 's3_file_key' not in event:
        return 's3_file_key needed'
    n_init = event['n_init']
    n_exp = event['n_exp']
    max_k = event['max_k']
    covars = event['covars']
    columns = event['columns']
    scale = event['scale']
    s3_file_key = event['s3_file_key']
    return submit(n_init, n_exp, max_k, covars, columns, scale, s3_file_key)

def submit(n_init, n_exp, max_k, covars, columns, scale, s3_file_key):
    job_id = str(uuid.uuid4())
    create_tasks(job_id, n_init, n_exp, max_k, covars, columns, s3_file_key, scale)
    return job_id

def create_tasks(job_id, n_init, n_experiments, max_k, covars, columns, s3_file_key, scale):
    """
    Creates all the tasks needed to complete a job. Adds database entries for each task and triggers an asynchronous
    functions to process the task.

    Parameters
    ----------
    job_id: str
    n_init: int
    n_experiments: int
    max_k: int
    covars: list(str)
    columns: list(str)
    s3_file_key: str
    scale: bool

    Returns
    -------
    None
    """
    task_status = 'pending'
    created_time = float_to_str(time.time())

    # Add tasks to DynamoDB
    task_id = 0
    tasks = []
    for _ in range(n_experiments):
        for k in range(1, max_k + 1):
            for covar in covars:
                covar_type, covar_tied = covar.lower().split('-')
                covar_tied = covar_tied == 'tied'
                task = dict(job_id = job_id,
                    task_id = task_id,
                    k = k,
                    covar_type = covar_type,
                    covar_tied = covar_tied,
                    n_init = n_init,
                    s3_file_key = s3_file_key,
                    columns = columns,
                    scale = scale,
                    task_status = task_status,
                    created_time = created_time)
                tasks += [task]
                task_id += 1
    dynamo_no_context_add_tasks(tasks)
    print("job created: " + job_id)

if __name__ == '__main__':
    s3_file_key = sys.argv[1]

    covars = ["full-tied", "full-untied", "diag-tied", "diag-untied", "spher-tied", "spher-untied"]
    columns = ['Dimension 1', 'Dimension 2']
    submit(1, 3, 1, covars, columns, True, s3_file_key)
