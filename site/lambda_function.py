import uuid
import sys
import csv
from create_job import create_tasks
from utils import read_csv, col_select

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

if __name__ == '__main__':
    s3_file_key = sys.argv[1]

    covars = ['full-tied']
    columns = ['Dimension 1', 'Dimension 2']
    submit(1, 1, 1, covars, columns, True, s3_file_key)