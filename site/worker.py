"""
Author: Angad Gill, Wei-Tsung Lin
"""
import time
import json
import sklearn_lite as preprocessing
from utils import *
from database import *
from sf_kmeans import sf_kmeans

def lambda_handler(event, context):
    message = event['Records'][0]['Sns']['Message']
    task = json.loads(message)
    job_id = task['job_id']
    task_id = task['task_id']
    k = task['k']
    covar_type = task['covar_type']
    covar_tied = task['covar_tied']
    n_init = task['n_init']
    s3_file_key = task['s3_file_key']
    columns = task['columns']
    scale = task['scale']
    response = work_task(job_id, task_id, k, covar_type, covar_tied, n_init, s3_file_key, columns, scale)
    return {"job_id": job_id, "task_id": task_id, "response": response}

def run_kmeans(data, n_clusters, covar_type, covar_tied, n_init):
    """
    Creates an instance of the `kmeans` object and runs `fit` using the data.

    Parameters
    ----------
    data: Pandas DataFrame
        Data containing only the columns to be used for `fit`
    n_clusters: int
    covar_type: str
    covar_tied: bool
    n_init: int

    Returns
    -------
    float, float, list(int)
        aic, bic, labels
    """
    kmeans = sf_kmeans.SF_KMeans(n_clusters=n_clusters, covar_type=covar_type, covar_tied=covar_tied, n_init=n_init,
                                 verbose=0)
    kmeans.fit(data)
    aic, bic = kmeans.aic(data), kmeans.bic(data)
    labels = [int(l) for l in kmeans.labels_]
    return aic, bic, labels

def work_task(job_id, task_id, k, covar_type, covar_tied, n_init, s3_file_key, columns, scale):
    """
    Performs the processing needed to complete a task. Downloads the task parameters and the file. Runs K-Means `fit`
    and updates the database with results.

    Sets `task_status` in the database to 'done' if completed successfully, else to 'error'.

    Parameters
    ----------
    job_id: str
    task_id: int
    k: int
    covar_type: str
    covar_tied: bool
    n_init: int
    s3_file_key: str
    columns: list(str)
    scale: bool

    Returns
    -------
    str
        'Done'
    """
    try:
        print('job_id:{}, task_id:{}'.format(job_id, task_id))
        start_time = time.time()
        start_read_time = time.time()
        data, attr = read_from_s3(job_id, task_id, s3_file_key)
        elapsed_read_time = time.time() - start_read_time

        start_processing_time = time.time()
        data = col_select(data, attr, columns)
        if scale:
            data = preprocessing.scale(data)
        aic, bic, labels = run_kmeans(data, k, covar_type, covar_tied, n_init)
        elapsed_processing_time = time.time() - start_processing_time

        elapsed_time = time.time() - start_time

        elapsed_time = float_to_str(elapsed_time)
        elapsed_read_time = float_to_str(elapsed_read_time)
        elapsed_processing_time = float_to_str(elapsed_processing_time)
        response = dynamo_no_context_update_task(job_id, task_id, aic, bic, labels, elapsed_time, elapsed_read_time, elapsed_processing_time)
    except Exception as e:
        response = dynamo_no_context_update_task_status(job_id, task_id, 'error')
        raise Exception(e)
    return 'Done'