import os
import sys
from utils import format_date_time, read_from_s3, filter_dict_list_by_keys
from database import dynamo_get_tasks
import pandas as pd
import numpy as np

S3_BUCKET = os.environ['S3_BUCKET']
EXCLUDE_COLUMNS = ['longitude', 'latitude']  # must be lower case
SPATIAL_COLUMNS = ['longitude', 'latitude']  # must be lower case

def report(job_id, x_axis, y_axis, min_members=None):
    """
    Generate report for a job

    Parameters
    ----------
    job_id: str
    x_axis: str
        Name of column from user dataset to be used for the x axis of the plot
    y_axis: str
        Name of column from user dataset to be used for the y axis of the plot
    min_members: int, optional
        Minimum number of members required in all clusters in an experiment to consider the experiment for the report.

    Returns
    -------
    html
    """
    # job_id is valid
    tasks = dynamo_get_tasks(job_id)
    if len(tasks) == 0:
        return 'No task found'
    n_tasks_done = len([x for x in tasks if x['task_status'] == 'done'])
    if len(tasks) != n_tasks_done:
        return 'All tasks not completed yet for job ID: {} {}/{}'.format(job_id, n_tasks_done, len(tasks))

    # all tasks are done
    if min_members is None:
        min_members = 10
    tasks = filter_by_min_members(tasks, min_members=min_members)
    start_time_date, start_time_clock = format_date_time(tasks[0]['created_time'])

    covar_types, covar_tieds, ks, labels, bics, task_ids = tasks_to_best_results(tasks)

    if x_axis is None or y_axis is None:
        # Visualize the first two columns that are not on the exclude list
        viz_columns = [c for c in job['columns'] if c.lower().strip() not in EXCLUDE_COLUMNS][:2]
    else:
        viz_columns = [x_axis, y_axis]

    data, columns = read_from_s3(job_id, 0, S3_BUCKET, tasks[0]['s3_file_key'])
    spatial_columns = [c for c in columns if c.lower() in SPATIAL_COLUMNS][:2]

    # recommendations for all covariance types
    covar_type_tied_k = {}
    for covar_type in covar_types:
        covar_type_tied_k[covar_type.capitalize()] = {}

    for covar_type, covar_tied, k in zip(covar_types, covar_tieds, ks):
        covar_type_tied_k[covar_type.capitalize()][['Untied', 'Tied'][covar_tied]] = k

    # task_id for all recommended assignments
    covar_type_tied_task_id = {}
    for covar_type in covar_types:
        covar_type_tied_task_id[covar_type.capitalize()] = {}

    for covar_type, covar_tied, task_id in zip(covar_types, covar_tieds, task_ids):
        covar_type_tied_task_id[covar_type.capitalize()][['Untied', 'Tied'][covar_tied]] = task_id

    result = dict(job_id=job_id, min_members=min_members,
                           covar_type_tied_k=covar_type_tied_k, covar_type_tied_task_id=covar_type_tied_task_id,
                           columns=columns, viz_columns=viz_columns, spatial_columns=spatial_columns,
                           start_time_date=start_time_date, start_time_clock=start_time_clock)
    return result

def filter_by_min_members(tasks, min_members=10):
    """
    Keep tasks only if they have at least `min_members` points in each cluster. Does not modify `tasks`.
    Parameters
    ----------
    tasks: list(dict)
        List of task objects for a job.
    min_members: int
    Returns
    -------
    list(dict)
    """
    filtered_tasks = []
    for task in tasks:
        if np.all(np.bincount(task['labels']) > min_members):
            filtered_tasks += [task]
    return filtered_tasks

def tasks_to_best_results(tasks):
    """
    Finds the best values for k, labels, and BIC for all covar_type and covar_tied. 'Best' corresponds to highest BIC
    value.
    Parameters
    ----------
    tasks: list(dict)
    Returns
    -------
    list(str), list(bool), list(int), list(list(int)), list(float)
        list of covar_type strings
        list of covar_tied bools
        list of k values
        list of labels
        list of BIC values
        list of task_id
    """
    # Filter list of dicts to reduce the size of Pandas DataFrame
    df = pd.DataFrame(filter_dict_list_by_keys(tasks, ['task_id', 'k', 'covar_type', 'covar_tied', 'bic', 'job_id']))

    # Subset df to needed columns and fix types
    df['bic'] = df['bic'].astype('float')
    df['k'] = df['k'].astype('int')

    # For each covar_type and covar_tied, find k that has the best (max.) mean bic
    df_best_mean_bic = df.groupby(['covar_type', 'covar_tied', 'k'], as_index=False).mean()
    df_best_mean_bic = df_best_mean_bic.sort_values('bic', ascending=False)
    df_best_mean_bic = df_best_mean_bic.groupby(['covar_type', 'covar_tied'], as_index=False).first()

    # Get labels from df that correspond to a bic closest to the best mean bic
    df = pd.merge(df, df_best_mean_bic, how='inner', on=['covar_type', 'covar_tied', 'k'], suffixes=('_x', '_y'))
    df = df.sort_values('bic_x', ascending=False)
    df = df.groupby(['covar_type', 'covar_tied', 'k'], as_index=False).first()
    labels = []
    for row in df['job_id']:
        labels += [t['labels'] for t in tasks if t['job_id'] == row]
    return df['covar_type'].tolist(), df['covar_tied'].tolist(), df['k'].tolist(), labels, df['bic_x'], \
           df['task_id'].tolist()

def lambda_handler(event, context):
    job_id = event['job_id']
    x_axis = event['x_axis']
    y_axis = event['y_axis']
    if 'min_members' not in event:
        min_members = 10    
    else:
        min_members = event['min_members']
    return report(job_id, x_axis, y_axis, min_members)

if __name__ == '__main__':
    job_id = sys.argv[1]
    print(report(job_id, 'Dimension 1', 'Dimension 2'))