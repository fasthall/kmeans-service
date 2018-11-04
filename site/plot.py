import io
import os
import sys
import boto3
import shutil
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from utils import filter_dict_list_by_keys
from database import dynamo_get_tasks
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

S3_BUCKET = os.environ['S3_BUCKET']

matplotlib.use('Agg')  # needed to ensure that plotting works on a server with no display

def read_from_s3_pd(job_id, task_id, bucket, key):
    path = '/tmp/' + str(job_id) + '/' + str(task_id)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, path + '/data.csv')
    return pd.read_csv(path + '/data.csv')

def write_to_local(data, job_id, file, onLambda=True):
    if onLambda:
        path = '/tmp/' + str(job_id) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path = './imgs/'
        if not os.path.exists(path):
            os.makedirs(path)
    with open(path + file, 'wb') as ofile:
        ofile.write(data)

def write_to_s3(job_id, bucket, file, onLambda=True):
    if onLambda:
        path = '/tmp/' + str(job_id) + '/'
    else:
        path = './imgs/'
    s3 = boto3.client('s3')
    s3.upload_file(path + file, bucket, job_id + '/' + file)

def plot_aic_bic(job_id, min_members=None, onLambda=True):
    """
    Generate the AIC-BIC plot as a PNG
    Parameters
    ----------
    job_id: str
    min_members: int, optional
        Minimum number of members required in all clusters in an experiment to consider the experiment for the report.
    Returns
    -------
    image/png
    """
    tasks = dynamo_get_tasks(job_id)
    if min_members is not None:
        tasks = filter_by_min_members(tasks, min_members)
    fig = plot_aic_bic_fig(tasks)
    aic_bic_plot = fig_to_png(fig)
    response = aic_bic_plot.getvalue()
    write_to_local(response, job_id, 'aic_bic_plot.png', onLambda)
    write_to_s3(job_id, S3_BUCKET, 'aic_bic_plot.png', onLambda)
    return response

def plot_count(job_id, min_members=None, onLambda=True):
    """
    Generate the Count plot as a PNG
    Parameters
    ----------
    job_id: str
    min_members: int, optional
        Minimum number of members required in all clusters in an experiment to consider the experiment for the report.
    Returns
    -------
    image/png
    """
    tasks = dynamo_get_tasks(job_id)
    if min_members is not None:
        tasks = filter_by_min_members(tasks, min_members)
    fig = plot_count_fig(tasks)
    count_plot = fig_to_png(fig)
    response = count_plot.getvalue()
    write_to_local(response, job_id, 'count_plot.png', onLambda)
    write_to_s3(job_id, S3_BUCKET, 'count_plot.png', onLambda)
    return response

def plot_cluster(job_id, x_axis, y_axis, show_ticks, min_members=None, onLambda=True):
    """
    Generate the Cluster plot as a PNG
    Parameters
    ----------
    job_id: str
    x_axis: str
        Name of column from user dataset to be used for the x axis of the plot
    y_axis: str
        Name of column from user dataset to be used for the y axis of the plot
    Returns
    -------
    image/png
    """
    tasks = dynamo_get_tasks(job_id)

    if min_members is not None:
        tasks = filter_by_min_members(tasks, min_members)
    covar_types, covar_tieds, ks, labels, bics, task_ids = tasks_to_best_results(tasks)
    s3_file_key = tasks[0]['s3_file_key']
    viz_columns = [x_axis, y_axis]
    data = read_from_s3_pd(job_id, 0, S3_BUCKET, s3_file_key)
    fig = plot_cluster_fig(data, viz_columns, zip(covar_types, covar_tieds, labels, ks, bics), show_ticks)
    cluster_plot = fig_to_png(fig)
    response = cluster_plot.getvalue()
    write_to_local(response, job_id, 'cluster_plot.png', onLambda)
    write_to_s3(job_id, S3_BUCKET, 'cluster_plot.png', onLambda)
    return response

def plot_correlation(job_id, onLambda=True):
    """
    Generate the Correlation heat map as a PNG
    Parameters
    ----------
    job_id: str
    Returns
    -------
    image/png
    """
    tasks = dynamo_get_tasks(job_id)
    s3_file_key = tasks[0]['s3_file_key']
    data = read_from_s3_pd(job_id, 0, S3_BUCKET, s3_file_key)
    fig = plot_correlation_fig(data)
    correlation_plot = fig_to_png(fig)
    response = correlation_plot.getvalue()
    write_to_local(response, job_id, 'correlation_plot.png', onLambda)
    write_to_s3(job_id, S3_BUCKET, 'correlation_plot.png', onLambda)
    return response

def fig_to_png(fig):
    """ Converts a matplotlib figure to a png (byte stream). """
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    return output

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

def plot_aic_bic_fig(tasks):
    """
    Creates AIC-BIC plot, as a 2-row x 3-col grid of point plots with 95% confidence intervals.
    Parameters
    ----------
    tasks: list(dict)
    Returns
    -------
    Matplotlib Figure object
    """
    sns.set(context='talk', style='whitegrid')
    # Filter list of dicts to reduce the size of Pandas DataFrame
    df = pd.DataFrame(filter_dict_list_by_keys(tasks, ['k', 'covar_type', 'covar_tied', 'bic', 'aic']))
    df['covar_type'] = [x.capitalize() for x in df['covar_type']]
    df['covar_tied'] = [['Untied', 'Tied'][x] for x in df['covar_tied']]
    df['aic'] = df['aic'].astype('float')
    df['bic'] = df['bic'].astype('float')
    df = pd.melt(df, id_vars=['k', 'covar_type', 'covar_tied'], value_vars=['aic', 'bic'], var_name='metric')
    f = sns.factorplot(x='k', y='value', col='covar_type', row='covar_tied', hue='metric', data=df,
                       row_order=['Tied', 'Untied'], col_order=['Full', 'Diag', 'Spher'], legend=True, legend_out=True,
                       ci=95, n_boot=100)
    f.set_titles("{col_name}-{row_name}")
    f.set_xlabels("Num. of Clusters (K)")
    return f.fig


def plot_cluster_fig(data, columns, covar_type_tied_labels_k_bics, show_ticks=True):
    """
    Creates cluster plot for the user data using label assignment provided, as a 2-row x 3-col scatter plot.
    Parameters
    ----------
    data: Pandas DataFrame
        User data file as a Pandas DataFrame
    columns: list(str)
        Column numbers from `data` to use as the x and y axes for the plot. Only the first two elements of the list
        are used.
    covar_type_tied_labels_k_bics: list((str, bool, list(int), int, float))
        [(covar_type, covar_tied, labels, k, bic), ... ]
    show_ticks: bool
        Show or hide tick marks on x and y axes.
    Returns
    -------
    Matplotlib Figure object.
    """
    sns.set(context='talk', style='white')
    columns = columns[:2]

    fig = plt.figure()
    placement = {'full': {True: 1, False: 4}, 'diag': {True: 2, False: 5}, 'spher': {True: 3, False: 6}}

    lim_left = data[columns[0]].min()
    lim_right = data[columns[0]].max()
    lim_bottom = data[columns[1]].min()
    lim_top = data[columns[1]].max()

    covar_type_tied_labels_k_bics = list(covar_type_tied_labels_k_bics)

    bics = [x[4] for x in covar_type_tied_labels_k_bics]
    max_bic = max(bics)

    for covar_type, covar_tied, labels, k, bic in covar_type_tied_labels_k_bics:
        plt.subplot(2, 3, placement[covar_type][covar_tied])
        plt.scatter(data[columns[0]], data[columns[1]], c=labels, cmap=plt.cm.rainbow, s=10)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.xlim(left=lim_left, right=lim_right)
        plt.ylim(bottom=lim_bottom, top=lim_top)
        if show_ticks is False:
            plt.xticks([])
            plt.yticks([])
        title = '{}-{}, K={}\nBIC: {:,.1f}'.format(covar_type.capitalize(), ['Untied', 'Tied'][covar_tied], k, bic)
        if bic == max_bic:
            plt.title(title, fontweight='bold')
        else:
            plt.title(title)
    plt.tight_layout()
    return fig


def plot_correlation_fig(data):
    """
    Creates a correlation heat map for all columns in user data.
    Parameters
    ----------
    data: Pandas DataFrame
        User data file as a Pandas DataFrame
    Returns
    -------
    Matplotlib Figure object.
    """
    sns.set(context='talk', style='white')
    fig = plt.figure()
    sns.heatmap(data.corr(), vmin=-1, vmax=1)
    plt.tight_layout()
    return fig


def plot_count_fig(tasks):
    """
    Create count plot, as a 2-row x 3-col bar plot of data points for each k in each covar.
    Parameters
    ----------
    tasks: list(dict)
    Returns
    -------
    Matplotlib Figure object.
    """
    sns.set(context='talk', style='whitegrid')
    df = pd.DataFrame(filter_dict_list_by_keys(tasks, ['k', 'covar_type', 'covar_tied']))
    df = df.loc[:, ['k', 'covar_type', 'covar_tied', 'bic', 'aic']]
    df['covar_type'] = [x.capitalize() for x in df['covar_type']]
    df['covar_tied'] = [['Untied', 'Tied'][x] for x in df['covar_tied']]
    f = sns.factorplot(x='k', kind='count', col='covar_type', row='covar_tied', data=df,
                      row_order=['Tied', 'Untied'], col_order=['Full', 'Diag', 'Spher'], legend=True, legend_out=True,
                      palette='Blues_d')
    f.set_titles("{col_name}-{row_name}")
    f.set_xlabels("Num. of Clusters (K)")
    return f.fig


def plot_spatial_cluster_fig(data, covar_type_tied_labels_k):
    """ Creates a 3x2 plot spatial plot using labels as the color """
    sns.set(context='talk', style='white')
    data.columns = [c.lower() for c in data.columns]
    fig = plt.figure()
    placement = {'full': {True: 1, False: 4}, 'diag': {True: 2, False: 5}, 'spher': {True: 3, False: 6}}

    lim_left = data['longitude'].min()
    lim_right = data['longitude'].max()
    lim_bottom = data['latitude'].min()
    lim_top = data['latitude'].max()
    for covar_type, covar_tied, labels, k in covar_type_tied_labels_k:
        plt.subplot(2, 3, placement[covar_type][covar_tied])
        plt.scatter(data['longitude'], data['latitude'], c=labels, cmap=plt.cm.rainbow, s=10)
        plt.xlim(left=lim_left, right=lim_right)
        plt.ylim(bottom=lim_bottom, top=lim_top)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('{}-{}, K={}'.format(covar_type.capitalize(), ['Untied', 'Tied'][covar_tied], k))
    plt.tight_layout()
    return fig

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
    plot = event['plot']
    job_id = event['job_id']
    if plot == 'aic_bic':
        plot_aic_bic(job_id)
    elif plot == 'count':
        plot_count(job_id)
    elif plot == 'cluster':
        x_axis = event['x_axis']
        y_axis = event['y_axis']
        show_ticks = event['show_ticks']
        if 'min_members' not in event:
            min_members = 10
        else:
            min_members = event['min_members']
        plot_cluster(job_id, x_axis, y_axis, show_ticks, min_members)
    elif plot == 'correlation':
        plot_correlation(job_id)

if __name__ == '__main__':
    job_id = sys.argv[1]
    plot_aic_bic(job_id, onLambda=False)
    plot_count(job_id, onLambda=False)
    plot_cluster(job_id, 'Dimension 1', 'Dimension 2', True, onLambda=False)
    plot_correlation(job_id, onLambda=False)