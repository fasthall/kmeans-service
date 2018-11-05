"""
Misc. utility functions for formatting, data wrangling, and plotting.

Author: Angad Gill, Wei-Tsung Lin
"""
import os
import csv
import time
import boto3
import shutil
import random
from config import S3_BUCKET
import pandas as pd

def float_to_str(num):
    """
    Convert float to str with 4 decimal places.

    Parameters
    ----------
    num: float

    Returns
    -------
    str

    """
    return '{:.4f}'.format(num)

def read_csv(fname):
    data = []
    with open(fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        attr = next(spamreader)
        for row in spamreader:
            tmp = []
            for r in row:
                tmp.append(float(r))
            data.append(tmp)
    return data, attr

def read_from_s3(job_id, task_id, bucket, key):
    path = '/tmp/' + str(job_id) + '/' + str(task_id)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, path + '/data.csv')
    return read_csv(path + '/data.csv')

def col_select(data, attr, columns):
    ndata = []
    for row in data:
        r = []
        for i in range(len(row)):
            if attr[i] in columns:
                r.append(row[i])
        ndata.append(r)
    return ndata

def unmarshalDynamoJson(node):
    data = {}
    data["M"] = node
    return unmarshalDynamoValue(data, True)

def unmarshalDynamoValue(node, mapAsObject):
    for key, value in node.items():
        if key == 'S':
            return value
        elif key == 'N':
            return int(value)
        elif key == 'L':
            data = []
            for item in value:
                data.append(unmarshalDynamoValue(item, mapAsObject))
            return data
        elif key == 'M':
            if mapAsObject:
                data = {}
                for key1, value1 in value.items():
                    data[key1] = unmarshalDynamoValue(value1, mapAsObject)
                return data
        elif key == 'BOOL':
            return value

def format_date_time(epoch_time):
    """
    Converts epoch time string to (Date, Time) formatted as ('04 April 2017', '11:01 AM').
    Parameters
    ----------
    epoch_time: str
        Epoch time converted to str.
    Returns
    -------
    (str, str)
        (Date, Time) formatted as ('04 April 2017', '11:01 AM')
    """
    start_time = time.localtime(float(epoch_time))
    start_time_date = time.strftime("%d %B %Y", start_time)
    start_time_clock = time.strftime("%I:%M %p", start_time)
    return start_time_date, start_time_clock


def filter_dict_list_by_keys(dict_list, keys):
    """
    Keep only keys specified in the function parameter `keys`. Does not modify dicts in `dict_list`.
    Parameters
    ----------
    dict_list: list(dict)
    keys: list(str)
    Returns
    -------
    list(dict)
    """
    new_dict_list = []
    for d in dict_list:
        new_d = {}
        for k, v in d.items():
            if k in keys:
                new_d[k] = v
        new_dict_list += [new_d]
    return new_dict_list

def s3_to_df(s3_file_key):
    """
    Downloads file from S3 and converts it to a Pandas DataFrame. Deletes the file from local disk when done.

    Parameters
    ----------
    s3_file_key: str
        Eucalyptus S3 file key

    Returns
    -------
    Pandas DataFrame
    """
    # Add random number to file name to avoid collisions with other processes on the same machine
    filename = '/tmp/{}_{}'.format(s3_file_key.replace('/', '_'), random.randint(1, 1e6))

    """ Amazon S3 code """
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET, s3_file_key, filename)

    """ Eucalyptus S3 code """
    # s3conn = boto.connect_walrus(aws_access_key_id=EUCA_KEY_ID, aws_secret_access_key=EUCA_SECRET_KEY, is_secure=False,
    #                              port=8773, path=EUCA_S3_PATH, host=EUCA_S3_HOST)
    # euca_bucket = s3conn.get_bucket(S3_BUCKET)
    # k = boto.s3.key.Key(bucket=euca_bucket, name=s3_file_key)
    # k.get_contents_to_filename(filename)

    df = pd.read_csv(filename)
    os.remove(filename)
    return df