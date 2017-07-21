"""
Misc. utility functions for formatting, data wrangling, and plotting.

Author: Angad Gill, Wei-Tsung Lin
"""
import os
import csv
import boto3
import shutil

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