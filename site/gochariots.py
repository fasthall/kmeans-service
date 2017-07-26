import requests
import json
from fnvhash import fnv1a_64

host = ""

class Record:
    """Record consists of key-value pairs and casual dependency"""
    def __init__(self, seed):
        self.tags = {}
        self.seed = seed
        self.hash = 0

    def setHash(self, hash):
        self.hash = hash

    def add(self, key, value):
        self.tags[key] = value

    def toJSON(self):
        r = {"seed": self.seed, "prehash": self.hash, "tags": self.tags}
        return r

def setHost(_host):
    global host
    if not _host.startswith("http://"):
        _host = "http://" + _host
    host = _host

def post(record):
    if host == "":
        print("Host not set yet")
        return
    return requests.post(host + "/record", json = record.toJSON())

def getHash(record):
    result = []
    for k in record.tags:
        result.append(fnv1a_64((k + ":" + record.tags[k]).encode()))
    return result