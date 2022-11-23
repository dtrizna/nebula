import os
import sys
from pandas import to_datetime

from .constants import AUDITD_FIELDS

def getScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def getNotebookPath():
    return os.path.dirname(os.path.realpath(sys.argv[1]))

def filterDictByKeys(dict, key_list=AUDITD_FIELDS):
    return {k: dict[k] for k in key_list if k in dict}

def splitDataFrameTimeStampToChunks(df, timeFieldName='TimeStamp', chunkSize='5min'):
    df[timeFieldName] = to_datetime(df[timeFieldName])
    df[timeFieldName] = df[timeFieldName].dt.floor(chunkSize)
    return df

def isolationForestAnomalyDetctions(arr):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(max_samples=100, random_state=42)
    clf.fit(arr)
    return clf.predict(arr)