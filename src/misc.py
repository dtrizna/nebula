import os
import sys
from pandas import to_datetime

def getRealPath(type="script"):
    idx = 1 if type == "notebook" else 0
    return os.path.dirname(os.path.realpath(sys.argv[idx]))

def filterDictByKeys(dict, key_list):
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

def json_unnormalize(df, sep="."):
    """
    The opposite of json_normalize
    """
    result = []
    for _, row in df.iterrows():
        parsed_row = {}
        for col_label,v in row.items():
            keys = col_label.split(sep)

            current = parsed_row
            for i, k in enumerate(keys):
                if i==len(keys)-1:
                    current[k] = v
                else:
                    if k not in current.keys():
                        current[k] = {}
                    current = current[k]
        # save
        result.append(parsed_row)
    return result
