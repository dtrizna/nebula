import os
import sys
from pandas import to_datetime
from collections.abc import Iterable
from .plots import plotCounterCountsLineplot

def getRealPath(type="script"):
    idx = 1 if type == "notebook" else 0
    return os.path.dirname(os.path.realpath(sys.argv[idx]))

def filterDictByKeys(dict, key_list):
    return {k: dict[k] for k in key_list if k in dict}

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield 

def flattenList(l):
    return [item for sublist in l for item in sublist]

def fix_random_seed(seed_value=1763):
    """Set seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def splitDataFrameTimeStampToChunks(df, timeFieldName='TimeStamp', chunkSize='5min'):
    df[timeFieldName] = to_datetime(df[timeFieldName])
    df[timeFieldName] = df[timeFieldName].dt.floor(chunkSize)
    return df

def dumpTokenizerFiles(tokenizer, outFolder, vocabSize=""):
    import pickle
    file = f"{outFolder}\\speakeasy_VocabSize_{vocabSize}.pkl"
    tokenizer.dumpVocab(file)
    print("Dumped vocab to {}".format(file))
    
    file = f"{outFolder}\\speakeasy_counter.pkl"
    print("Dumped vocab counter to {}".format(file))
    with open(file, "wb") as f:
        pickle.dump(tokenizer.counter, f)

    file = f"{outFolder}\\speakeasy_counter_plot.png"
    plotCounterCountsLineplot(tokenizer.counter, file)
    print("Dumped vocab counter plot to {}".format(file))

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
