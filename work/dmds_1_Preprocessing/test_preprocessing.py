import os
root = os.path.dirname(os.path.abspath("."))
if 'nebula' in os.listdir(root):
    REPO_ROOT = os.path.join(root, "nebula")
elif 'nebula' in os.listdir(os.path.join(root, "..", "..")):
    REPO_ROOT = os.path.join(root, "..")
else:
    raise ValueError

import sys
sys.path.append(REPO_ROOT)
sys.path.append('.')

import json
import numpy as np
from torch import Tensor
from nebula.models.dmds import DMDSGatedCNN
from nebula.models.dmds.preprocessor import DMDSPreprocessor
from nebula.preprocessing.wrappers import parse_cruparamer_xmlsample

LIMIT = 5

# ================== CRUPARAMER DATA 
CRUPARAMER_DATA_DIR = os.path.join(REPO_ROOT, "data", "data_raw", "CruParamer", "Datacon2019-Malicious-Code-DataSet-Stage1", "train", "black")
CRUPARAMER_SAMPLES = [os.path.join(CRUPARAMER_DATA_DIR, x) for x in os.listdir(CRUPARAMER_DATA_DIR)[:LIMIT]]

X_cruparamer = []
for sample in CRUPARAMER_SAMPLES:
    parser = DMDSPreprocessor(seq_len=1000)
    api_w_args, _ = parse_cruparamer_xmlsample(sample)
    for api_bundle in api_w_args:
        api_name = api_bundle.split(" ")[0]
        args = [x for x in api_bundle.split(" ")[1:] if x]
        parser.parse(api_name, args)
    padded = parser.pad_sequence()
    X_cruparamer.append(padded)
X_cruparamer = np.stack(X_cruparamer)

# ================== SPEAKEASY DATA
SPEAKEASY_DATA_DIR = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_trainset", "report_backdoor")
SPEAKEASY_SAMPLES = [os.path.join(SPEAKEASY_DATA_DIR, x) for x in os.listdir(SPEAKEASY_DATA_DIR)[:LIMIT]]

SPEAKEASY_SAMPLES_JSON = []
for sample in SPEAKEASY_SAMPLES:
    with open(sample) as f:
        SPEAKEASY_SAMPLES_JSON.append(json.load(f))


X_speakeasy = []
for sample in SPEAKEASY_SAMPLES_JSON:
    parser = DMDSPreprocessor()
    sample = sample[0]
    for api in sample['apis']:
        api_name = api['api_name'].split(".")[-1]
        args = api['args']
        parser.parse(api_name, args)
    X_speakeasy.append(parser.pad_sequence())
X_speakeasy = np.stack(X_speakeasy)

def modelpass(x, seq_len=500):
    model = DMDSGatedCNN(ndim=x.shape[2], seq_len=seq_len)
    y = model(Tensor(x))
    return y, y.shape

print(X_cruparamer.shape)
print(modelpass(X_cruparamer, seq_len=1000))

print(X_speakeasy.shape)
print(modelpass(X_speakeasy))
