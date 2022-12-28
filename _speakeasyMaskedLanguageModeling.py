import json
import torch
import pickle
import logging
import numpy as np
import time
import os
import sys
from sklearn.utils import shuffle
sys.path.extend(['../..', '.'])
from nebula.models import Cnn1DLinearLM
from nebula.pretraining import SelfSupervisedPretraining, MaskedLanguageModel

# =========== SCRIPT CONFIG ===========

outputFolder = rf"evaluation\MaskedLanguageModeling\output_{int(time.time())}"
train_limit = None #13000
random_state = 42
nSplits = 3

# ===== LOGGING SETUP =====

os.makedirs(outputFolder, exist_ok=True)
# loging setup
logFile = f"PreTrainingEvaluation_randomState_{random_state}_nSplits_{nSplits}_limit_{train_limit}.log"
logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.warning(" [!] Starting Masked Language Model evaluation over {} splits!".format(nSplits))

# =========== LOADING DATA ===========

xTrainFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_10000_maxLen_2048_x.npy"
xTrain = np.load(xTrainFile)
yTrainFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_y.npy"
yTrain = np.load(yTrainFile)
xTestFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset_WithAPIargs\speakeasy_VocabSize_10000_maxLen_2048_x.npy"
xTest = np.load(xTestFile)
yTestFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset_WithAPIargs\speakeasy_y.npy"
yTest = np.load(yTestFile)

if train_limit:
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=random_state)
    xTrain = xTrain[:train_limit]
    yTrain = yTrain[:train_limit]

vocabFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_10000.pkl"
with open(vocabFile, 'rb') as f:
    vocab = pickle.load(f)

logging.warning(f" [!] Loaded data and vocab. X train size: {xTrain.shape}, X test size: {xTest.shape}, vocab size: {len(vocab)}")

# =========== PRETRAINING CONFIG ===========

modelConfig = {
    "vocabSize": len(vocab),
}
modelClass = Cnn1DLinearLM # only difference is that .pretrain() is implemented
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

languageModelClass = MaskedLanguageModel
languageModelClassConfig = {
    "vocab": vocab,
    "mask_probability": 0.15,
    "random_state": random_state
}
pretrainingConfig = {
    "vocab": vocab,
    "modelClass": modelClass,
    "modelConfig": modelConfig,
    "pretrainingTaskClass": languageModelClass,
    "pretrainingTaskConfig": languageModelClassConfig,
    "device": device,
    "unlabeledDataSize": 0.8,
    "pretraingEpochs": 5,
    "downstreamEpochs": 3,
    "verbosityBatches": 50,
    "batchSize": 256,
    "randomState": random_state,
}

# =========== PRETRAINING RUN ===========

pretrain = SelfSupervisedPretraining(**pretrainingConfig)
metrics = pretrain.runSplits(xTrain, yTrain, xTest, yTest, outputFolder=outputFolder, nSplits=nSplits)
metricsOutFile = f"{outputFolder}/metrics_{languageModelClass.__name__}_nSplits_{nSplits}_limit_{train_limit}.json"
with open(metricsOutFile, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.warning(f" [!] Finished pre-training evaluation over {nSplits} splits! Saved metrics to:\n\t{metricsOutFile}")