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

train_limit = None # 13000
random_state = 42
logFile = f"PreTrainingEvaluation_randomState_{random_state}_limit_{train_limit}.log"

nSplits = 3
downStreamEpochs = 3
preTrainEpochs = 7
unlabeledDataSize = 0.8
falsePositiveRates = [0.003, 0.01, 0.03, 0.1]
outputFolder = rf"evaluation\MaskedLanguageModeling\deeperCnn_unlabeledDataSize_{unlabeledDataSize}_preTrain_{preTrainEpochs}_downStream_{downStreamEpochs}_nSplits_{nSplits}_{int(time.time())}"

# ===== LOGGING SETUP =====

os.makedirs(outputFolder, exist_ok=True)
# loging setup
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
    "filterSizes": [2, 3, 4, 5],
    "numFilters": [192, 192, 192, 192],
    "hiddenNeurons": [768, 512, 256, 128]
}
modelClass = Cnn1DLinearLM # only difference is that .pretrain() is implemented
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

languageModelClass = MaskedLanguageModel
languageModelClassConfig = {
    "vocab": vocab,
    "mask_probability": 0.15,
    "random_state": random_state,
}
pretrainingConfig = {
    "vocab": vocab,
    "modelClass": modelClass,
    "modelConfig": modelConfig,
    "pretrainingTaskClass": languageModelClass,
    "pretrainingTaskConfig": languageModelClassConfig,
    "device": device,
    "unlabeledDataSize": unlabeledDataSize,
    "pretraingEpochs": preTrainEpochs,
    "downstreamEpochs": downStreamEpochs,
    "verbosityBatches": 50,
    "batchSize": 256,
    "randomState": random_state,
    "falsePositiveRates": falsePositiveRates,
}

# =========== PRETRAINING RUN ===========

pretrain = SelfSupervisedPretraining(**pretrainingConfig)
metrics = pretrain.runSplits(xTrain, yTrain, xTest, yTest, outputFolder=outputFolder, nSplits=nSplits)
metricsOutFile = f"{outputFolder}/metrics_{languageModelClass.__name__}_nSplits_{nSplits}_limit_{train_limit}.json"
with open(metricsOutFile, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.warning(f" [!] Finished pre-training evaluation over {nSplits} splits! Saved metrics to:\n\t{metricsOutFile}")