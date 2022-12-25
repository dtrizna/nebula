import sys
sys.path.append('../..')
from nebula.models import Cnn1DLinearLM
from nebula.pretraining import SSLPretraining
from sklearn.utils import shuffle
import logging
import pickle
import numpy as np 
import torch

train_limit = 13000

xTrainFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_10000_maxLen_2048_x.npy"
xTrain = np.load(xTrainFile)
yTrainFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_y.npy"
yTrain = np.load(yTrainFile)
xTestFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset_WithAPIargs\speakeasy_VocabSize_10000_maxLen_2048_x.npy"
xTest = np.load(xTestFile)
yTestFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset_WithAPIargs\speakeasy_y.npy"
yTest = np.load(yTestFile)

if train_limit:
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=42)
    xTrain = xTrain[:train_limit]
    yTrain = yTrain[:train_limit]


vocabFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_10000.pkl"
with open(vocabFile, 'rb') as f:
    vocab = pickle.load(f)

logging.warning(f" [!] Loaded data and vocab. X train size: {xTrain.shape}, X test size: {xTest.shape}, vocab size: {len(vocab)}")

modelConfig = {
    "vocabSize": len(vocab)
}
modelClass = Cnn1DLinearLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device=device, pretrinEpochs=1, downstreamEpochs=1, verbosityBatches=100, batchSize=256
# converth previous line to a dict
config = {
    "device": device,
    "pretraingEpochs": 1,
    "downstreamEpochs": 1,
    "verbosityBatches": 100,
    "batchSize": 256,
    "verbosityBatches": 10
}

pretrain = SSLPretraining(xTrain, yTrain, xTest, yTest, vocab, modelClass, modelConfig, **config)
metrics = pretrain.run(xTrain, yTrain, xTest, yTest, vocab, modelClass, modelConfig, **config)

fpr = 0.01
print("\nFPR: ", fpr)
for m in metrics:
    print(f"{metrics[m][fpr]['tpr']:.2f} +- {metrics[m][fpr]['tpr_std']:.2f} -- {m}")

fpr = 0.001
print("\nFPR: ", fpr)
for m in metrics:
    print(f"{metrics[m][fpr]['tpr']:.2f} +- {metrics[m][fpr]['tpr_std']:.2f} -- {m}")
import pdb;pdb.set_trace()