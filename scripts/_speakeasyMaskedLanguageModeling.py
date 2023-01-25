import json
import torch
import logging
import numpy as np
import time
import os
import sys
from sklearn.utils import shuffle
sys.path.extend(['../..', '.'])
from nebula.models import Cnn1DLinearLM
from nebula.attention import ReformerLM, TransformerEncoderLM
from nebula.pretraining import MaskedLanguageModel
from nebula.evaluation import SelfSupervisedPretraining
from nebula.misc import get_path, set_random_seed
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..")

# supress UndefinedMetricWarning, which appears when a batch has only one class
import warnings
warnings.filterwarnings("ignore")

# =========== SCRIPT CONFIG ===========
train_limit = None # 5000
random_state = 42
set_random_seed(random_state)
rest = 0 # seconds to cool down between splits
logFile = f"PreTrainingEvaluation_randomState_{random_state}_limit_{train_limit}.log"

#for unlabeledDataSize in [0.75, 0.8, 0.85, 0.9, 0.95]:
unlabeledDataSize = 0.8
nSplits = 2
downStreamEpochs = 3
preTrainEpochs = 5
falsePositiveRates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
modelType = "TransformerLM"
outputFolder = os.path.join(SCRIPT_PATH, "..", "evaluation", "MaskedLanguageModeling", 
    f"{modelType}_unlabeledDataSize_{unlabeledDataSize}_preTrain_{preTrainEpochs}_downStream_{downStreamEpochs}_nSplits_{nSplits}_{int(time.time())}")

# ===== LOGGING SETUP =====
os.makedirs(outputFolder, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# =========== LOADING DATA ===========
logging.warning(f" [!] Starting Masked Language Model evaluation over {nSplits} splits!")

vocabSize = 50000
maxLen = 512
xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
xTrain = np.load(xTrainFile)
yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE", "speakeasy_y.npy")
yTrain = np.load(yTrainFile)
xTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
xTest = np.load(xTestFile)
yTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE", "speakeasy_y.npy")
yTest = np.load(yTestFile)

if train_limit:
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=random_state)
    xTrain = xTrain[:train_limit]
    yTrain = yTrain[:train_limit]
    xTest, yTest = shuffle(xTest, yTest, random_state=random_state)
    xTest = xTest[:train_limit]
    yTest = yTest[:train_limit]

vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", "speakeasy_BPE_50000_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocabSize = len(vocab) # adjust it to exact number of tokens in the vocabulary

logging.warning(f" [!] Loaded data and vocab. X train size: {xTrain.shape}, X test size: {xTest.shape}, vocab size: {len(vocab)}")

# =========== PRETRAINING CONFIG ===========

# modelConfig = {
#     "vocabSize": len(vocab),
#     "hiddenNeurons": [512, 256, 128]
# }
# modelClass = Cnn1DLinearLM # only difference is that .pretrain() is implemented
# modelConfig = {
#     "vocabSize": len(vocab),
#     "hiddenNeurons": [64]
# }
# modelClass = ReformerLM
modelClass = TransformerEncoderLM
modelConfig = {
    "vocabSize": vocabSize,  # size of vocabulary
    "maxLen": maxLen,  # maximum length of the input sequence
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "dropout": 0.3
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

languageModelClass = MaskedLanguageModel
languageModelClassConfig = {
    "vocab": vocab,
    "mask_probability": 0.15,
    "random_state": random_state,
}
pretrainingConfig = {
    "modelClass": modelClass,
    "modelConfig": modelConfig,
    "pretrainingTaskClass": languageModelClass,
    "pretrainingTaskConfig": languageModelClassConfig,
    "device": device,
    "unlabeledDataSize": unlabeledDataSize,
    "pretraingEpochs": preTrainEpochs,
    "downstreamEpochs": downStreamEpochs,
    "verbosityBatches": 100,
    "batchSize": 64,
    "randomState": random_state,
    "falsePositiveRates": falsePositiveRates,
    "optimizerStep": 10000
}

# =========== PRETRAINING RUN ===========
msg = f" [!] Initiating Self-Supervised Learning Pretraining\n"
msg += f"     Language Modeling {languageModelClass.__name__}\n"
msg += f"     Model {modelClass.__name__} with config:\n\t{modelConfig}\n"
logging.warning(msg)

pretrain = SelfSupervisedPretraining(**pretrainingConfig)
metrics = pretrain.run_splits(xTrain, yTrain, xTest, yTest, outputFolder=outputFolder, nSplits=nSplits, rest=rest)
metricsOutFile = f"{outputFolder}/metrics_{languageModelClass.__name__}_nSplits_{nSplits}_limit_{train_limit}.json"
with open(metricsOutFile, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.warning(f" [!] Finished pre-training evaluation over {nSplits} splits! Saved metrics to:\n\t{metricsOutFile}")
