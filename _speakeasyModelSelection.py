import numpy as np
import torch
import logging
import os
import sys
import time
from sklearn.utils import shuffle
sys.path.extend(['..', '.'])
from nebula import ModelInterface
from nebula.models import Cnn1DLinear, Cnn1DLSTM, LSTM
from nebula.attention import TransformerEncoderModel
from nebula.evaluation import CrossValidation

# ============== SCRIPT CONFIG
train_limit = None
runName = f"Cnn1DLinear_VocabSize_maxLen"
runType = "WithError"

# ============== Cross-Valiation CONFIG
nFolds = 3
epochs = 3
fprValues = [0.0001, 0.001, 0.01, 0.1]
rest = 30 # seconds to rest between folds (to cool down)
metricFilePrefix = "" # is set later, within the loop
random_state = 42

# ============== REPORTING CONFIG

outputFolder = rf"C:\Users\dtrizna\Code\nebula\evaluation\crossValidation\{runType}"
os.makedirs(outputFolder, exist_ok=True)
outputFolder = os.path.join(outputFolder, runName)
os.makedirs(outputFolder, exist_ok=True)

# loging setup
logFile = f"CrossValidationRun_{int(time.time())}.log"
logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# =============== MODEL CONFIG

modelClass = Cnn1DLinear
modelArch = {
    "vocabSize": None,
    "embeddingDim": 64,
    "hiddenNeurons": [512, 256, 128],
    "batchNormConv": False,
    "batchNormFFNN": False,
    "filterSizes": [2, 3, 4, 5]
}
# modelClass = TransformerEncoderModel
# modelArch = {
#     "vocabSize": vocabSize,  # size of vocabulary
#     "dModel": 32,  # embedding & transformer dimension
#     "nHeads": 8,  # number of heads in nn.MultiheadAttention
#     "dHidden": 128,  # dimension of the feedforward network model in nn.TransformerEncoder
#     "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#     "numClasses": 1, # binary classification
#     "hiddenNeurons": [64],
#     "layerNorm": False,
#     "dropout": 0.5
# }
# modelClass = Cnn1DLSTM
# modelArch = {
#     "vocabSize": None,
#     "embeddingDim": 64,
#     "lstmHidden": 128,
#     "lstmLayers": 1,
#     "lstmDropout": 0, # if > 0, need lstmLayers > 1
#     "lstmBidirectional": True,
#     "batchNormConv": False,
#     "hiddenNeurons": [256, 128],
#     "batchNormFFNN": False,
# }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelInterfaceClass = ModelInterface
modelInterfaceConfig = {
    "device": device,
    "model": None, # will be set later
    "lossFunction": torch.nn.BCEWithLogitsLoss(),
    "optimizerClass": torch.optim.Adam,
    "optimizerConfig": {"lr": 0.001},
    "outputFolder": None, # will be set later
    "verbosityBatches": 100
}

# =============== CROSS-VALIDATION LOOP

# 1. CROSS VALIDATING OVER DIFFERENT VOCABULARY AND PADDING SIZES
trainSetPath = rf"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_{runType}"
trainSetsFiles = sorted([x for x in os.listdir(trainSetPath) if x.endswith("_x.npy")])
y_train_orig = np.load(os.path.join(trainSetPath, "speakeasy_y.npy"))

existingRuns = [x for x in os.listdir(outputFolder) if x.endswith(".json")]

for i, file in enumerate(trainSetsFiles):
    x_train = np.load(os.path.join(trainSetPath, file))

    vocabSize = int(file.split("_")[2])
    modelArch["vocabSize"] = vocabSize
    maxLen = int(file.split("_")[4])
    metricFilePrefix = f"maxLen_{maxLen}_"
    existingRunPrefix = f"maxLen_{maxLen}_vocabSize_{vocabSize}_"
    logging.warning(f" [!] Starting valiation of file {i}/{len(trainSetsFiles)}: {file}")
    
    # what to skip
    if maxLen not in [2048] or vocabSize not in [10000, 15000]:
        logging.warning(f" [!] Skipping {existingRunPrefix} as it is not in the list of parameters to test")
        continue

    # skip if already exists
    if any([x for x in existingRuns if existingRunPrefix in x]):
        logging.warning(f" [!] Skipping {existingRunPrefix} as it already exists")
        continue
    
    logging.warning(f" [!] Running Cross Validation with vocabSize: {vocabSize} | maxLen: {maxLen}")

# 2. CROSS VALIDATION OVER DIFFERENT MODEL CONFIGRATIONS
# vocabSize = 10000
# maxLen = 2048
# x_train = np.load(rf"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_VocabSize_{vocabSize}_maxLen_{maxLen}_x.npy")
# y_train = np.load(r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset_WithAPIargs\speakeasy_y.npy")

# for heads in [2, 4, 8, 16]:
#     modelArch["nHeads"] = heads
# for dim in [16, 32]:
#     modelArch["dModel"] = dim
# for dHidden in [64, 128, 192, 256]:
#     modelArch["dHidden"] = dHidden
# for nLayers in [1, 2, 4, 8]:
#     modelArch["nLayers"] = nLayers
# for hiddenLayer in [[32], [64], [128], [32, 16], [64, 32, 16]]:
#     modelArch["hiddenNeurons"] = hiddenLayer

    if train_limit:
        x_train, y_train = shuffle(x_train, y_train_orig, random_state=random_state)
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
    else:
        y_train = y_train_orig

    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")
    
    # =============== DO Cross Validation
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolder)
    cv.run_folds(x_train, y_train, epochs=epochs, folds=nFolds, fprValues=fprValues)
    cv.dump_metrics(prefix=metricFilePrefix)
    cv.log_avg_metrics()

    if rest:
        time.sleep(rest)
