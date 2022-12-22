import numpy as np
import torch
import logging
import os
import sys
import time
from sklearn.utils import shuffle
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelInterface
from nebula.attention import TransformerEncoderModel
from nebula.evaluation import CrossValidation

runName = f"Cnn1DLinear_VocabSize_maxLen"
outputFolder = os.path.join(r"C:\Users\dtrizna\Code\nebula\evaluation\crossValidation", runName)
os.makedirs(outputFolder, exist_ok=True)

# loging setup
logFile = f"CrossValidationRun_{int(time.time())}.log"
logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# =============== Cross-Valiation CONFIG
train_limit = None # 5000
nFolds = 3
epochs = 3
fprValues = [0.0001, 0.001, 0.01, 0.1]
rest = 30 # seconds to rest between folds (to cool down)

# ================ MODEL CONFIG

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
#     "nTokens": 1500,  # size of vocabulary
#     "dModel": 192,  # embedding & transformer dimension
#     "nHeads": 2,  # number of heads in nn.MultiheadAttention
#     "dHidden": 200,  # dimension of the feedforward network model in nn.TransformerEncoder
#     "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#     "numClasses": 1, # binary classification
#     "hiddenNeurons": [64],
#     "layerNorm": False,
#     "dropout": 0.5
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
trainSetPath = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset"
trainSetsFiles = sorted([x for x in os.listdir(trainSetPath) if x.endswith("_x.npy")])
y_train = np.load(os.path.join(trainSetPath, "speakeasy_y.npy"))

for file in trainSetsFiles:
    x_train = np.load(os.path.join(trainSetPath, file))
    if train_limit:
        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]

    vocabSize = int(file.split("_")[2])
    modelArch["vocabSize"] = vocabSize
    maxLen = int(file.split("_")[4])

    logging.warning(f" [!] VocabSize: {vocabSize} | MaxLen: {maxLen}")
    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")
    
    # =============== DO Cross Validation
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolder)
    cv.run_folds(x_train, y_train, epochs=epochs, folds=nFolds, fprValues=fprValues)
    cv.dump_metrics(prefix=f"maxLen_{maxLen}_")
    cv.log_avg_metrics()

    if rest:
        time.sleep(rest)
