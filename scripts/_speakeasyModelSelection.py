import numpy as np
import torch
import logging
import os
import sys
import time
from sklearn.utils import shuffle
sys.path.extend(['..', '.'])
from nebula import ModelTrainer
from nebula.models import Cnn1DLinear, Cnn1DLSTM, LSTM
from nebula.attention import TransformerEncoderModel, ReformerLM
from nebula.evaluation import CrossValidation
from nebula.misc import get_path

# supress UndefinedMetricWarning, which appears when a batch has only one class
import warnings
warnings.filterwarnings("ignore")

# ============== SCRIPT CONFIG
train_limit = None
runType = "_modelSelection_50k"
runName = f"LSTM"

# ============== Cross-Valiation CONFIG
nFolds = 2
epochs = 3
fprValues = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
rest = 30 # seconds to rest between folds (to cool down)
metricFilePrefix = "" # if vocabSize loop - it is set later, within the loop
random_state = 42

vocabSize = 50000
maxLen = 2048

# ============== REPORTING CONFIG

SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..")
outputFolder = os.path.join(REPO_ROOT, "evaluation", "crossValidation", runType)
os.makedirs(outputFolder, exist_ok=True)
outputFolder = os.path.join(outputFolder, runName)
os.makedirs(outputFolder, exist_ok=True)

# loging setup
logFile = f"CrossValidationRun_{metricFilePrefix}{int(time.time())}.log"
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# =============== MODEL CONFIG

# modelClass = Cnn1DLinear
# modelArch = {
#     "vocabSize": vocabSize,
#     "embeddingDim": 64,
#     "hiddenNeurons": [512, 256, 128],
#     "batchNormConv": False,
#     "batchNormFFNN": False,
#     "filterSizes": [2, 3, 4, 5],
#     "dropout": 0.3
# }
# modelClass = TransformerEncoderModel
# modelArch = {
#     "vocabSize": vocabSize,  # size of vocabulary
#     "maxLen": maxLen,  # maximum length of the input sequence
#     "dModel": 64,  # embedding & transformer dimension
#     "nHeads": 8,  # number of heads in nn.MultiheadAttention
#     "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
#     "nLayers": 8,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#     "numClasses": 1, # binary classification
#     "hiddenNeurons": [64],
#     "layerNorm": False,
#     "dropout": 0.3
# }
# modelClass = ReformerLM
# modelArch = {
#     "vocabSize": vocabSize,
#     "maxLen": maxLen,
#     "dim": 64,
#     "heads": 4,
#     "depth": 4,
#     "meanOverSequence": True,
#     "dropout": 0.3,
# }
modelClass = LSTM
modelArch = {
    "vocabSize": vocabSize,
    "embeddingDim": 64,
    "lstmHidden": 256,
    "lstmLayers": 1,
    "lstmDropout": 0.1, # if > 0, need lstmLayers > 1
    "lstmBidirectional": True,
    "hiddenNeurons": [256, 128],
    "batchNormFFNN": False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelInterfaceClass = ModelTrainer
modelInterfaceConfig = {
    "device": device,
    "model": None, # will be set later
    "lossFunction": torch.nn.BCEWithLogitsLoss(),
    "optimizerClass": torch.optim.Adam,
    "optimizerConfig": {"lr": 2.5e-4},
    "optimSchedulerClass": "step",
    "step_size": 1000,
    "outputFolder": None, # will be set later
    "batchSize": 32,
    "verbosityBatches": 100
}

# =============== CROSS-VALIDATION LOOP

# 1. CROSS VALIDATING OVER DIFFERENT VOCABULARY AND PADDING SIZES
# trainSetPath = os.path.join(REPO_ROOT, "data", "data_filtered", f"speakeasy_trainset_{runType}")
# trainSetsFiles = sorted([x for x in os.listdir(trainSetPath) if x.endswith("_x.npy")])
# y_train_orig = np.load(os.path.join(trainSetPath, "speakeasy_y.npy"))

# existingRuns = [x for x in os.listdir(outputFolder) if x.endswith(".json")]

# for i, file in enumerate(trainSetsFiles):
#     x_train = np.load(os.path.join(trainSetPath, file))

#     vocabSize = int(file.split("_")[2])
#     modelArch["vocabSize"] = vocabSize
#     maxLen = int(file.split("_")[4])
#     metricFilePrefix = f"maxLen_{maxLen}_"
#     existingRunPrefix = f"maxLen_{maxLen}_vocabSize_{vocabSize}_"
#     logging.warning(f" [!] Starting valiation of file {i}/{len(trainSetsFiles)}: {file}")
    
#     # what to skip
#     if maxLen not in [2048] or vocabSize not in [10000, 15000]:
#         logging.warning(f" [!] Skipping {existingRunPrefix} as it is not in the list of parameters to test")
#         continue

#     # skip if already exists
#     if any([x for x in existingRuns if existingRunPrefix in x]):
#         logging.warning(f" [!] Skipping {existingRunPrefix} as it already exists")
#         continue
    
#     logging.warning(f" [!] Running Cross Validation with vocabSize: {vocabSize} | maxLen: {maxLen}")

# 2. CROSS VALIDATION OVER DIFFERENT MODEL CONFIGRATIONS
x_train = np.load(os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_50k", f"speakeasy_VocabSize_{vocabSize}_maxLen_{maxLen}_x.npy"))
y_train = np.load(os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_50k", "speakeasy_y.npy"))

# for heads in [2, 4, 8, 16]:
#     modelArch["nHeads"] = heads
# for dim in [16, 32]:
#     modelArch["dModel"] = dim
# for dHidden in [64, 128, 192, 256]:
#     modelArch["dHidden"] = dHidden
# for nLayers in [1, 2, 4, 8]:
#     modelArch["nLayers"] = nLayers
#for hiddenLayer in [[64]]:
for hiddenLayer in [[256, 64]]:
    modelArch["hiddenNeurons"] = hiddenLayer

    if train_limit:
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]

    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")
    
    # =============== DO Cross Validation
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolder)
    cv.run_folds(x_train, y_train, epochs=epochs, folds=nFolds, fprValues=fprValues)
    cv.dump_metrics(prefix=metricFilePrefix)
    cv.log_avg_metrics()

    if rest:
        time.sleep(rest)
