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
train_folder = "speakeasy_trainset_BPE"
runType = "Transformer_512_BPE"
runName = f"longRun_30_epochs"

# ============== REPORTING CONFIG

SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..")
outputFolder = os.path.join(REPO_ROOT, "evaluation", "crossValidation", runType)
os.makedirs(outputFolder, exist_ok=True)
outputFolder = os.path.join(outputFolder, runName)
os.makedirs(outputFolder, exist_ok=True)

# loging setup
logFile = f"CrossValidationRun_{int(time.time())}.log"
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# ============== Cross-Valiation CONFIG
nFolds = 3
epochs = 30
fprValues = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
rest = 30 # seconds to rest between folds (to cool down the GPU)

vocabSize = 50000
maxLen = 512

batchSize = 16 # 16 for Transformer; 192 for CNNLinear
optim_step_size = 10000 # 1000-2000 for Transformer; 500 for CNNLinear

random_state = 42
metricFilePrefix = f""

logging.warning(f" [!] Cross-Validation config: {vocabSize} vocab size, {maxLen} maxLen, {train_limit} train_limit, {train_folder} train_folder, {random_state} random_state, {batchSize} batchSize, {optim_step_size} optim_step_size")

# =============== MODEL CONFIG

# modelClass = Cnn1DLinear
# modelArch = {
#     "vocabSize": vocabSize,
#     "maxLen": maxLen,
#     "embeddingDim": 64,
#     "hiddenNeurons": [512, 256, 128],
#     "batchNormConv": False,
#     "batchNormFFNN": False,
#     "filterSizes": [2, 3, 4, 5],
#     "dropout": 0.3
# }
modelClass = TransformerEncoderModel
modelArch = {
    "vocabSize": vocabSize,  # size of vocabulary
    "maxLen": maxLen,  # maximum length of the input sequence
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "layerNorm": False,
    "dropout": 0.3
}
# modelClass = ReformerLM
# modelArch = {
#     "vocabSize": vocabSize,
#     "maxLen": maxLen,
#     "dim": 64,
#     "heads": 4,
#     "depth": 4,
#     "meanOverSequence": True,
#     "classifierDropout": 0.3,
# }
# modelClass = LSTM
# modelArch = {
#     "vocabSize": vocabSize,
#     "embeddingDim": 64,
#     "lstmHidden": 256,
#     "lstmLayers": 1,
#     "lstmDropout": 0.1, # if > 0, need lstmLayers > 1
#     "lstmBidirectional": True,
#     "hiddenNeurons": [256, 128],
#     "batchNormFFNN": False,
# }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelInterfaceClass = ModelTrainer
modelInterfaceConfig = {
    "device": device,
    "model": None, # will be set later within CrossValidation class
    "lossFunction": torch.nn.BCEWithLogitsLoss(),
    "optimizerClass": torch.optim.Adam,
    "optimizerConfig": {"lr": 2.5e-4},
    "optimSchedulerClass": "step",
    "optim_step_size": optim_step_size,
    "outputFolder": None, # will be set later
    "batchSize": batchSize,
    "verbosityBatches": 100,
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
x_train = np.load(os.path.join(REPO_ROOT, "data", "data_filtered", train_folder, f"speakeasy_VocabSize_{vocabSize}_maxLen_{maxLen}_x.npy"))
y_train = np.load(os.path.join(REPO_ROOT, "data", "data_filtered", train_folder, "speakeasy_y.npy"))

# for heads in [2, 4, 8, 16]:
#     modelArch["nHeads"] = heads
# for dim in [16, 32]:
#     modelArch["dModel"] = dim
# for dHidden in [64, 128, 192, 256]:
#     modelArch["dHidden"] = dHidden
# for nLayers in [1, 2, 4, 8, 12]:
#     modelArch["nLayers"] = nLayers
# for hiddenLayer in [[256, 64], [512, 256, 64]]:
#     modelArch["hiddenNeurons"] = hiddenLayer
for _ in [1]:
    if train_limit:
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]

    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")
    
    # =============== DO Cross Validation
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolder)
    cv.run_folds(x_train, y_train, epochs=epochs, folds=nFolds, fprValues=fprValues, random_state=random_state)
    cv.dump_metrics(prefix=metricFilePrefix)
    cv.log_avg_metrics()

    if rest:
        time.sleep(rest)
