import numpy as np
import logging
import os
import json
import time
from sklearn.utils import shuffle

import sys
sys.path.extend(['..', '.'])
from nebula import ModelTrainer
from nebula.models import Cnn1DLinear, Cnn1DLSTM, LSTM
from nebula.models.attention import TransformerEncoderModel, TransformerEncoderChunks, TransformerEncoderModelLM
from nebula.models.reformer import ReformerLM
from nebula.evaluation import CrossValidation
from nebula.misc import get_path, set_random_seed,clear_cuda_cache

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, AdamW

# supress UndefinedMetricWarning, which appears when a batch has only one class
import warnings
warnings.filterwarnings("ignore")
clear_cuda_cache()

SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")

# ============== REPORTING CONFIG
LIMIT = None
EPOCHS = 6
maxlen = 512
modelClass = TransformerEncoderChunks

runType = f"Transformer_Engineering"
timestamp = int(time.time())

runName = f"norm_first_AdamW_GradClip_limit_{LIMIT}_{timestamp}"
outputFolder = os.path.join(REPO_ROOT, "evaluation", "crossValidation", runType, runName)
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
random_state = 1763
set_random_seed(random_state)

# transform above variables to dict
run_config = {
    "train_limit": LIMIT,
    "nFolds": 3,
    "epochs": EPOCHS,
    "fprValues": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    "rest": 0,
    "maxlen": maxlen,
    "batchSize": 64,
    # 1000-2000 for 10 epochs; 10000 for 30 epochs
    "optim_step_size": 3000,
    "random_state": random_state,
    "chunk_size": 64,
    "verbosity_batches": 100
}
with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
    json.dump(run_config, f, indent=4)

# ===== LOADING DATA ==============
vocab_size = 50000
xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", f"speakeasy_vocab_size_{vocab_size}_maxlen_{maxlen}_x.npy")
x_train = np.load(xTrainFile)
yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", "speakeasy_y.npy")
y_train = np.load(yTrainFile)

if run_config['train_limit']:
    x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
    x_train = x_train[:run_config['train_limit']]
    y_train = y_train[:run_config['train_limit']]

vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", f"speakeasy_BPE_{vocab_size}_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab) # adjust it to exact number of tokens in the vocabulary

logging.warning(f" [!] Loaded data and vocab. X train size: {x_train.shape}, vocab size: {len(vocab)}")

# =============== MODEL CONFIG
# modelClass = Cnn1DLinear
# modelArch = {
#     "vocab_size": vocab_size,
#     "maxlen": maxlen,
#     "embeddingDim": 64,
#     "hiddenNeurons": [512, 256, 128],
#     "batchNormConv": False,
#     "batchNormFFNN": False,
#     "filterSizes": [2, 3, 4, 5],
#     "dropout": 0.2
# }

#modelClass = TransformerEncoderChunks
modelArch = {
    "vocab_size": vocab_size,  # size of vocabulary
    "maxlen": maxlen,  # maximum length of the input sequence
    "chunk_size": run_config["chunk_size"],
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "layerNorm": False,
    "dropout": 0.3,
    "mean_over_sequence": False
}

# modelClass = ReformerLM
# modelArch = {
#     "vocab_size": vocab_size,
#     "maxlen": maxlen,
#     "dim": 64,
#     "heads": 4,
#     "depth": 4,
#     "meanOverSequence": True,
#     "classifierDropout": 0.3,
# }

# modelClass = LSTM
# modelArch = {
#     "vocab_size": vocab_size,
#     "embeddingDim": 64,
#     "lstmHidden": 256,
#     "lstmLayers": 1,
#     "lstmDropout": 0.1, # if > 0, need lstmLayers > 1
#     "lstmBidirectional": True,
#     "hiddenNeurons": [256, 128],
#     "batchNormFFNN": False,
# }

# dump model config as json
with open(os.path.join(outputFolder, f"model_config.json"), "w") as f:
    json.dump(modelArch, f, indent=4)

device = "cuda" if cuda.is_available() else "cpu"
modelInterfaceClass = ModelTrainer
modelInterfaceConfig = {
    "device": device,
    "model": None, # will be set later within CrossValidation class
    "lossFunction": BCEWithLogitsLoss(),
    "optimizerClass": AdamW,
    "optimizerConfig": {"lr": 2.5e-4},
    "optim_scheduler": "step",
    "optim_step_size": run_config["optim_step_size"],
    "outputFolder": None, # will be set later
    "batchSize": run_config["batchSize"],
    "verbosityBatches": run_config["verbosity_batches"],
}

# =============== CROSS-VALIDATION LOOP

# 1. CROSS VALIDATING OVER DIFFERENT VOCABULARY AND PADDING SIZES
# trainSetPath = os.path.join(REPO_ROOT, "data", "data_filtered", f"speakeasy_trainset_{runType}")
# trainSetsFiles = sorted([x for x in os.listdir(trainSetPath) if x.endswith("_x.npy")])
# y_train_orig = np.load(os.path.join(trainSetPath, "speakeasy_y.npy"))

# existingRuns = [x for x in os.listdir(outputFolder) if x.endswith(".json")]

# for i, file in enumerate(trainSetsFiles):
#     x_train = np.load(os.path.join(trainSetPath, file))

#     vocab_size = int(file.split("_")[2])
#     modelArch["vocab_size"] = vocab_size
#     maxlen = int(file.split("_")[4])
#     metricFilePrefix = f"maxlen_{maxlen}_"
#     existingRunPrefix = f"maxlen_{maxlen}_vocab_size_{vocab_size}_"
#     logging.warning(f" [!] Starting valiation of file {i}/{len(trainSetsFiles)}: {file}")
    
#     # what to skip
#     if maxlen not in [2048] or vocab_size not in [10000, 15000]:
#         logging.warning(f" [!] Skipping {existingRunPrefix} as it is not in the list of parameters to test")
#         continue

#     # skip if already exists
#     if any([x for x in existingRuns if existingRunPrefix in x]):
#         logging.warning(f" [!] Skipping {existingRunPrefix} as it already exists")
#         continue
    
#     logging.warning(f" [!] Running Cross Validation with vocab_size: {vocab_size} | maxlen: {maxlen}")

# 2. CROSS VALIDATION OVER DIFFERENT MODEL CONFIGRATIONS

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

    logging.warning(f" [!] Starting valiation of model: {modelClass.__name__}")
    # to differentiate different run results -- use this
    metricFilePrefix = f"" 

    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")
    
    # =============== DO Cross Validation
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolder)
    cv.run_folds(
        x_train, 
        y_train, 
        epochs=run_config["epochs"], 
        folds=run_config["nFolds"], 
        fprValues=run_config["fprValues"], 
        random_state=random_state
    )
    cv.dump_metrics(prefix=metricFilePrefix)
    cv.log_avg_metrics()

    if run_config["rest"]:
        time.sleep(run_config["rest"])
