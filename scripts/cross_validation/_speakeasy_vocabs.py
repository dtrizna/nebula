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
runType = f"BPE_50k"
vocab_size = 50000
DATA_FOLDER = os.path.join(REPO_ROOT, "data", "data_filtered", f"speakeasy_trainset_{runType}")
modelClass = TransformerEncoderChunks

timestamp = int(time.time())

runName = f"norm_first_AdamW_limit_{LIMIT}_{timestamp}"
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
    "maxlen": None,
    "batchSize": 64,
    # 1000-2000 for 10 epochs; 10000 for 30 epochs
    "optim_step_budget": 3000,
    "random_state": random_state,
    "chunk_size": 64,
    "verbosity_batches": 100,
    "clip_grad_norm": [None, 0.1, 0.5, 1]
}
with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
    json.dump(run_config, f, indent=4)

# =============== MODEL CONFIG
model_config = {
    "vocab_size": None,  # size of vocabulary
    "maxlen": None,  # maximum length of the input sequence
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

# dump model config as json
with open(os.path.join(outputFolder, f"model_config.json"), "w") as f:
    json.dump(model_config, f, indent=4)

# =============== MODEL INTERFACE CONFIG
device = "cuda" if cuda.is_available() else "cpu"
model_trainer_class = ModelTrainer
model_trainer_config = {
    "device": device,
    "model": None, # will be set later within CrossValidation class
    "lossFunction": BCEWithLogitsLoss(),
    "optimizerClass": AdamW,
    "optimizerConfig": {"lr": 2.5e-4},
    "optim_scheduler": "step",
    "optim_step_budget": run_config["optim_step_budget"],
    "outputFolder": None, # will be set later
    "batchSize": run_config["batchSize"],
    "verbosityBatches": run_config["verbosity_batches"],
    "clip_grad_norm": None
}

# =============== CROSS-VALIDATION LOOP

# CROSS VALIDATING OVER DIFFERENT VOCABULARY AND PADDING SIZES
trainSetsFiles = sorted([x for x in os.listdir(DATA_FOLDER) if x.endswith("_x.npy")])
y_train = np.load(os.path.join(DATA_FOLDER, "speakeasy_y.npy"))

existingRuns = [x for x in os.listdir(outputFolder) if x.endswith(".json")]

for i, file in enumerate(trainSetsFiles):
    logging.warning(f" [!] Starting valiation of file {i}/{len(trainSetsFiles)}: {file}")
    
    x_train = np.load(os.path.join(DATA_FOLDER, file))

    if run_config['train_limit']:
        x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
        x_train = x_train[:run_config['train_limit']]
        y_train = y_train[:run_config['train_limit']]

    vocab_size = int(file.split("_")[2])
    model_config["vocab_size"] = vocab_size
    model_config["vocab_size"] = vocab_size
    
    maxlen = int(file.split("_")[4])
    model_config["maxlen"] = maxlen

    vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", f"speakeasy_BPE_{vocab_size}_vocab.json")
    with open(vocabFile, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab) # adjust it to exact number of tokens in the vocabulary

    existingRunPrefix = f"maxlen_{maxlen}_vocab_size_{vocab_size}_"
    # skip if already exists
    if any([x for x in existingRuns if existingRunPrefix in x]):
        logging.warning(f" [!] Skipping {existingRunPrefix} as it already exists")
        continue
    
    # =============== DO Cross Validation
    logging.warning(f" [!] Starting valiation of model: {modelClass.__name__}")
    logging.warning(f" [!] Running Cross Validation with vocab_size: {vocab_size} | maxlen: {maxlen}")
    logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")
    
    cv = CrossValidation(model_trainer_class, model_trainer_config, modelClass, model_config, outputFolder)
    cv.run_folds(
        x_train, 
        y_train, 
        epochs=run_config["epochs"], 
        folds=run_config["nFolds"], 
        fprValues=run_config["fprValues"], 
        random_state=random_state
    )
    cv.dump_metrics(prefix=existingRunPrefix)
    cv.log_avg_metrics()

    if run_config["rest"]:
        time.sleep(run_config["rest"])
