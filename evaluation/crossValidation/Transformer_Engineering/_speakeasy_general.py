import numpy as np
import logging
import os
import json
import time
from sklearn.utils import shuffle

import sys
sys.path.extend(['../../..', '.'])
from nebula import ModelTrainer
from nebula.models.attention import TransformerEncoderChunks
from nebula.evaluation import CrossValidation
from nebula.misc import get_path, set_random_seed,clear_cuda_cache

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, AdamW, SGD

# supress UndefinedMetricWarning, which appears when a batch has only one class
import warnings
warnings.filterwarnings("ignore")
clear_cuda_cache()

SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..", "..")

# ============== REPORTING CONFIG
LIMIT = None # 5000
EPOCHS = 6
maxlen = 512
modelClass = TransformerEncoderChunks
random_state = 1763

runType = f"Transformer_Engineering"
timestamp = int(time.time())
runName = f"n_batches_grad_update_limit_{LIMIT}_{timestamp}"

# TODO 1: how to deal with scheduler and time budget?
# TODO 2: add time budget to cross-validation class
# time budget
time_budget = 10 # minutes

# scheduler budget
batch_size = 64
trainset_size_in_3_validation_split = LIMIT if LIMIT else 50750 # 50750 is the size of the trainset in 3 validation split
optim_step_budget = ((trainset_size_in_3_validation_split//batch_size) * EPOCHS) + 50 # 50 extra steps to stabilize schedulers

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

run_config = {
    "train_limit": LIMIT,
    "nFolds": 3,
    "epochs": EPOCHS,
    "fprValues": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    "rest": 0,
    "maxlen": maxlen,
    "batchSize": batch_size,
    "optim_step_budget": optim_step_budget,
    "random_state": random_state,
    "chunk_size": 64,
    "verbosity_n_batches": 100,
    "clip_grad_norm": 1.0,
    "norm_first": True,
    "n_batches_grad_update": [1, 8, 32, 64],
    "optim_scheduler": None,
    "time_budget": int(time_budget*60)
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
model_config = {
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
    "mean_over_sequence": False,
    "norm_first": run_config['norm_first'],
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
    "loss_function": BCEWithLogitsLoss(),
    "optimizer_class": AdamW,
    "optimizer_config": {"lr": 2.5e-4},
    "optim_scheduler": run_config["optim_scheduler"],
    "optim_step_budget": run_config["optim_step_budget"],
    "outputFolder": None, # will be set later
    "batchSize": run_config["batchSize"],
    "verbosity_n_batches": run_config["verbosity_n_batches"],
    "clip_grad_norm": run_config["clip_grad_norm"],
    "n_batches_grad_update": run_config["n_batches_grad_update"],
    "time_budget": run_config["time_budget"],
}

# =============== CROSS-VALIDATION LOOP
# CROSS VALIDATION OVER DIFFERENT MODEL CONFIGRATIONS
logging.warning(f" [!] Using device: {device} | Dataset size: {len(x_train)}")

# GRADIENT CLIPS RUN
# for grad_clips in [None, 0.1, 0.5, 1]:
#     metricFilePrefix =f"grad_clip_{grad_clips}"
#     model_trainer_config["clip_grad_norm"] = grad_clips 
#     logging.warning(f" [!] Starting valiation of model: {modelClass.__name__} with grad clip: {grad_clips}")

# OPTIMIZER RUN
# optim_dict = {"adamw": AdamW, "adam": Adam}
# for optim_class in optim_dict:
#     metricFilePrefix =f"optimizier_{optim_class}"
#     model_trainer_config['optimizer_class'] = optim_dict[optim_class]
#     logging.warning(f" [!] Starting valiation of model: {modelClass.__name__} with optimizer: {optim_class}")

# PRE/POST LAYER NORM RUN
# for norm in run_config["norm_first"]:
#     metricFilePrefix = f"norm_first_{norm}"
#     model_config["norm_first"] = norm
#     logging.warning(f" [!] Starting valiation of model: {modelClass.__name__} with layer input normalization: {norm}")

# BATCHES GRAD UPDATE RUN
for n_batches_grad_update in run_config["n_batches_grad_update"]:
    metricFilePrefix = f"n_batches_grad_update_{n_batches_grad_update}"
    model_trainer_config["n_batches_grad_update"] = n_batches_grad_update
    logging.warning(f" [!] Starting valiation of model: {modelClass.__name__} with grad. update after n. batches: {n_batches_grad_update}")

# LEARNING RATE SCHEDULES
# for optim_scheduler in run_config["optim_scheduler"]:
#     metricFilePrefix = f"optim_scheduler_{optim_scheduler}"
#     model_trainer_config["optim_scheduler"] = optim_scheduler
#     logging.warning(f" [!] Starting valiation of model: {modelClass.__name__} with optimizer scheduler: {optim_scheduler}")

    set_random_seed(random_state)
    cv = CrossValidation(model_trainer_class, model_trainer_config, modelClass, model_config, outputFolder)
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
