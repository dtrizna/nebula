import numpy as np
import logging
import os
import json
import time
from sklearn.utils import shuffle

import sys
sys.path.extend(['../..', '.'])
from nebula import ModelTrainer
from nebula.models import Cnn1DLinear, LSTM
from nebula.models.attention import TransformerEncoderModel
from nebula.models.neurlux import NeurLuxModel
from nebula.evaluation import CrossValidation
from nebula.misc import get_path, set_random_seed,clear_cuda_cache

from model import CharEmbedding, CharTransformer, TransformerEncoderModelCls

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, AdamW

# supress UndefinedMetricWarning, which appears when a batch has only one class
import warnings
warnings.filterwarnings("ignore")
clear_cuda_cache()

# ============== REPORTING CONFIG
maxlen = 512
epochs = None
timerun = 5 # min
LIMIT = None
runType = f"./realrun_{int(time.time())}"

SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")
outputFolder = runType # os.path.join(REPO_ROOT, "evaluation", "crossValidation", runType)
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

# transform above variables to dict
run_config = {
    "train_limit": LIMIT,
    "nFolds": 3,
    "epochs": epochs,
    "fprValues": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    "rest": 0,
    "maxlen": maxlen,
    "batchSize": 32,
    # 3000 with 64 batch size will make steps each ~4.5 epoch
    "optim_step_budget": 3000,
    "random_state": random_state,
    "verbosity_n_batches": 100,
    "time_budget": timerun # min
}
with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
    json.dump(run_config, f, indent=4)

# ===== LOADING DATA ==============
xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k_new_v2", f"speakeasy_vocab_size_50000_maxlen_{maxlen}_x.npy")
x_train = np.load(xTrainFile)
yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k_new_v2", "speakeasy_y.npy")
y_train = np.load(yTrainFile)

logging.warning(f" [!] Loaded data. X train size: {x_train.shape}, y train size: {y_train.shape}")

if run_config['train_limit']:
    x_train, y_train = shuffle(x_train, y_train, random_state=random_state)
    x_train = x_train[:run_config['train_limit']]
    y_train = y_train[:run_config['train_limit']]

vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", "bpe_50000_sentencepiece_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab) # adjust it to exact number of tokens in the vocabulary

logging.warning(f" [!] Loaded data and vocab. X train size: {x_train.shape}, vocab size: {len(vocab)}")

# =============== MODEL CONFIG
models = []

modelClass = TransformerEncoderModelCls
modelArch = {
    "vocab_size": len(vocab),
    "maxlen": 512,
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "layerNorm": False,
    "dropout": 0.3,
    "norm_first": True
}
models.append((modelClass, modelArch))

modelClass = CharTransformer
modelArch = {
    "patch_size": 16,
    "in_chans": 1,
    "embed_dim": 128,
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "hiddenNeurons": [512, 256, 128],
    "transformer_layers": 4,
}
models.append((modelClass, modelArch))

modelClass = Cnn1DLinear
modelArch = {
    "vocab_size": vocab_size,
    "maxlen": maxlen,
    "embeddingDim": 96,
    "hiddenNeurons": [512, 256, 128],
    "batchNormConv": False,
    "batchNormFFNN": False,
    "filterSizes": [2, 3, 4, 5],
    "dropout": 0.3
}
models.append((modelClass, modelArch))

# modelClass = LSTM
# modelArch = {
#     "vocab_size": vocab_size,
#     "embeddingDim": 64,
#     "lstmHidden": 256,
#     "lstmLayers": 1,
#     "lstmDropout": 0.1, # if > 0, need lstmLayers > 1
#     "lstmBidirectional": True,
#     "hiddenNeurons": [512, 256, 128],
#     "batchNormFFNN": False,
# }
# models.append((modelClass, modelArch))

modelClass = TransformerEncoderModel
modelArch = {
    "vocab_size": vocab_size,  # size of vocabulary
    "maxlen": maxlen,  # maximum length of the input sequence
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "layerNorm": False,
    "dropout": 0.3
}
models.append((modelClass, modelArch))

modelClass = NeurLuxModel
modelArch = {
    "embedding_dim": 256,
    "vocab_size": vocab_size,
    "seq_len": run_config["maxlen"],
}
models.append((modelClass, modelArch))

# ==== TRAINER CONFIG =====
device = "cuda" if cuda.is_available() else "cpu"
modelInterfaceClass = ModelTrainer
modelInterfaceConfig = {
    "device": device,
    "model": None, # will be set later within CrossValidation class
    "loss_function": BCEWithLogitsLoss(),
    "optimizer_class": AdamW,
    "optimizer_config": {"lr": 2.5e-4},
    "optim_scheduler": None, # "step",
    "optim_step_budget": run_config["optim_step_budget"],
    "outputFolder": None, # will be set later
    "batchSize": None, # run_config["batchSize"],
    "verbosity_n_batches": run_config["verbosity_n_batches"],
    "time_budget": run_config["time_budget"] * 60, # convert to seconds
}

# ====== CROSS-VALIDATION LOOP =========
start = int(time.time())
for modelClass, modelArch in models:
    modelName = modelClass.__name__
    logging.warning(f"\n[!!!] Starting evaluation of {modelName}")
    modelInterfaceConfig["batchSize"] = run_config["batchSize"]
    logging.warning(" [!] Using batch size: {}".format(modelInterfaceConfig["batchSize"]))
    set_random_seed(random_state)

    timestamp = int(time.time())
    runName = f"{modelName}_{timestamp}"
    outputFolderSub = os.path.join(outputFolder, runName)
    os.makedirs(outputFolderSub, exist_ok=True)

    # dump model config as json
    with open(os.path.join(outputFolderSub, f"model_config.json"), "w") as f:
        json.dump(modelArch, f, indent=4)

    # ======== ACTUALLY DO CV =========
    cv = CrossValidation(modelInterfaceClass, modelInterfaceConfig, modelClass, modelArch, outputFolderSub)
    cv.run_folds(
        x_train, 
        y_train, 
        epochs=run_config["epochs"], 
        folds=run_config["nFolds"], 
        false_positive_rates=run_config["fprValues"], 
        random_state=random_state
    )
    cv.dump_metrics()
    cv.log_avg_metrics()
    del cv, modelClass, modelArch
    clear_cuda_cache()

    if run_config["rest"]:
        time.sleep(run_config["rest"])

end = int(time.time())
if run_config["train_limit"]:
    dataset_size = 72e3
    estimate = (end-start-(len(models)*run_config['rest']))*(dataset_size/run_config['train_limit'])/60
    logging.warning(f"Estimate full running time: {estimate}min")
