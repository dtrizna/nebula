import json
import time
import os
import sys
import logging
import numpy as np
from torch import cuda
from sklearn.utils import shuffle
sys.path.extend([r"..\..", '.'])
from nebula.models.attention import TransformerEncoderChunksLM
from nebula.pretraining import MaskedLanguageModelTrainer
from nebula.evaluation import SelfSupervisedPretraining
from nebula.misc import get_path, set_random_seed, clear_cuda_cache
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")

# ===== LOGGING SETUP =====
model_class = TransformerEncoderChunksLM
run_name = f"{model_class.__name__}"
timestamp = int(time.time())

LIMIT = 1000
PREFIX = "TEST_"
outputFolder = os.path.join(REPO_ROOT, "evaluation", f"{PREFIX}MaskedLanguageModeling", 
    f"{run_name}_{timestamp}")
os.makedirs(outputFolder, exist_ok=True)

logFile = f"{run_name}.log"
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# ====== SCRIPT CONFIG =====

random_state = 42
set_random_seed(random_state)

run_config = {
    "unlabeledDataSize": 0.8,
    "nSplits": 3,
    "downStreamEpochs": 3,
    "preTrainEpochs": 3, # 10
    "falsePositiveRates": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    "modelType": model_class.__name__,
    "train_limit": LIMIT,
    "random_state": random_state,
    "batchSize": 64,
    "optim_step_budget": 50, # 5000
    'verbosity_n_batches': 100,
    "dump_model_every_epoch": True,
    "dump_data_splits": True,
    "remask_epochs": 2,
    "mask_probability": 0.15
}
with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
    json.dump(run_config, f, indent=4)

logging.warning(f" [!] Starting Masked Language Model evaluation over {run_config['nSplits']} splits!")

# ===== LOADING DATA ==============
vocab_size = 50000
maxlen = 512
xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", f"speakeasy_vocab_size_50000_maxlen_{maxlen}_x.npy")
xTrain = np.load(xTrainFile)
yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", "speakeasy_y.npy")
yTrain = np.load(yTrainFile)
xTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE_50k", f"speakeasy_vocab_size_50000_maxlen_{maxlen}_x.npy")
xTest = np.load(xTestFile)
yTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE_50k", "speakeasy_y.npy")
yTest = np.load(yTestFile)

if run_config['train_limit']:
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=random_state)
    xTrain = xTrain[:run_config['train_limit']]
    yTrain = yTrain[:run_config['train_limit']]
    xTest, yTest = shuffle(xTest, yTest, random_state=random_state)
    xTest = xTest[:run_config['train_limit']]
    yTest = yTest[:run_config['train_limit']]

vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", "speakeasy_BPE_50000_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab) # adjust it to exact number of tokens in the vocabulary

logging.warning(f" [!] Loaded data and vocab. X train size: {xTrain.shape}, X test size: {xTest.shape}, vocab size: {len(vocab)}")

# =========== PRETRAINING CONFIG ===========
model_config = {
    "vocab_size": vocab_size,  # size of vocabulary
    "maxlen": maxlen,  # maximum length of the input sequence
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "dropout": 0.3
}
# dump model config as json
with open(os.path.join(outputFolder, f"model_config.json"), "w") as f:
    json.dump(model_config, f, indent=4)

languageModelClass = MaskedLanguageModelTrainer
languageModelClassConfig = {
    "vocab": vocab, # needs vocab to mask sequences
    "mask_probability": 0.15,
    "random_state": random_state,
}

device = "cuda" if cuda.is_available() else "cpu"
pretrainingConfig = {
    "modelClass": model_class,
    "modelConfig": model_config,
    "pretrainingTaskClass": languageModelClass,
    "pretrainingTaskConfig": languageModelClassConfig,
    "device": device,
    "unlabeledDataSize": run_config["unlabeledDataSize"],
    "pretraingEpochs": run_config["preTrainEpochs"],
    "downstreamEpochs": run_config["downStreamEpochs"],
    "verbosity_n_batches": run_config["verbosity_n_batches"],
    "batchSize": run_config["batchSize"],
    "randomState": random_state,
    "false_positive_rates": run_config["falsePositiveRates"],
    "optim_step_budget": run_config["optim_step_budget"],
    "outputFolder": outputFolder,
    "dump_model_every_epoch": run_config["dump_model_every_epoch"],
    "dump_data_splits": run_config["dump_data_splits"],
    "remask_epochs": run_config["remask_epochs"],
}

# =========== PRETRAINING RUN ===========
msg = f" [!] Initiating Self-Supervised Learning Pretraining\n"
msg += f"     Language Modeling {languageModelClass.__name__}\n"
msg += f"     Model {model_class.__name__} with config:\n\t{model_config}\n"
logging.warning(msg)

pretrain = SelfSupervisedPretraining(**pretrainingConfig)
metrics = pretrain.run_splits(xTrain, yTrain, xTest, yTest, nSplits=run_config['nSplits'], rest=0)
metricsOutFile = f"{outputFolder}/metrics_{languageModelClass.__name__}_nSplits_{run_config['nSplits']}_limit_{run_config['train_limit']}.json"
with open(metricsOutFile, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.warning(f" [!] Finished pre-training evaluation over {run_config['nSplits']} splits! Saved metrics to:\n\t{metricsOutFile}\n\n")

del pretrain
clear_cuda_cache()
