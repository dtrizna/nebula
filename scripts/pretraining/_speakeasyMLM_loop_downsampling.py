import json
import torch
import logging
import numpy as np
import time
import os
import sys
from sklearn.utils import shuffle
sys.path.extend(['../..', '.'])
from nebula.models.attention import TransformerEncoderChunksLM
from nebula.pretraining import MaskedLanguageModel
from nebula.evaluation import SelfSupervisedPretraining
from nebula.misc import get_path, set_random_seed, clear_cuda_cache
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")

# ===== LOGGING SETUP =====
outputFolder = os.path.join(REPO_ROOT, "evaluation", "MaskedLanguageModeling", "downsample_tests")
os.makedirs(outputFolder, exist_ok=True)

timestamp = int(time.time())
logFile = f"downsample_{timestamp}.log"
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

# ==== uSize Loop =======
for downsample_unlabeled_data in [0.9]:# [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    logging.warning(f" [!] Starting uSize downsample {downsample_unlabeled_data} evaluation!")
    random_state = 43
    set_random_seed(random_state)

    modelClass = TransformerEncoderChunksLM
    run_name = f"downsample_U_{downsample_unlabeled_data}"
    timestamp = int(time.time())

    outputFolder = os.path.join(outputFolder,  f"{run_name}_{timestamp}")
    os.makedirs(outputFolder, exist_ok=True)

    run_config = {
        "unlabeledDataSize": 0.9, #uSize,
        "nSplits": 3,
        "downStreamEpochs": 3,
        "preTrainEpochs": 10,
        "falsePositiveRates": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
        "modelType": modelClass.__name__,
        "train_limit": None,
        "random_state": random_state,
        "batchSize": 64,
        "optimizerStep": 0, #5000,
        "verbosity_batches": 100,
        "downsample_unlabeled_data": downsample_unlabeled_data, #False
        "training_types": ['pretrained']
    }
    with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    logging.warning(f" [!] Starting Masked Language Model evaluation over {run_config['nSplits']} splits!")

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
    modelConfig = {
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
        json.dump(modelConfig, f, indent=4)

    languageModelClass = MaskedLanguageModel
    languageModelClassConfig = {
        "vocab": vocab, # needs vocab to mask sequences
        "mask_probability": 0.15,
        "random_state": random_state,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrainingConfig = {
        "modelClass": modelClass,
        "modelConfig": modelConfig,
        "pretrainingTaskClass": languageModelClass,
        "pretrainingTaskConfig": languageModelClassConfig,
        "device": device,
        "unlabeledDataSize": run_config["unlabeledDataSize"],
        "pretraingEpochs": run_config["preTrainEpochs"],
        "downstreamEpochs": run_config["downStreamEpochs"],
        "verbosityBatches": run_config["verbosity_batches"],
        "batchSize": run_config["batchSize"],
        "randomState": random_state,
        "falsePositiveRates": run_config["falsePositiveRates"],
        "optimizerStep": run_config["optimizerStep"],
        "outputFolder": outputFolder,
        "dump_data_splits": True,
        "downsample_unlabeled_data": run_config["downsample_unlabeled_data"],
        "training_types": run_config["training_types"]
    }

    # =========== PRETRAINING RUN ===========
    msg = f" [!] Initiating Self-Supervised Learning Pretraining\n"
    msg += f"     Language Modeling {languageModelClass.__name__}\n"
    msg += f"     Model {modelClass.__name__} with config:\n\t{modelConfig}\n"
    logging.warning(msg)

    pretrain = SelfSupervisedPretraining(**pretrainingConfig)
    metrics = pretrain.run_splits(xTrain, yTrain, xTest, yTest, nSplits=run_config['nSplits'], rest=0)
    metricsOutFile = f"{outputFolder}/metrics_{languageModelClass.__name__}_nSplits_{run_config['nSplits']}_limit_{run_config['train_limit']}.json"
    with open(metricsOutFile, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.warning(f" [!] Finished pre-training evaluation over {run_config['nSplits']} splits! Saved metrics to:\n\t{metricsOutFile}\n\n")

    del pretrain
    clear_cuda_cache()
