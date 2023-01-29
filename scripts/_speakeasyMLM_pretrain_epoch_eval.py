import json
import torch
import logging
import numpy as np
import time
import os
import sys
from sklearn.utils import shuffle
sys.path.extend(['../..', '.'])
from nebula.attention import TransformerEncoderLM
from nebula.pretraining import MaskedLanguageModel
from nebula.evaluation import SelfSupervisedPretraining
from nebula.misc import get_path, set_random_seed, clear_cuda_cache
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..")

# ===== LOGGING SETUP =====
logFile = f"uSize_evaluation.log"
logging.basicConfig(
    filename=os.path.join(SCRIPT_PATH, "..", "evaluation", "MaskedLanguageModeling", logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# ==== uSize Loop =======
for uSize in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]:
    random_state = 42
    set_random_seed(random_state)

    modelClass = TransformerEncoderLM
    #run_name = modelClass.__name__
    run_name = f"uSize_{uSize}"
    timestamp = int(time.time())

    outputFolder = os.path.join(SCRIPT_PATH, "..", "evaluation", "MaskedLanguageModeling", 
        f"{run_name}_{timestamp}")
    os.makedirs(outputFolder, exist_ok=True)

    run_config = {
        "unlabeledDataSize": uSize,
        "nSplits": 3,
        "downStreamEpochs": 3,
        "preTrainEpochs": 10,
        "falsePositiveRates": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
        "modelType": modelClass.__name__,
        "train_limit": None,
        "random_state": random_state,
        "batchSize": 64,
        "optimizerStep": 5000,
        'verbosity_batches': 100,
        "dump_model_every_epoch": False,
        "dump_data_splits": True,
        "remask_every_epoch": False,
        "mask_probability": 0.15
    }
    with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    logging.warning(f" [!] Starting Masked Language Model evaluation over {run_config['nSplits']} splits!")

    # ===== LOADING DATA ==============

    vocabSize = 50000
    maxLen = 512
    xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
    xTrain = np.load(xTrainFile)
    yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE", "speakeasy_y.npy")
    yTrain = np.load(yTrainFile)
    xTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
    xTest = np.load(xTestFile)
    yTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE", "speakeasy_y.npy")
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
    vocabSize = len(vocab) # adjust it to exact number of tokens in the vocabulary

    logging.warning(f" [!] Loaded data and vocab. X train size: {xTrain.shape}, X test size: {xTest.shape}, vocab size: {len(vocab)}")

    # =========== PRETRAINING CONFIG ===========
    modelConfig = {
        "vocabSize": vocabSize,  # size of vocabulary
        "maxLen": maxLen,  # maximum length of the input sequence
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
        "verbosityBatches": 100,
        "batchSize": run_config["batchSize"],
        "randomState": random_state,
        "falsePositiveRates": run_config["falsePositiveRates"],
        "optimizerStep": run_config["optimizerStep"],
        "outputFolder": outputFolder,
        "dump_model_every_epoch": run_config["dump_model_every_epoch"],
        "dump_data_splits": run_config["dump_data_splits"],
        "remask_every_epoch": run_config["remask_every_epoch"],
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
    logging.warning(f" [!] Finished pre-training evaluation over {run_config['nSplits']} splits! Saved metrics to:\n\t{metricsOutFile}")

    del pretrain
    clear_cuda_cache()
