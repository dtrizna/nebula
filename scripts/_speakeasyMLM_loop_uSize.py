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
outputFolder = os.path.join(SCRIPT_PATH, "..", "evaluation", "MaskedLanguageModeling", "uSize_loop")
os.makedirs(outputFolder, exist_ok=True)

timestamp = int(time.time())
logFile = f"uSize_evaluation_{timestamp}.log"
logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

# ==== uSize Loop =======
for uSize in [0.7, 0,75, 0.8, 0.85, 0.9, 0.95, 0.97]:
    logging.warning(f" [!] Starting uSize {uSize} evaluation!")
    random_state = 1763
    set_random_seed(random_state)

    modelClass = TransformerEncoderLM
    run_name = f"uSize_{uSize}"
    timestamp = int(time.time())

    outputFolder = os.path.join(outputFolder, f"{run_name}_{timestamp}")
    os.makedirs(outputFolder, exist_ok=True)

    config = {
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
        "training_types": ['pretrained', 'non_pretrained']
    }
    with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    logging.warning(f" [!] Starting Masked Language Model evaluation over {config['nSplits']} splits!")

    if config['train_limit']:
        xTrain, yTrain = shuffle(xTrain, yTrain, random_state=random_state)
        xTrain = xTrain[:config['train_limit']]
        yTrain = yTrain[:config['train_limit']]
        xTest, yTest = shuffle(xTest, yTest, random_state=random_state)
        xTest = xTest[:config['train_limit']]
        yTest = yTest[:config['train_limit']]

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
        "unlabeledDataSize": config["unlabeledDataSize"],
        "pretraingEpochs": config["preTrainEpochs"],
        "downstreamEpochs": config["downStreamEpochs"],
        "verbosityBatches": config["verbosity_batches"],
        "batchSize": config["batchSize"],
        "randomState": random_state,
        "falsePositiveRates": config["falsePositiveRates"],
        "optimizerStep": config["optimizerStep"],
        "outputFolder": outputFolder,
        "dump_data_splits": True,
        "training_types": config["training_types"]
    }

    # =========== PRETRAINING RUN ===========
    msg = f" [!] Initiating Self-Supervised Learning Pretraining\n"
    msg += f"     Language Modeling {languageModelClass.__name__}\n"
    msg += f"     Model {modelClass.__name__} with config:\n\t{modelConfig}\n"
    logging.warning(msg)

    pretrain = SelfSupervisedPretraining(**pretrainingConfig)
    metrics = pretrain.run_splits(xTrain, yTrain, xTest, yTest, nSplits=config['nSplits'], rest=0)
    metricsOutFile = f"{outputFolder}/metrics_{languageModelClass.__name__}_nSplits_{config['nSplits']}_limit_{config['train_limit']}.json"
    with open(metricsOutFile, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.warning(f" [!] Finished pre-training evaluation over {config['nSplits']} splits! Saved metrics to:\n\t{metricsOutFile}")

    del pretrain
    clear_cuda_cache()
