import os
import json
import time
import logging
import numpy as np
from collections import defaultdict
from sklearn.utils import shuffle

import torch
from torch.optim import Adam, AdamW
from torch.nn import BCEWithLogitsLoss

import sys
sys.path.extend(['../..', '.'])
from nebula import ModelTrainer
from nebula.models.attention import TransformerEncoderChunksLM
from nebula.pretraining import MaskedLanguageModel
from nebula.evaluation import SelfSupervisedPretraining
from nebula.misc import get_path, set_random_seed, clear_cuda_cache
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")

random_state = 42
set_random_seed(random_state)

uSize = 0.8
run_name = f"uSize_{uSize}_mask_every_epoch"
modelClass = TransformerEncoderChunksLM

run_config = {
    "unlabeledDataSize": uSize,
    "nSplits": 3,
    "downStreamEpochs": 3,
    "preTrainEpochs": 30,
    "falsePositiveRates": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
    "modelType": modelClass.__name__,
    "train_limit": None,
    "random_state": random_state,
    'batchSize': 64,
    'optim_step_budget': 10000, # 5000 is too small for 30 epochs
    'verbosity_batches': 100,
    "dump_model_every_epoch": True,
    "dump_data_splits": True,
    "remask_epochs": False,
    "mask_probability": 0.15
}

timestamp = int(time.time())
outputFolder = os.path.join(REPO_ROOT, "evaluation", "MaskedLanguageModeling", "pretrain_epoch_analysis",
    f"{run_name}_{timestamp}")
os.makedirs(outputFolder, exist_ok=True)

with open(os.path.join(outputFolder, f"run_config.json"), "w") as f:
    json.dump(run_config, f, indent=4)

# ===== LOGGING SETUP =====
logFile = f"{run_name}.log"

logging.basicConfig(
    filename=os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

# ============ MODEL ============
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

# ======= PRETRAINING =========
languageModelClass = MaskedLanguageModel
languageModelClassConfig = {
    "vocab": vocab, # needs vocab to mask sequences
    "mask_probability": run_config["mask_probability"],
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
    "optim_step_budget": run_config["optim_step_budget"],
    "outputFolder": outputFolder,
    "dump_model_every_epoch": run_config["dump_model_every_epoch"],
    "dump_data_splits": run_config["dump_data_splits"],
    "remask_epochs": run_config["remask_epochs"],
}

# =========== PRETRAINING RUN ===========
msg = f" [!] Initiating Self-Supervised Learning Pretraining\n"
msg += f"     Language Modeling {languageModelClass.__name__}\n"
msg += f"     Model {modelClass.__name__} with config:\n\t{modelConfig}\n"
logging.warning(msg)

pretrain = SelfSupervisedPretraining(**pretrainingConfig)
metrics = pretrain.run_splits(xTrain, yTrain, xTest, yTest, nSplits=run_config['nSplits'], rest=0)

metricsOutFile = f"{outputFolder}/results_full.json"
with open(metricsOutFile, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.warning(f" [!] Finished pre-training evaluation over {run_config['nSplits']} splits! Saved metrics to:\n\t{metricsOutFile}")

# ========= EVALUATING PRE-TRAIN LENGTH UTILITY =========
del pretrain
clear_cuda_cache()

logging.warning(f" [*] Evaluating pre-training length utility -- looping over per-epoch models...")
split_data = [x for x in os.listdir(outputFolder) if x.endswith(".npz")]
split_data = [os.path.join(outputFolder, x) for x in split_data]

pretrainFileFolder = os.path.join(outputFolder, "preTraining", "training_files")
pretrainModelFiles = [x for x in os.listdir(pretrainFileFolder) if x.startswith("epoch_")]
pretrainModelFiles = [os.path.join(pretrainFileFolder, x) for x in pretrainModelFiles]

modelTrainerConfig = {
            "device": device,
            "model": None,
            "lossFunction": BCEWithLogitsLoss(),
            "optimizerClass": AdamW,
            "optimizerConfig": {"lr": 2.5e-4},
            "optim_scheduler": "step",
            "optim_step_budget": run_config["optim_step_budget"],
            "verbosityBatches": run_config["verbosity_batches"],
            "batchSize": run_config["batchSize"],
            "falsePositiveRates": run_config["falsePositiveRates"],
        }

epochs = len(pretrainModelFiles)//len(split_data)
tprs, f1s, aucs = defaultdict(list), defaultdict(list), defaultdict(list)
for split in range(len(split_data)):
    loaded = np.load(split_data[split])
    labeled_x = loaded['labeled_x']
    labeled_y = loaded['labeled_y']
    models = []
    for epoch in range(1,epochs+1):
        logging.warning(f" [!] Evaluating model from split: {split} | epoch: {epoch}")
        model_file = [x for x in pretrainModelFiles if f"epoch_{epoch}" in x][split]
        model = modelClass(**modelConfig)
        model.load_state_dict(torch.load(model_file))
        
        # finetune model
        modelTrainerConfig['model'] = model
        modelTrainer = ModelTrainer(**modelTrainerConfig)
        modelTrainer.fit(labeled_x, labeled_y, epochs=run_config['downStreamEpochs'])

        _, tpr, f1, auc = modelTrainer.evaluate(xTest, yTest)
        tprs[epoch].append(tpr)
        f1s[epoch].append(f1)
        aucs[epoch].append(auc)

        del modelTrainer
        clear_cuda_cache()

results = {}

# create a dict from tprs and f1s means and std, and dump as json
for epoch in range(1, epochs+1):
    tprs_mean = np.nanmean(np.array(tprs[epoch]), axis=0)
    tprs_std = np.nanstd(np.array(tprs[epoch]), axis=0)
    f1s_mean = np.nanmean(np.array(f1s[epoch]), axis=0)
    f1s_std = np.nanstd(np.array(f1s[epoch]), axis=0)

    results[epoch] = {}
    # zip to dict values in tprs_mean with false positive rates
    results[epoch]['tpr_mean'] = dict(zip(run_config["falsePositiveRates"], tprs_mean.tolist()))
    results[epoch]['tpr_std'] = dict(zip(run_config["falsePositiveRates"], tprs_std.tolist()))
    results[epoch]['f1_mean'] = dict(zip(run_config["falsePositiveRates"], f1s_mean.tolist()))
    results[epoch]['f1_std'] = dict(zip(run_config["falsePositiveRates"], f1s_std.tolist()))
    results[epoch]['auc_mean'] = np.nanmean(aucs[epoch], axis=0)
    results[epoch]['auc_std'] = np.nanstd(aucs[epoch], axis=0)

with open(os.path.join(outputFolder, "results_per_epoch.json"), "w") as f:
    json.dump(results, f, indent=4)
