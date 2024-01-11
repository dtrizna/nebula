import os
import json
import time
import logging
import numpy as np
from sklearn.utils import shuffle

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import torch

import sys
sys.path.append("../..")
sys.path.append(".")
from nebula import ModelTrainer
from nebula.models.attention import TransformerEncoderModelLM, PositionalEncoding, TransformerEncoderChunks
from nebula.misc import get_path, set_random_seed, clear_cuda_cache
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")

# ====== LOGGING
logFile = f"CrossValidationRun_{int(time.time())}.log"
logging.basicConfig(
    filename=None, #os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


random_state = 42
set_random_seed(random_state)

run_config = {
    "train_limit": 1000,
    "optim_step_budget": 1000,
    "batch_size": 64,
    "chunk_size": 96
}

# ===== LOADING DATA ==============
vocabSize = 50000
maxLen = 512
xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
xTrain = np.load(xTrainFile)
yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", "speakeasy_y.npy")
yTrain = np.load(yTrainFile)
xTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE_50k", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
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

vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", "bpe_50000_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocabSize = len(vocab) # adjust it to exact number of tokens in the vocabulary


modelArch = {
    "vocabSize": vocabSize,  # size of vocabulary
    "maxLen": maxLen,  # maximum length of the input sequence
    "chunk_size": run_config["chunk_size"],
    "dModel": 96,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "classifier_head": [128],
    "layerNorm": False,
    "dropout": 0.3
}

model = TransformerEncoderChunks(**modelArch)

device = "cuda" if cuda.is_available() else "cpu"
modelInterfaceConfig = {
    "device": device,
    "model": model, 
    "loss_function": BCEWithLogitsLoss(),
    "optimizer_class": AdamW,
    "optimizer_config": {"lr": 2.5e-4},
    "optim_scheduler": "step",
    "optim_step_budget": run_config["optim_step_budget"],
    "outputFolder": None, # will be set later
    "batchSize": run_config["batch_size"],
    "verbosity_n_batches": 100,
}

# ===== TRAINING ==============
import torch

trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(xTrain).long(), torch.from_numpy(yTrain).float()),
            batch_size=run_config["batch_size"],
            shuffle=True
        )

loss = BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=2.5e-4)

for i, (data, target) in enumerate(trainLoader):
    print(f"Batch: {i}")
    logits = model(data)
    loss_value = loss(logits, target.reshape(-1,1))
    print(f"Loss: {loss_value.item():.6f}")
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
