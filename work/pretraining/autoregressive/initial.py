import os
basepath = os.path.dirname(os.path.abspath(__file__))
import sys
REPO_ROOT = os.path.join(basepath, "..", "..", "..")
sys.path.extend([REPO_ROOT, "."])

import json
import logging
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from nebula.pretraining import AutoRegressiveModel
from nebula.models.attention import TransformerEncoderChunksLM

# ============ DATA ============
LIMIT = 100
x = np.load(os.path.join(basepath, "x_train_full.npy"))[:LIMIT]

with open(os.path.join(basepath, "tokenizer_50000_vocab.json"), 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab) # adjust it to exact number of tokens in the vocabulary

logging.warning(f" [!] Loaded data and vocab. X size: {x.shape}, vocab size: {len(vocab)}")

# ============ MODEL ============
model_config = {
    "vocab_size": vocab_size,  # size of vocabulary
    "maxlen": x.shape[1],  # maximum length of the input sequence
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "dropout": 0.3
}
model = TransformerEncoderChunksLM(**model_config)

# ============ PRE-TRAINING ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_trainer_config = {
    "device": device,
    "model": model,
    "modelForwardPass": model.pretrain,
    "loss_function": CrossEntropyLoss(),
    "optimizer_class": AdamW,
    "optimizer_config": {"lr": 3e-4},
    "optim_scheduler": None,
    "optim_step_budget": None,
    "verbosity_n_batches": 10,
    "batchSize": 8,
}

pretraining_task = AutoRegressiveModel(
    block_size=256
)
pretraining_task.pretrain(x, model_trainer_config, epochs=1)
