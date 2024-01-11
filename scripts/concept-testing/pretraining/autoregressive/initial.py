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

from nebula.pretraining import AutoRegressiveModelTrainer
from nebula.models.attention import TransformerEncoderChunks

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
    "classifier_head": [64],
    "dropout": 0.3,
    "pretrain_layers": [1024]
}
model = TransformerEncoderChunks(**model_config)

# ============ PRE-TRAINING ============
BATCH_SIZE = 8
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
    "batchSize": BATCH_SIZE,
}

pretraining_task = AutoRegressiveModelTrainer(
    vocab=vocab,
    block_size=16,
    model_trainer_config=model_trainer_config,
)
pretraining_task.pretrain(
    x,
    epochs=1,
    random_offsets=32
)
