import os
import sys
import json

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from sklearn.utils import shuffle

# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.misc import fix_random_seed
fix_random_seed(0)

# ==============
# LOAD DATA
# ==============
logging.info(f" [*] Loading data...")
train_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k_new_v2")
xTrainFile = os.path.join(train_folder, f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
xTrain = np.load(xTrainFile)
yTrainFile = os.path.join(train_folder, "speakeasy_y.npy")
yTrain = np.load(yTrainFile)
test_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_testset_BPE_50k_new_v2")
xTestFile = os.path.join(test_folder, f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
xTest = np.load(xTestFile)
yTestFile = os.path.join(test_folder, "speakeasy_y.npy")
yTest = np.load(yTestFile)

vocabFile = os.path.join(train_folder, r"speakeasy_vocab_size_50000_tokenizer_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)
logging.info(f" [*] Loaded data.")

# ===================
# MODELING
# ===================
logging.info(f" [*] Loading model...")
from nebula.models import TransformerEncoderChunks
model_config = {
    "vocab_size": vocab_size,
    "maxlen": 512,
    "chunk_size": 64, # input splitting to chunks
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64], # classifier ffnn dims
    "layerNorm": False,
    "dropout": 0.3,
    "mean_over_sequence": False,
    "norm_first": True
}
model = TransformerEncoderChunks(**model_config)
model_path = os.path.join(REPOSITORY_ROOT, r"nebula\objects\speakeasy_BPE_50000_torch.model")

from torch import load
state_dict = load(model_path)
# clean up from LM layers if present
state_dict = {k: v for k, v in state_dict.items() if "pretrain_layers" not in k}
model.load_state_dict(state_dict)
logging.info(f" [!] Model ready.")

# ===================
# TRAINING
# ===================
from torch import cuda
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from nebula import ModelTrainer

TIME_BUDGET = 5 # minutes
device = "cuda" if cuda.is_available() else "cpu"
model_trainer_config = {
    "device": device,
    "model": model,
    "loss_function": BCEWithLogitsLoss(),
    "optimizer_class": AdamW,
    "optimizer_config": {"lr": 3e-4},
    "optim_scheduler": None,
    "optim_step_budget": None,
    "outputFolder": "out",
    "batchSize": 96,
    "verbosity_n_batches": 100,
    "clip_grad_norm": 1.0,
    "n_batches_grad_update": 1,
    "time_budget": int(TIME_BUDGET*60) if TIME_BUDGET else None,
}

model_trainer = ModelTrainer(**model_trainer_config)
model_trainer.train(xTrain, yTrain)
