import os
import sys
import json
from time import time

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from sklearn.metrics import f1_score

# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.data_utils import create_dataloader
from nebula.lit_utils import LitTrainerWrapper
from nebula.misc import fix_random_seed

if __name__ == "__main__":

    fix_random_seed(0)

    # ==============
    # LOAD DATA
    # ==============
    logging.info(f" [*] Loading data...")
    train_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k")
    xTrainFile = os.path.join(train_folder, f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
    x_train = np.load(xTrainFile)
    yTrainFile = os.path.join(train_folder, "speakeasy_y.npy")
    y_train = np.load(yTrainFile)
    test_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", "speakeasy_testset_BPE_50k")
    xTestFile = os.path.join(test_folder, f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
    x_test = np.load(xTestFile)
    yTestFile = os.path.join(test_folder, "speakeasy_y.npy")
    y_test = np.load(yTestFile)

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

    logging.info(f" [!] Model ready.")

    # ===================
    # TRAINING
    # ===================

    train_loader = create_dataloader(x_train, y_train, batch_size=256, shuffle=True, workers=4)
    test_loader = create_dataloader(x_test, y_test, batch_size=256, shuffle=False, workers=4)

    lit_trainer = LitTrainerWrapper(
        pytorch_model=model,
        name = "test_training",
        log_folder=f"./test_run_{int(time())}",
        epochs = 2,
        scheduler="onecycle",
        model_file="test.ckpt",
    )
    lit_trainer.train_lit_model(
        X_train_loader=train_loader,
        X_test_loader=test_loader,
    )

    # get f1 scores
    y_train_pred = lit_trainer.predict_lit_model(train_loader)
    f1_train = f1_score(y_train, y_train_pred)
    print(f"[!] Train F1: {f1_train*100:.2f}%")

    y_test_pred = lit_trainer.predict_lit_model(test_loader)
    f1_test = f1_score(y_test, y_test_pred)
    print(f"[!] Test F1: {f1_test*100:.2f}%")
