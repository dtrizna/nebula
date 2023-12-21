import os
import sys
sys.path.extend(['..', '.'])
from nebula.misc import get_path
from nebula.models.attention import TransformerEncoderChunks
from nebula.evaluation.lit_cv import LitCrossValidation

import time
import json
import numpy as np
from watermark import watermark
from sklearn.utils import shuffle


if __name__ == "__main__":
    print(watermark(packages="torch,lightning,numpy", python=True))

    SCRIPT_PATH = get_path(type="script")
    REPO_ROOT = os.path.join(SCRIPT_PATH, "..")
    out_folder = os.path.join(REPO_ROOT, "scripts-lit", f"lit_cv_run_out_{int(time.time())}")
    os.makedirs(out_folder, exist_ok=True)

    # DATA

    x_tr_file = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", f"speakeasy_vocab_size_50000_maxlen_512_x.npy")
    x_train = np.load(x_tr_file)
    y_tr_file = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", "speakeasy_y.npy")
    y_train = np.load(y_tr_file)

    LIMIT = 10000
    if LIMIT:
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        x_train = x_train[:LIMIT]
        y_train = y_train[:LIMIT]

    vocabFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE_50k", "speakeasy_vocab_size_50000_tokenizer_vocab.json")
    with open(vocabFile, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab) # adjust it to exact number of tokens in the vocabulary


    # CV AND MODEL

    model_class = TransformerEncoderChunks
    model_config = {
        "vocab_size": vocab_size,  # size of vocabulary
        "maxlen": 512,  # maximum length of the input sequence
        "chunk_size": 64,
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 2, # binary classification
        "hiddenNeurons": [64],
        "layerNorm": False,
        "dropout": 0.3,
        "mean_over_sequence": False
    }

    litcv = LitCrossValidation(
        model_class,
        model_config,
        folds=3,
        dump_data_splits=True,
        batch_size=64,
        dataloader_workers=8,
        out_folder=out_folder
    )
    litcv.run(
        x_train,
        y_train,
        max_epochs=3
        #max_time={"minutes": 5}
    )
