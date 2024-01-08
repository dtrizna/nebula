import os
REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

import sys
sys.path.extend([REPO_ROOT, "."])

import time
import json
import logging
import numpy as np
from collections import defaultdict
from sklearn.utils import shuffle

from nebula.preprocessing.wrappers import (
    preprocess_nebula_speakeasy,
)

from nebula import ModelTrainer
from nebula.models import TransformerEncoderChunks
from nebula.evaluation import CrossValidation
from nebula.misc import set_random_seed, clear_cuda_cache

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

LIMIT = None
RANDOM_SEED = 1763
TIME_BUDGET = 5 # minutes
INFOLDER = None # if data is processed already

VOCAB = int(5e4)
TOKENIZER_TYPE = "bpe"
SPEAKEASY_TRAINSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_trainset")
SPEAKEASY_TESTSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_testset")

if __name__ == "__main__":
    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_seqlen_{int(time.time())}"
        os.makedirs(out_folder_root, exist_ok=True)

    # =========== set out logging to both file and stdout
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_folder_root, f"log_{int(time.time())}.txt")),
            logging.StreamHandler()
        ]
    )

    datafolders = {}
    models = defaultdict(dict)
    for SEQ_LEN in [int(x) for x in [64, 128, 256, 512, 1024, 2048]]:
        logging.warning(f"[!!!] Starting run with seqlen : {SEQ_LEN}")
        datafolders[SEQ_LEN] = os.path.join(out_folder_root, f"nebula_vocab_{VOCAB}_seqlen_{SEQ_LEN}")
        # =========== 'nebula' & 'speakeasy' preprocessing
        _, y_train, y_paths_train, = preprocess_nebula_speakeasy(
            folder=SPEAKEASY_TRAINSET_PATH,
            limit=LIMIT,
            vocab_size=VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders[SEQ_LEN],
            tokenizer_type=TOKENIZER_TYPE
        )
        if TOKENIZER_TYPE == "bpe":
            tokenizer_model = os.path.join(datafolders[SEQ_LEN], f"tokenizer_{VOCAB}.model")
        else:
            tokenizer_model = os.path.join(datafolders[SEQ_LEN], f"tokenizer_{VOCAB}_vocab.json")
        _, y_test, y_paths_test = preprocess_nebula_speakeasy(
            folder=SPEAKEASY_TESTSET_PATH,
            limit=LIMIT,
            vocab_size=VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders[SEQ_LEN],
            tokenizer_model=tokenizer_model,
            tokenizer_type=TOKENIZER_TYPE
        )

        # ============= DEFINE MODEL =============
        with open(os.path.join(datafolders[SEQ_LEN], f"tokenizer_{VOCAB}_vocab.json")) as f:
            nebula_vocab = json.load(f)
        models[SEQ_LEN]['class'] = TransformerEncoderChunks
        models[SEQ_LEN]['config'] = {
            "vocab_size": len(nebula_vocab),
            "maxlen": SEQ_LEN,
            "chunk_size": 64,
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": 1, # binary classification
            "hiddenNeurons": [64],
            "layerNorm": False,
            "dropout": 0.3,
            "mean_over_sequence": False,
            "norm_first": True
        }

        device = "cuda" if cuda.is_available() else "cpu"
        model_trainer_class = ModelTrainer
        model_trainer_config = {
            "device": device,
            "model": None, # will be set later within CrossValidation class
            "loss_function": BCEWithLogitsLoss(),
            "optimizer_class": AdamW,
            "optimizer_config": {"lr": 3e-4},
            "optim_scheduler": None,
            "optim_step_budget": None,
            "outputFolder": None, # will be set later within CrossValidation class
            "batchSize": 64,
            "verbosity_n_batches": 100,
            "clip_grad_norm": 1.0,
            "n_batches_grad_update": 1,
            "time_budget": int(TIME_BUDGET*60) if TIME_BUDGET else None,
        }

        # ============= TRAINING LOOP ============= 
        logging.warning(f" [!!!] Starting CV over {SEQ_LEN}!")
        
        suffix = LIMIT if LIMIT else "full"
        x_train = np.load(os.path.join(datafolders[SEQ_LEN], f"x_train_{suffix}.npy"))
        y_train = np.load(os.path.join(datafolders[SEQ_LEN], f"y_train_{suffix}.npy"))
        
        model_class = models[SEQ_LEN]['class']
        model_config = models[SEQ_LEN]['config']
        
        if LIMIT:
            x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED)
            x_train = x_train[:LIMIT]
            y_train = y_train[:LIMIT]
            
        set_random_seed(RANDOM_SEED)
        cv_outfolder = os.path.join(out_folder_root, f"cv_{SEQ_LEN}_lim{LIMIT}_r{RANDOM_SEED}_t{TIME_BUDGET}")
        cv = CrossValidation(
            model_trainer_class,
            model_trainer_config,
            model_class,
            model_config,
            output_folder_root=cv_outfolder
        )
        cv.run_folds(
            x_train, 
            y_train, 
            epochs=None, # time budget specified
            folds=3, 
            random_state=RANDOM_SEED
        )
        cv.dump_metrics(prefix=f"{SEQ_LEN}")
        cv.log_avg_metrics()

        del cv, x_train, y_train
        clear_cuda_cache()
