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
    preprocess_quovadis_speakeasy,
    preprocess_neurlux,
)

from nebula import ModelTrainer
from nebula.models.neurlux import NeurLuxModel
from nebula.models.quovadis import QuoVadisModel
from nebula.models import TransformerEncoderChunks
from nebula.evaluation import CrossValidation
from nebula.misc import set_random_seed, clear_cuda_cache

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

LIMIT = None
RANDOM_SEED = 1763
TIME_BUDGET = 5 # minutes
INFOLDER = "out_speakeasy" # if data is processed already

NEBULA_VOCAB = 50000
NEURLUX_VOCAB = 10000
QUO_VADIS_TOP_API = 600
SEQ_LEN = 512
SPEAKEASY_TRAINSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_trainset")
SPEAKEASY_TESTSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_testset")

if __name__ == "__main__":
    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_speakeasy_{int(time.time())}"
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
    datafolders['nebula'] = os.path.join(out_folder_root, f"nebula_speakeasy_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['neurlux'] = os.path.join(out_folder_root, f"neurlux_speakeasy_vocab_{NEURLUX_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['quovadis'] = os.path.join(out_folder_root, f"quovadis_speakeasy_vocab_{QUO_VADIS_TOP_API}_seqlen_{SEQ_LEN}")
    
    # =========== 'nebula' & 'speakeasy' preprocessing
    _, y_train, y_paths_train, = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TRAINSET_PATH,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula'],
    )
    _, y_test, y_paths_test = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TESTSET_PATH,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula'],
        tokenizer_model=os.path.join(datafolders['nebula'], f"tokenizer_{NEBULA_VOCAB}.model"),
    )

    # =========== 'neurlux' & 'speakeasy' preprocessing
    _, neurlux_vocab_file = preprocess_neurlux(
        y_paths_train, 
        y=y_train,
        outfolder=datafolders['neurlux'],
        limit=LIMIT,
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN

    )
    _ = preprocess_neurlux(
        y_paths_test,
        y=y_test,
        vocab_file=neurlux_vocab_file,
        outfolder=datafolders['neurlux'],
        limit=LIMIT,
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN
    )

    # ============ 'quo vadis' & 'speakeasy' preprocessing
    _, quovadis_vocab_file = preprocess_quovadis_speakeasy(
        y_paths_train,
        y=y_train,
        seq_len=SEQ_LEN,
        top_api=QUO_VADIS_TOP_API,
        outfolder=datafolders['quovadis'],
        limit=LIMIT,
    )
    _ = preprocess_quovadis_speakeasy(
        y_paths_test,
        y=y_test,
        vocab_file=quovadis_vocab_file,
        seq_len=SEQ_LEN,
        top_api=QUO_VADIS_TOP_API,
        outfolder=datafolders['quovadis'],
        limit=LIMIT,
    )

    # ============= DEFINE MODELS =============
    models = defaultdict(dict)

    with open(neurlux_vocab_file) as f:
        neurlux_vocab = json.load(f)
    models['neurlux']['class'] = NeurLuxModel
    models['neurlux']['config'] = {
        "embedding_dim": 256,
        "vocab_size": len(neurlux_vocab),
        "seq_len": SEQ_LEN,
    }

    models['quovadis']['class'] = QuoVadisModel
    models['quovadis']['config'] = {
        "vocab": quovadis_vocab_file,
        "seq_len": SEQ_LEN
    }
    
    with open(os.path.join(datafolders['nebula'], f"tokenizer_{NEBULA_VOCAB}_vocab.json")) as f:
        nebula_vocab = json.load(f)
    models['nebula']['class'] = TransformerEncoderChunks
    models['nebula']['config'] = {
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
        "batchSize": 96,
        "verbosity_n_batches": 100,
        "clip_grad_norm": 1.0,
        "n_batches_grad_update": 1,
        "time_budget": int(TIME_BUDGET*60) if TIME_BUDGET else None,
    }

    # ============= TRAINING LOOP ============= 
    for run_name in models.keys():
        logging.warning(f" [!!!] Starting CV over {run_name}!")
        
        suffix = LIMIT if LIMIT else "full"
        x_train = np.load(os.path.join(datafolders[run_name], f"x_train_{suffix}.npy"))
        y_train = np.load(os.path.join(datafolders[run_name], f"y_train_{suffix}.npy"))
        
        model_class = models[run_name]['class']
        model_config = models[run_name]['config']
        
        if LIMIT:
            x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED)
            x_train = x_train[:LIMIT]
            y_train = y_train[:LIMIT]
            
        set_random_seed(RANDOM_SEED)
        cv_outfolder = os.path.join(out_folder_root, f"cv_{run_name}_lim{LIMIT}_r{RANDOM_SEED}_t{TIME_BUDGET}")
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
        cv.dump_metrics(prefix=f"{run_name}")
        cv.log_avg_metrics()

        del cv, x_train, y_train
        clear_cuda_cache()
