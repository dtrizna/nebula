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

from nebula.models import TransformerEncoderChunks
from nebula.evaluation.lit_cv import LitCrossValidation
from nebula.lit_utils import LitTrainerWrapper

from nebula.misc import set_random_seed, clear_cuda_cache

LIMIT = None
RANDOM_SEED = 33
INFOLDER = "out_tokenizer_1710259149" # r"evaluation\dynamic_sota\out_50" # if data is processed already
BYTE_FALLBACK = True

VOCAB = 50000
SEQ_LEN = 512
SPEAKEASY_TRAINSET_PATH = os.path.join(REPO_ROOT, "..", "speakeasy", "windows_emulation_trainset")
SPEAKEASY_TESTSET_PATH = os.path.join(REPO_ROOT, "..", "speakeasy", "windows_emulation_testset")

if __name__ == "__main__":
    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_tokenizer_{int(time.time())}"
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
    for run_name in ['bpe_wo_bytes', 'bpe', 'whitespace']:
        datafolders[run_name] = os.path.join(out_folder_root, f"nebula_{run_name}_vocab_{VOCAB}_seqlen_{SEQ_LEN}")
        
        # =========== 'nebula' & 'speakeasy' preprocessing
        byte_fallback = False if "wo_bytes" in run_name else True
        tokenizer_type = "bpe" if "bpe" in run_name else run_name
        
        _, y_train, y_paths_train, = preprocess_nebula_speakeasy(
            folder=SPEAKEASY_TRAINSET_PATH,
            limit=LIMIT,
            vocab_size=VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders[run_name],
            tokenizer_type=tokenizer_type,
            byte_fallback=byte_fallback
        )

        if "bpe" in run_name:
            tokenizer_model = os.path.join(datafolders[run_name], f"tokenizer_{VOCAB}.model")
        else:
            tokenizer_model = os.path.join(datafolders[run_name], f"tokenizer_{VOCAB}_vocab.json")
        
        _, y_test, y_paths_test = preprocess_nebula_speakeasy(
            folder=SPEAKEASY_TESTSET_PATH,
            limit=LIMIT,
            vocab_size=VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders[run_name],
            tokenizer_model=tokenizer_model,
            tokenizer_type=tokenizer_type,
            byte_fallback=byte_fallback
        )

        # ============= DEFINE MODEL =============
        with open(os.path.join(datafolders[run_name], f"tokenizer_{VOCAB}_vocab.json")) as f:
            nebula_vocab = json.load(f)
        models[run_name]['class'] = TransformerEncoderChunks
        models[run_name]['config'] = {
            "vocab_size": len(nebula_vocab),
            "maxlen": SEQ_LEN,
            "chunk_size": 64,
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": 1, # binary classification
            "classifier_head": [64],
            "layerNorm": False,
            "dropout": 0.5,
            # "mean_over_sequence": False,
            "norm_first": True
        }

        # ============= TRAINING LOOP ============= 
        cv_outfolder = os.path.join(out_folder_root, f"training_{run_name}_lim{LIMIT}_r{RANDOM_SEED}")
        # if os.path.exists(cv_outfolder):
        #     logging.warning(f" [!] CV output folder {cv_outfolder} already exists, skipping!")
        #     continue

        logging.warning(f" [!!!] Starting {run_name} training!")
        
        suffix = LIMIT if LIMIT else "full"
        x_train = np.load(os.path.join(datafolders[run_name], f"x_train_{suffix}.npy"))
        y_train = np.load(os.path.join(datafolders[run_name], f"y_train_{suffix}.npy"))
        x_test = np.load(os.path.join(datafolders[run_name], f"x_test_{suffix}.npy"))
        y_test = np.load(os.path.join(datafolders[run_name], f"y_test_{suffix}.npy"))
        
        model_class = models[run_name]['class']
        model_config = models[run_name]['config']
        model = model_class(**model_config)

        set_random_seed(RANDOM_SEED)
        lit_trainer = LitTrainerWrapper(
            pytorch_model=model,
            name=run_name,
            log_folder=cv_outfolder,
            epochs=30,
            scheduler="onecycle",
            device="gpu",
            nr_of_devices=1,
            # data config
            batch_size=512,
            dataloader_workers=4,
            # misc
            random_state=RANDOM_SEED,
            verbose=True
        )
        train_dataloader = lit_trainer.create_dataloader(x_train, y_train)
        test_dataloader = lit_trainer.create_dataloader(x_test, y_test, shuffle=False)
        lit_trainer.train_lit_model(
            train_loader=train_dataloader,
            val_loader=test_dataloader
        )

        del lit_trainer, x_train, y_train
        clear_cuda_cache()
