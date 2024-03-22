import os
import sys
import json
import logging
import numpy as np
from time import time
from sklearn.utils import shuffle
from lightning import seed_everything

# correct path to repository root
if os.path.basename(os.getcwd()) == "pretraining":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..", "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.lit_pretraining import (
    AutoRegressiveModelTrainer,
    SelfSupervisedLearningEvalFramework
)
from nebula.lit_utils import LitTrainerWrapper
from nebula.models.attention import TransformerEncoderModel

VOCABS = {'8k': 8192} # also: {'16k': 16384, '32k': 32768}

# if clean training
# LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"out_autoregressive_unlabeled_ratio_loop_{int(time())}")
TIMESTAMPS = None

# if previously trained
LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"out_autoregressive_unlabeled_ratio_loop_1709722547")
# TIMESTAMPS = ["1704971666"]

# TEST
# LIMIT = 5000
# PRETRAINING_EPOCHS = 1
# DOWNSTREAM_EPOCHS = 2

# PROD
LIMIT = None
PRETRAINING_EPOCHS = 3
DOWNSTREAM_EPOCHS = 5

AR_PRETRAIN_BATCH_SIZE = 64
AR_CONTEXT_LEN = 256

DOWNSTREAM_BATCH_SIZE = 16
DOWNSTREAM_ACCUMULATE_GRAD_BATCHES = 4
MAX_SEQ_LEN = 512

DEVICE = "gpu"
DATALOADER_WORKERS = 4
LOG_EVERY_N_STEPS = 10
SSL_EVAL_SPLITS = 1
GRAD_CLIP_VALUE = 1.0
VERBOSE = True

# NOTE: scheduler is incompatible with REMASK_EVERY_N_EPOCHS 
# and DUMP_MODEL_EVERY_EPOCH because of lightning nuances
# USE:
SCHEDULER = "onecycle"
DUMP_MODEL_EVERY_EPOCH = False
REBUILD_DL_EVERY_N_EPOCHS = False
# OR
# SCHEDULER = None
# DUMP_MODEL_EVERY_EPOCH = True
# REBUILD_DL_EVERY_N_EPOCHS = 1


def main(vocab_size_str='8k', random_state=33):
    seed_everything(random_state)

    # ==============
    # LOAD DATA
    # ==============
    logging.warning(f"[*] Loading data...")
    train_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", f"speakeasy_trainset_BPE_{vocab_size_str}_lim_None")
    x_train = np.load(os.path.join(train_folder, f"speakeasy_vocab_size_{VOCABS[vocab_size_str]}_maxlen_{MAX_SEQ_LEN}_x.npy"))
    y_train = np.load(os.path.join(train_folder, "speakeasy_y.npy"))

    test_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", f"speakeasy_testset_BPE_{vocab_size_str}_lim_None")
    x_test = np.load(os.path.join(test_folder, f"speakeasy_vocab_size_{VOCABS[vocab_size_str]}_maxlen_{MAX_SEQ_LEN}_x.npy"))
    y_test = np.load(os.path.join(test_folder, "speakeasy_y.npy"))

    # shuffle and limit
    limit = LIMIT
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = x_train[:limit]
    y_train = y_train[:limit]
    x_test, y_test = shuffle(x_test, y_test, random_state=0)
    x_test = x_test[:limit]
    y_test = y_test[:limit]

    vocabFile = [x for x in os.listdir(train_folder) if x.endswith("vocab.json")][0]
    vocabFile = os.path.join(train_folder, vocabFile)
    with open(vocabFile, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    logging.warning(f"[!] Data ready.")

    # ===================
    # RATIOS LOOP
    # ===================

    unlabeled_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for unlabeled_ratio in unlabeled_ratios:
        seed_everything(random_state)

        # check if already trained
        downstream_pretrained_folders = [x for x in os.listdir(LOG_ROOT_FOLDER) if f"unlabeled_ratio_{unlabeled_ratio}_downstream_pretrained" in x]
        if len(downstream_pretrained_folders) > 0:
            downstream_pretrained_folder_name = downstream_pretrained_folders[0]
            last_checkpoint_path = os.path.join(LOG_ROOT_FOLDER, downstream_pretrained_folder_name, r"pretrained_csv\version_0\checkpoints\last.ckpt")
            if os.path.exists(last_checkpoint_path):
                logging.warning(f"[!] Skipping ratio {unlabeled_ratio} - already trained.")
                continue

        logging.warning(f"[!] Starting training for unlabeled ratio {unlabeled_ratio}...")

        # ============= SELF-SUPERVISED MODEL'S OBJECT ===============
        model_config = {
            "vocab_size": vocab_size,
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": 1, # binary classification
            # pre-training settings
            "pretrain_layers": None,
            "classifier_head": None,
            # NOTE: for pretraining 0 is good, for finetuning try 0.1+
            "dropout": 0.0,
            "pooling": None,
            "causal_attention": True,
        }

        # autoregressive model
        model_config['pretrain_layers'] = [1024]
        model_config['maxlen'] = AR_CONTEXT_LEN # context length of sampled values
        ar_model = TransformerEncoderModel(**model_config)

        # ============= TRAINER OBJECT CONFIGS ===============
        pre_trainer_config = {
            "vocab": vocab,
            "pretrain_epochs": PRETRAINING_EPOCHS,
            "dump_model_every_epoch": DUMP_MODEL_EVERY_EPOCH,
            "rebuild_dataloader_every_n_epochs": REBUILD_DL_EVERY_N_EPOCHS,
        }
        trainer_config = {
            "log_every_n_steps": LOG_EVERY_N_STEPS,
            "device": DEVICE,
            "accumulate_grad_batches": None,
            "gradient_clip_val": GRAD_CLIP_VALUE,
            "scheduler": SCHEDULER,
            "dataloader_workers": DATALOADER_WORKERS,
            "random_state": random_state,
            "verbose": VERBOSE,
            "precision": 16
        }

        # ========== AUTO-REGRESSIVE PRE-TRAINER ===========
        ar_lit_pretrainer = AutoRegressiveModelTrainer(
            # autoregressive only params
            context_len=AR_CONTEXT_LEN,
            # lit trainer config
            pytorch_model=ar_model,
            name="pretraining",
            log_folder=f"unlabeled_ratio_{unlabeled_ratio}_pretraining",
            batch_size=AR_PRETRAIN_BATCH_SIZE,
            **trainer_config,
            **pre_trainer_config
        )

        # ===================
        # config update for downstream trainers
        # ===================

        model_config['maxlen'] = MAX_SEQ_LEN # lengths of the masked sequences
        model_config['causal_attention'] = False
        model_config['pooling'] = "flatten"
        model_config['dropout'] = 0.3
        model_config['pretrain_layers'] = None
        model_config['classifier_head'] = [64]
        downstream_model = TransformerEncoderModel(**model_config)

        trainer_config["accumulate_grad_batches"] = DOWNSTREAM_ACCUMULATE_GRAD_BATCHES
        trainer_config["precision"] = 32
        downstream_trainer = LitTrainerWrapper(
            # trainer config
            epochs=DOWNSTREAM_EPOCHS,
            pytorch_model=downstream_model,
            name=f"downstream_{unlabeled_ratio}",
            log_folder=f"unlabeled_ratio_{unlabeled_ratio}_downstream",
            skip_trainer_init=True,
            batch_size=DOWNSTREAM_BATCH_SIZE,
            **trainer_config
        )

        # ==============
        # EVAL RUN
        # ==============
        eval_run = SelfSupervisedLearningEvalFramework(
            pretrainer=ar_lit_pretrainer,
            downstream_trainer=downstream_trainer,
            training_types=['pretrained'],
            supervised_data_ratio=0.2,
            unlabeled_data_ratio=unlabeled_ratio,
            log_folder=LOG_ROOT_FOLDER,
            dump_data_splits=True,
            random_state=random_state,
            n_splits=SSL_EVAL_SPLITS
        )
        eval_run.run_splits(x_train, y_train, x_test, y_test, previous_run_idxs=TIMESTAMPS)


if __name__ == "__main__":
    if not os.path.exists(LOG_ROOT_FOLDER):
        os.makedirs(LOG_ROOT_FOLDER)
    
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_ROOT_FOLDER, f"log_{int(time())}.txt")),
            logging.StreamHandler()
        ]
    )

    config = {k:v for k,v in globals().items() if k.isupper()}
    with open(os.path.join(LOG_ROOT_FOLDER, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    main()
