import os
import sys
import json
import numpy as np
from time import time
from sklearn.utils import shuffle
from lightning.lite.utilities.seed import seed_everything

# correct path to repository root
if os.path.basename(os.getcwd()) == "pretraining":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..", "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.lit_pretraining import (
    AutoRegressiveModelTrainer,
    MaskedLanguageModelTrainer,
    SelfSupervisedLearningEvalFramework
)
from nebula.lit_utils import LitTrainerWrapper
from nebula.models.attention import TransformerEncoderModel

VOCABS = {'8k': 8192} # also: {'16k': 16384, '32k': 32768}

# if clean training
# LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"out_mlm_vs_ar_{int(time())}")
# TIMESTAMPS = None

# if previously trained
LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"out_mlm_vs_ar")
TIMESTAMPS = ["1704971666"]

# TEST
# LIMIT = 5000
# PRETRAINING_EPOCHS = 1
# DOWNSTREAM_EPOCHS = 2

# PROD
LIMIT = None
PRETRAINING_EPOCHS = 3
DOWNSTREAM_EPOCHS = 5

MLM_PRETRAIN_BATCH_SIZE = 64
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
# and DUMP_MODEL_EVERY_EPOCH because of lighning nuances
# USE:
SCHEDULER = "onecycle"
DUMP_MODEL_EVERY_EPOCH = False
REBUILD_DL_EVERY_N_EPOCHS = False
# OR
# SCHEDULER = None
# DUMP_MODEL_EVERY_EPOCH = True
# REBUILD_DL_EVERY_N_EPOCHS = 1


def main(vocab_size_str, random_state=33):
    seed_everything(random_state)

    # ==============
    # LOAD DATA
    # ==============
    print(f"[*] Loading data...")
    train_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", f"speakeasy_trainset_BPE_{vocab_size_str}_lim_None")
    x_train = np.load(os.path.join(train_folder, f"speakeasy_vocab_size_{VOCABS[vocab_size_str]}_maxlen_512_x.npy"))
    y_train = np.load(os.path.join(train_folder, "speakeasy_y.npy"))

    test_folder = os.path.join(REPOSITORY_ROOT, "data", "data_filtered", f"speakeasy_testset_BPE_{vocab_size_str}_lim_None")
    x_test = np.load(os.path.join(test_folder, f"speakeasy_vocab_size_{VOCABS[vocab_size_str]}_maxlen_512_x.npy"))
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

    print(f"[!] Data ready.")

    # ===================
    # PRE-TRAINER SETUP
    # ===================
    
    # ============= MODEL OBJECTS ===============

    model_config = {
        "vocab_size": vocab_size,
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1, # binary classification
        # pre-training settings
        "pretrain_layers": [1024],
        "classifier_head": None, # classifier ffnn dims
        # NOTE: for pretraining 0 is good, for finetuning try 0.1+
        "dropout": 0.0,
        "pooling": None,
        "causal_attention": True,
    }

    # autoregressive model
    model_config['maxlen'] = AR_CONTEXT_LEN # context length of sampled values
    ar_model = TransformerEncoderModel(**model_config)

    # mlm_model
    model_config['maxlen'] = MAX_SEQ_LEN # lengths of the masked sequences
    model_config['causal_attention'] = False
    model_config['pooling'] = "flatten"
    # TODO: model ouptuts shape: 
    # torch.Size([32768, 8193]) == torch.Size([64, 512, 8193])
    # but should: torch.Size([64, 8193])
    mlm_model = TransformerEncoderModel(**model_config)
    
    # ============= TRAINER OBJECT CONFIGS ===============
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
    pre_trainer_config = {
        "vocab": vocab,
        "pretrain_epochs": PRETRAINING_EPOCHS,
        "dump_model_every_epoch": DUMP_MODEL_EVERY_EPOCH,
        "rebuild_dataloader_every_n_epochs": REBUILD_DL_EVERY_N_EPOCHS,
    }

    # ========== AUTO-REGRESSIVE PRE-TRAINER ===========
    ar_lit_trainer = AutoRegressiveModelTrainer(
        # autoregressive only params
        context_len=AR_CONTEXT_LEN,
        # lit trainer config
        pytorch_model=ar_model,
        name="ar",
        log_folder=f"ar_pretraining",
        batch_size=AR_PRETRAIN_BATCH_SIZE,
        **trainer_config,
        **pre_trainer_config
    )
    # ========== MLM PRE-TRAINER ===========
    mlm_lit_trainer = MaskedLanguageModelTrainer(
        # masked language model only params
        mask_probability = 0.15,
        # mask the token with distribution (80% mask, 10% random, 10% same), same as Devlin et al (https://arxiv.org/abs/1810.04805)
        mask_distribution = [0.8, 0.1],
        # how to construct the target for the masked tokens: 1 if onehot, sum of maskings if count
        masked_target_type = "onehot",
        # lit trainer config
        pytorch_model=mlm_model,
        name="mlm",
        log_folder=f"mlm_pretraining",
        batch_size=MLM_PRETRAIN_BATCH_SIZE,
        **trainer_config,
        **pre_trainer_config
    )

    # ===================
    # config update for downstream trainers
    # ===================
    trainer_config["accumulate_grad_batches"] = DOWNSTREAM_ACCUMULATE_GRAD_BATCHES
    trainer_config["precision"] = 32

    model_config['dropout'] = 0.3
    model_config['pretrain_layers'] = None
    model_config['classifier_head'] = [64]
    
    # ===================
    # EVAL LOOP
    # ===================

    runs = [
        mlm_lit_trainer,
        ar_lit_trainer,
    ]
    for pretrainer in runs:
        downstream_model = TransformerEncoderModel(**model_config)
        downstream_trainer = LitTrainerWrapper(
            # trainer config
            epochs=DOWNSTREAM_EPOCHS,
            pytorch_model=downstream_model,
            name=f"{pretrainer.name}_downstream",
            log_folder=f"{pretrainer.name}_downstream",
            skip_trainer_init=True,
            batch_size=DOWNSTREAM_BATCH_SIZE,
            **trainer_config
        )
        eval_run = SelfSupervisedLearningEvalFramework(
            pretrainer=pretrainer,
            downstream_trainer=downstream_trainer,
            unlabeled_data_ratio=0.8,
            log_folder=LOG_ROOT_FOLDER,
            dump_data_splits=True,
            random_state=random_state,
            n_splits=SSL_EVAL_SPLITS
        )
        eval_run.run_splits(x_train, y_train, x_test, y_test, previous_run_idxs=TIMESTAMPS)


if __name__ == "__main__":
    if not os.path.exists(LOG_ROOT_FOLDER):
        os.makedirs(LOG_ROOT_FOLDER)

    config = {k:v for k,v in globals().items() if k.isupper()}
    with open(os.path.join(LOG_ROOT_FOLDER, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    for vocab in VOCABS.keys():
        main(vocab)
