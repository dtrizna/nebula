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

from nebula.lit_pretraining import AutoRegressiveModelTrainer, SelfSupervisedLearningEvalFramework
from nebula.lit_utils import LitTrainerWrapper
from nebula.models.attention import TransformerEncoderChunks, TransformerEncoderModel

VOCABS = {'8k': 8192}#, '16k': 16384, '32k': 32768}

# if clean training
LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"out_autoregressive_no_ffnn_{int(time())}")
TIMESTAMPS = None

# if previously trained
# LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"out_autoregressive_1704652486")
# TIMESTAMPS = ["1704652487"]

VERBOSE = True
LIMIT = 10000

PRETRAINING_EPOCHS = 2
PRETRAIN_BATCH_SIZE = 64
CONTEXT_LEN = 256

DOWNSTREAM_EPOCHS = 3
DOWNSTREAM_BATCH_SIZE = 16
MAX_SEQ_LEN = 512

DEVICE = "gpu"
DATALOADER_WORKERS = 4
LOG_EVERY_N_STEPS = 10
SSL_EVAL_SPLITS = 1
# efficient training methods
GRAD_CLIP_VALUE = 1.0
ACCUMULATE_GRAD_BATCHES = 4

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
    # MODELS
    # ===================
    print(f"[*] Loading model...")
    model_config = {
        "vocab_size": vocab_size,
        # "chunk_size": 64, # input splitting to chunks
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1, # binary classification
        # pre-training settings
        "maxlen": CONTEXT_LEN, # pretraining uses sampled contexts
        # NOTE: for pretraining 0 is good, for finetuning try 0.1+
        "dropout": 0.0,
        "pooling": None,
        "causal_attention": True,
        "pretrain_layers": [1024],
        "classifier_head": None, # classifier ffnn dims
    }
    lm_model = TransformerEncoderModel(**model_config)

    # rewriting settings for downstream model
    model_config['maxlen'] = MAX_SEQ_LEN # uses full sequences
    model_config['dropout'] = 0.3
    model_config['pretrain_layers'] = None
    model_config['causal_attention'] = False
    model_config['classifier_head'] = [64]
    # NOTE: downstream model requires pooling to aggregate Transformer output
    model_config['pooling'] = "flatten"
    downstream_model = TransformerEncoderModel(**model_config)
    
    print(f"[!] Models ready.")
    
    # ===================
    # TRAINING
    # ===================
    NAME = f"vocab_{vocab_size_str}"

    trainer_config = {
        "log_every_n_steps": LOG_EVERY_N_STEPS,
        "device": DEVICE,
        "accumulate_grad_batches": None,
        "gradient_clip_val": GRAD_CLIP_VALUE,
        "scheduler": SCHEDULER,
        "dataloader_workers": DATALOADER_WORKERS,
        "random_state": random_state,
        "verbose": VERBOSE
    }
    lit_mlm = AutoRegressiveModelTrainer(
        # autoregressive only
        context_len=CONTEXT_LEN,
        # pretrainer config
        vocab=vocab,
        pretrain_epochs=PRETRAINING_EPOCHS,
        dump_model_every_epoch=DUMP_MODEL_EVERY_EPOCH,
        rebuild_dataloader_every_n_epochs=REBUILD_DL_EVERY_N_EPOCHS,
        # lit trainer config
        pytorch_model=lm_model,
        name = NAME + "_pretraining",
        log_folder=f"{NAME}_pretraining",
        batch_size=PRETRAIN_BATCH_SIZE,
        precision=16,
        **trainer_config
    )

    trainer_config["accumulate_grad_batches"] = ACCUMULATE_GRAD_BATCHES
    downstream_trainer = LitTrainerWrapper(
        # trainer config
        epochs=DOWNSTREAM_EPOCHS,
        pytorch_model=downstream_model,
        name = NAME + "_downstream",
        log_folder=f"{NAME}_downstream",
        skip_trainer_init=True,
        batch_size=DOWNSTREAM_BATCH_SIZE,
        **trainer_config
    )
    
    eval_run = SelfSupervisedLearningEvalFramework(
        pretrainer=lit_mlm,
        downstream_trainer=downstream_trainer,
        unlabeled_data_ratio=0.8,
        log_folder=LOG_ROOT_FOLDER,
        dump_data_splits=True,
        random_state=random_state,
        n_splits=SSL_EVAL_SPLITS,
        training_types = ['full_data'],
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
