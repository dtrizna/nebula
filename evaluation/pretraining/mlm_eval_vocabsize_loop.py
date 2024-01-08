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

from nebula.lit_pretraining import MaskedLanguageModelTrainer, SelfSupervisedLearningEvalFramework
from nebula.lit_utils import LitTrainerWrapper
from nebula.models.attention import TransformerEncoderChunks

VOCABS = {'8k': 8192}#, '16k': 16384, '32k': 32768}

# if clean training
LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"long_pretrain_remask_{int(time())}")
TIMESTAMPS = None
# if previously trained
# LOG_ROOT_FOLDER = os.path.join(REPOSITORY_ROOT, "evaluation", "pretraining", f"long_pretrain_remask_1704440720")
# TIMESTAMPS = ["1704440720"]

VERBOSE = False
LIMIT = None

PRETRAINING_EPOCHS = 50
DOWNSTREAM_EPOCHS = 10
DATALOADER_WORKERS = 4
LOG_EVERY_N_STEPS = 10
SSL_EVAL_SPLITS = 1
# efficient training methods
GRAD_CLIP_VALUE = 1.0
ACCUMULATE_GRAD_BATCHES = None # default, batch_sizes are big enough

# NOTE: scheduler is incompatible with REMASK_EVERY_N_EPOCHS 
# and DUMP_MODEL_EVERY_EPOCH because of lighning nuances
# USE:
# SCHEDULER = "onecycle"
REMASK_EVERY_N_EPOCHS = False
# DUMP_MODEL_EVERY_EPOCH = False
# OR
SCHEDULER = None
REMASK_EVERY_N_EPOCHS = 2
DUMP_MODEL_EVERY_EPOCH = True


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
        "maxlen": 512,
        "chunk_size": 64, # input splitting to chunks
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1, # binary classification
        "hiddenNeurons": [64], # classifier ffnn dims
        "layerNorm": False,
        "dropout": None,
        "norm_first": True,
        "pretrain_layers": [1024]
    }
    # NOTE: for pretraining 0 is good, for finetuning try 0.1+
    model_config['dropout'] = 0.0
    lm_model = TransformerEncoderChunks(**model_config)
    model_config['dropout'] = 0.3
    model_config['pretrain_layers'] = None
    downstream_model = TransformerEncoderChunks(**model_config)

    print(f"[!] Models ready.")
    
    # ===================
    # TRAINING
    # ===================
    NAME = f"vocab_{vocab_size_str}"

    trainer_config = {
        "log_every_n_steps": LOG_EVERY_N_STEPS,
        "device": "gpu",
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        "gradient_clip_val": GRAD_CLIP_VALUE,
        "scheduler": SCHEDULER,
        "dataloader_workers": DATALOADER_WORKERS,
        "random_state": random_state,
        "verbose": VERBOSE
    }
    lit_mlm = MaskedLanguageModelTrainer(
        # pretrainer config
        vocab=vocab,
        pretrain_epochs=PRETRAINING_EPOCHS,
        rebuild_dataloader_every_n_epochs=REMASK_EVERY_N_EPOCHS,
        dump_model_every_epoch=DUMP_MODEL_EVERY_EPOCH,
        # trainer config
        pytorch_model=lm_model,
        name = NAME + "_pretraining",
        log_folder=f"{NAME}_pretraining",
        batch_size=512,
        precision=16,
        **trainer_config
    )
    downstream_trainer = LitTrainerWrapper(
        # trainer config
        epochs=DOWNSTREAM_EPOCHS,
        pytorch_model=downstream_model,
        name = NAME + "_downstream",
        log_folder=f"{NAME}_downstream",
        skip_trainer_init=True,
        batch_size=256,
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
        training_types = ['pretrained', 'non_pretrained', 'full_data'],
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
