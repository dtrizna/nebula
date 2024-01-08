import os
import sys
import json
import numpy as np
from time import time
from sklearn.utils import shuffle

# correct path to repository root
if os.path.basename(os.getcwd()) == "pretraining":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..", "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.lit_pretraining import MaskedLanguageModelTrainer, SelfSupervisedLearningEvalFramework
from nebula.lit_utils import LitTrainerWrapper
from nebula.models.attention import TransformerEncoderChunks
from lightning.lite.utilities.seed import seed_everything

if __name__ == "__main__":
    random_state = 0
    seed_everything(random_state)

    # ==============
    # LOAD DATA
    # ==============
    print(f"[*] Loading data...")
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

    # shuffle and limit
    limit = 5000
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_train = x_train[:limit]
    y_train = y_train[:limit]
    x_test, y_test = shuffle(x_test, y_test, random_state=0)
    x_test = x_test[:limit]
    y_test = y_test[:limit]

    vocabFile = os.path.join(train_folder, r"speakeasy_vocab_size_50000_tokenizer_vocab.json")
    with open(vocabFile, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"[!] Data ready.")

    # ===================
    # MODELING
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
        "pretrain_layers": [1024],
        "pooling": None
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
    VERBOSE = True
    PRETRAINING_EPOCHS = 2
    DOWNSTREAM_EPOCHS = 3
    DATALOADER_WORKERS = 4
    LOG_EVERY_N_STEPS = 1
    SSL_EVAL_SPLITS = 2
    # efficient training methods
    GRAD_CLIP_VALUE = 1.0
    # might leave here default, batch sizes are reasonably big
    ACCUMULATE_GRAD_BATCHES = None

    # NOTE: scheduler is incompatible with REMASK_EVERY_N_EPOCHS 
    # and DUMP_MODEL_EVERY_EPOCH because of lighning nuances
    # USE:
    # SCHEDULER = "onecycle"
    REMASK_EVERY_N_EPOCHS = False
    DUMP_MODEL_EVERY_EPOCH = False
    # OR
    SCHEDULER = None
    # REMASK_EVERY_N_EPOCHS = 2
    # DUMP_MODEL_EVERY_EPOCH = True
    
    NAME = "no_scheduler_grad_clip_1"

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
        log_folder=f"z_{NAME}_pretraining",
        batch_size=512,
        precision=16,
        **trainer_config
    )
    downstream_trainer = LitTrainerWrapper(
        # trainer config
        epochs=DOWNSTREAM_EPOCHS,
        pytorch_model=downstream_model,
        name = NAME + "_downstream",
        log_folder=f"z_{NAME}_downstream",
        skip_trainer_init=True,
        batch_size=128,
        **trainer_config
    )
    eval_run = SelfSupervisedLearningEvalFramework(
        pretrainer=lit_mlm,
        downstream_trainer=downstream_trainer,
        unlabeled_data_ratio=0.8,
        log_folder=f"./z_out_mlm_effective_training_techniques_lim_{limit}",
        dump_data_splits=True,
        random_state=random_state,
        n_splits=SSL_EVAL_SPLITS
    )

    eval_run.run_splits(x_train, y_train, x_test, y_test)
