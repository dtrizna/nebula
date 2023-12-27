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
from nebula.models.attention import TransformerEncoderChunksLM
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
    limit = 1000
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
        "dropout": 0.1,
        "mean_over_sequence": False,
        "norm_first": True
    }
    model = TransformerEncoderChunksLM(**model_config)

    print(f"[!] Model ready.")

    # ===================
    # TRAINING
    # ===================
    VERBOSE = False
    PRETRAINING_EPOCHS = 4
    DOWNSTREAM_EPOCHS = 3
    REMASK_EVERY_N_EPOCHS = 2
    BATCH_SIZE = 128
    DATALOADER_WORKERS = 4
    LOG_EVERY_N_STEPS = 1
    SSL_EVAL_SPLITS = 3
    NAME = "mlm_testrun"
    lit_mlm = MaskedLanguageModelTrainer(
        # pretrain config
        vocab=vocab,
        pretrain_epochs=PRETRAINING_EPOCHS,
        remask_epochs=REMASK_EVERY_N_EPOCHS,
        dump_model_every_epoch=False,
        # trainer config
        pytorch_model=model,
        name = NAME + "_pretraining",
        log_folder=f"z_{NAME}_pretraining",
        log_every_n_steps=LOG_EVERY_N_STEPS,
        device="gpu",
        # data config
        batch_size=BATCH_SIZE,
        dataloader_workers=DATALOADER_WORKERS,
        # misc
        random_state=random_state,
        verbose=VERBOSE
    )
    downstream_trainer = LitTrainerWrapper(
        # trainer config
        epochs=DOWNSTREAM_EPOCHS,
        pytorch_model=model,
        name = NAME + "_downstream",
        log_folder=f"z_{NAME}_downstream",
        log_every_n_steps=LOG_EVERY_N_STEPS,
        device="gpu",
        skip_trainer_init=True,
        # data config
        batch_size=BATCH_SIZE,
        dataloader_workers=DATALOADER_WORKERS,
        # misc
        random_state=random_state,
        verbose=VERBOSE
    )
    eval_run = SelfSupervisedLearningEvalFramework(
        pretrainer=lit_mlm,
        downstream_trainer=downstream_trainer,
        unlabeled_data_ratio=0.8,
        log_folder=f"./z_mlm_eval_{int(time())}",
        dump_data_splits=True,
        random_state=random_state,
        n_splits=SSL_EVAL_SPLITS
    )

    eval_run.run_splits(x_train, y_train, x_test, y_test)
