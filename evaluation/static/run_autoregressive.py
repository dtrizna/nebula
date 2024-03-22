import os
import sys
import json
import pickle
import logging
import numpy as np
from time import time
from torch import compile
from sklearn.utils import shuffle
from lightning import seed_everything

# correct path to repository root
if os.path.basename(os.getcwd()) == "static":
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

from tokenizer import StaticBytesTokenizer

PRETRAINING_EPOCHS = 10
DOWNSTREAM_EPOCHS = 15

AR_PRETRAIN_BATCH_SIZE = 64
AR_CONTEXT_LEN = 256

DOWNSTREAM_BATCH_SIZE = 16
DOWNSTREAM_ACCUMULATE_GRAD_BATCHES = 4
VOCAB_SIZE = 32768

# NOTE: try to reduce to e.g. 0.95 and compare?
CHAR_COVERAGE = 0.9995

# NOTE: this is the maximum length of the TOKENIZED sequence
MAX_SEQ_LEN = 1024
# NOTE: this is the maximum length of the RAW BYTE sequence (further tokenized)
# greedly expanded to MAX_SEQ_LEN * 16 (~16 bytes per token) since reading full PEs
# result in process dying (is Killed by OS at ~75k PE files, see err.txt)
MAX_RAW_BYTE_LEN = MAX_SEQ_LEN * 16

DEVICE = "gpu"
NR_OF_DEVICES = 2
DATALOADER_WORKERS = 4
LOG_EVERY_N_STEPS = 10
SSL_EVAL_SPLITS = 1
GRAD_CLIP_VALUE = 1.0
VERBOSE = True
SCHEDULER = "onecycle"

LIMIT = None
RANDOM_SEED = 33
LOG_ROOT_FOLDER = os.path.join(
    REPOSITORY_ROOT,
    "evaluation",
    "static",
    f"out_autoregressive_lim{LIMIT}_rs{RANDOM_SEED}_v2"
)

def load_testset_paths(limit=None, random_seed=33):
    test_folder = os.path.join("/data", "datasets", "pe", "pe_testset")

    goodware_folder = os.path.join(test_folder, "goodware")
    goodware_files = [os.path.join(goodware_folder, f) for f in os.listdir(goodware_folder)]
    if limit is not None:
        np.random.seed(random_seed)
        goodware_files = np.random.choice(goodware_files, limit//2, replace=False).tolist()

    malware_folder = os.path.join(test_folder, "malware")
    malware_files = [os.path.join(malware_folder, f) for f in os.listdir(malware_folder)]
    if limit is not None:
        np.random.seed(random_seed)
        malware_files = np.random.choice(malware_files, limit//2, replace=False).tolist()

    x_test_filenames = goodware_files + malware_files
    y_test = np.array([0] * len(goodware_files) + [1] * len(malware_files))
    x_test_filenames, y_test = shuffle(x_test_filenames, y_test, random_state=random_seed)

    return x_test_filenames, y_test


def load_trainset_paths(limit=None, random_seed=33):
    train_folder = os.path.join("/data", "datasets", "pe", "pe_trainset", "PeX86Exe")

    subfolders = os.listdir(train_folder)

    goodware_files = []
    malware_files = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(train_folder, subfolder)
        if os.path.isdir(subfolder_path) and subfolder == "clean":
            goodware_files += [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)]
        else:
            malware_files += [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)]

    if limit is not None:
        np.random.seed(random_seed)
        goodware_files = np.random.choice(goodware_files, limit//2, replace=False).tolist()
        np.random.seed(random_seed)
        malware_files = np.random.choice(malware_files, limit//2, replace=False).tolist()

    x_train_filenames = goodware_files + malware_files
    y_train = np.array([0] * len(goodware_files) + [1] * len(malware_files))
    x_train_filenames, y_train = shuffle(x_train_filenames, y_train, random_state=random_seed)

    return x_train_filenames, y_train



def main(random_state=33):
    seed_everything(random_state)

    # ==============
    # LOAD DATA
    # ==============
    logging.warning(f"[*] Initializing tokenizer...")

    # initializations
    tokenizer_folder = os.path.join(LOG_ROOT_FOLDER, "tokenizer")
    
    # check if there is a trained tokenizer model
    tokenizer_model_file = os.path.join(tokenizer_folder, "sp_bpe.model")
    tokenizer_vocab_file = os.path.join(tokenizer_folder, "sp_bpe_vocab.json")
    if os.path.exists(tokenizer_model_file) and os.path.exists(tokenizer_vocab_file):
        tokenizer = StaticBytesTokenizer(
            vocab_size=VOCAB_SIZE,
            model_path=tokenizer_model_file,
            outfolder=tokenizer_folder
        )
    else:
        tokenizer = StaticBytesTokenizer(
            vocab_size=VOCAB_SIZE,
            outfolder=tokenizer_folder
        )

    logging.warning(f"[*] Loading data...")
    x_train_filenames, y_train = load_trainset_paths(limit=LIMIT, random_seed=random_state)
    x_test_filenames, y_test = load_testset_paths(limit=LIMIT, random_seed=random_state)

    # read all bytes from training files
    logging.warning(f"[*] Reading train data bytes...")
    train_bytez_pkl = os.path.join(LOG_ROOT_FOLDER, f"x_train_bytez_lim{LIMIT}_rs{random_state}.pkl")
    if os.path.exists(train_bytez_pkl):
        with open(train_bytez_pkl, 'rb') as f:
            x_train_bytez = pickle.load(f)
    else:
        # NOTE: takes ~1 min to load 1300 files
        x_train_bytez = tokenizer.read_corpus_bytez(x_train_filenames, seqlen=MAX_RAW_BYTE_LEN)
        with open(train_bytez_pkl, 'wb') as f:
            pickle.dump(x_train_bytez, f)
    
    # train tokenizer
    if not (os.path.exists(tokenizer_model_file) and os.path.exists(tokenizer_vocab_file)):
        logging.warning(f"[*] Training tokenizer...")
        tokenizer.train(corpus_bytez=x_train_bytez)

    # encode byte to numerical vectors of token ids
    logging.warning(f"[!] Encoding train data...")
    x_train_encoded_file = os.path.join(LOG_ROOT_FOLDER, f'x_train_encoded_lim{LIMIT}_rs{RANDOM_SEED}.npy')
    if os.path.exists(x_train_encoded_file):
        x_train = np.load(x_train_encoded_file)
    else:
        x_train = tokenizer.encode(x_train_bytez, pad_len=MAX_SEQ_LEN)
        np.save(x_train_encoded_file, x_train)
        np.save(os.path.join(LOG_ROOT_FOLDER, f"y_train_lim{LIMIT}_rs{RANDOM_SEED}.npy"), y_train)

    del x_train_bytez

    # read bytes from test files
    logging.warning(f"[*] Reading test data bytes...")
    test_bytez_pkl = os.path.join(LOG_ROOT_FOLDER, f"x_test_bytez_lim{LIMIT}_rs{random_state}.pkl")
    if os.path.exists(test_bytez_pkl):
        with open(test_bytez_pkl, 'rb') as f:
            x_test_bytez = pickle.load(f)
    else:
        x_test_bytez = tokenizer.read_corpus_bytez(x_test_filenames, seqlen=MAX_RAW_BYTE_LEN)
        with open(test_bytez_pkl, 'wb') as f:
            pickle.dump(x_test_bytez, f)

    logging.warning(f"[!] Encoding test data...")
    x_test_encoded_file = os.path.join(LOG_ROOT_FOLDER, f'x_test_encoded_lim{LIMIT}_rs{RANDOM_SEED}.npy')
    if os.path.exists(x_test_encoded_file):
        x_test = np.load(x_test_encoded_file)
    else:
        x_test = tokenizer.encode(x_test_bytez, pad_len=MAX_SEQ_LEN)
        np.save(x_test_encoded_file, x_test)
        np.save(os.path.join(LOG_ROOT_FOLDER, f"y_test_lim{LIMIT}_rs{RANDOM_SEED}.npy"), y_test)

    del x_test_bytez

    logging.warning(f"[!] Data ready.")
    
    # ============= SELF-SUPERVISED MODEL'S OBJECT ===============
    model_config = {
        "vocab_size": VOCAB_SIZE,
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
    compile(ar_model)

    # ============= TRAINER OBJECT CONFIGS ===============
    pre_trainer_config = {
        "vocab": tokenizer.vocab,
        "pretrain_epochs": PRETRAINING_EPOCHS,
        "dump_model_every_epoch": False,
        "rebuild_dataloader_every_n_epochs": False,
    }
    trainer_config = {
        "log_every_n_steps": LOG_EVERY_N_STEPS,
        "device": DEVICE,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": GRAD_CLIP_VALUE,
        "scheduler": SCHEDULER,
        "dataloader_workers": DATALOADER_WORKERS,
        "random_state": random_state,
        "verbose": VERBOSE,
        "nr_of_devices": NR_OF_DEVICES,
        "precision": 16
    }

    # ========== AUTO-REGRESSIVE PRE-TRAINER ===========
    ar_lit_pretrainer = AutoRegressiveModelTrainer(
        # autoregressive only params
        context_len=AR_CONTEXT_LEN,
        # lit trainer config
        pytorch_model=ar_model,
        name="pretraining",
        log_folder=f"ar_pretraining",
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
    compile(downstream_model)

    trainer_config["accumulate_grad_batches"] = DOWNSTREAM_ACCUMULATE_GRAD_BATCHES
    trainer_config["precision"] = 32
    downstream_trainer = LitTrainerWrapper(
        # trainer config
        epochs=DOWNSTREAM_EPOCHS,
        pytorch_model=downstream_model,
        name=f"downstream",
        log_folder=f"ar_downstream",
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
        training_types=['pretrained', 'non_pretrained', 'full_data'],
        supervised_data_ratio=0.2,
        unlabeled_data_ratio=0.8,
        log_folder=LOG_ROOT_FOLDER,
        dump_data_splits=True,
        random_state=random_state,
        n_splits=SSL_EVAL_SPLITS
    )
    eval_run.run_splits(x_train, y_train, x_test, y_test)


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

    main(RANDOM_SEED)
