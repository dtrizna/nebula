import os
REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

import sys
sys.path.extend([REPO_ROOT, "."])

import time
import json
import string
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas import read_csv

from nebula import ModelTrainer
from nebula.models import TransformerEncoderChunks
from nebula.evaluation import CrossValidation
from nebula.misc import set_random_seed, clear_cuda_cache
from nebula.preprocessing.json import JSONTokenizerBPE, JSONTokenizerNaive
from nebula.constants import JSON_CLEANUP_SYMBOLS
from nebula.preprocessing.normalization import (
    normalizeStringPath
)

from torch import cuda
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW

LIMIT = None
VOCAB = int(3e4)
TIME_BUDGET = 5 # minutes
INFOLDER = None # if data is processed already

RANDOM_SEED = 1763

SEQ_LEN = 512
TOKENIZER_TYPE = "whitespace"
SUFFIX = LIMIT if LIMIT else "full"

data_folder = os.path.join(REPO_ROOT, r"data\data_raw\Avast\Public_Avast_CTU_CAPEv2_Dataset_Small\public_small_reports")
SAMPLE_PATHS = [os.path.join(data_folder, x) for x in os.listdir(data_folder)[:LIMIT]]
label_file = os.path.join(REPO_ROOT, r"data\data_raw\Avast\Public_Avast_CTU_CAPEv2_Dataset_Small\public_labels.csv")
LABEL_DF = read_csv(label_file)

LABEL_FIELD = 'classification_family'
LABEL_MAP = dict(zip(
    sorted(LABEL_DF[LABEL_FIELD].unique()),
    list(range(LABEL_DF[LABEL_FIELD].nunique()))
))

FIELDS = {
    'resolved_apis': [
        'resolved_apis'
    ],
    'executed_commands': [
        'executed_commands'
    ],
    'mutexes': [
        'mutexes'
    ],
    'files': [
        'files'
    ],
    'keys': [
        'keys'
    ],
    'all': [
        'resolved_apis',
        'executed_commands',
        'mutexes',
        'files',
        'keys'
    ]
}

CAPA_NORMALIZERS = {
            "resolved_apis": lambda x: x.lower().translate(str.maketrans(' ', ' ', string.punctuation)),
            'executed_commands': lambda x: normalizeStringPath(x),
            "mutexes": lambda x: x.lower().translate(str.maketrans(' ', ' ', string.punctuation)),
            'files': lambda x: normalizeStringPath(x),
            'keys': lambda x: x.lower().translate(str.maketrans(' ', ' ', string.punctuation)),
        }

if __name__ == "__main__":
    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_avast_fields_{int(time.time())}"
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
    for run_type in FIELDS:
        logging.warning(f" [!!!] Working on {run_type}!")
        datafolders[run_type] = os.path.join(out_folder_root, f"{run_type}_vocab_{VOCAB}_seqlen_{SEQ_LEN}")
        # =========== 'nebula' & 'avast capa' preprocessing
        if os.path.exists(os.path.join(datafolders[run_type], f"x_test_{SUFFIX}.npy")):
            logging.warning(f" [*] Data already processed, skipping: {datafolders[run_type]}")
            continue

        os.makedirs(datafolders[run_type], exist_ok=True)
        X_raw = []
        y = []
        logging.warning(" [!] Reading and normalizing reports...")
        for example in tqdm(SAMPLE_PATHS):
            hhash = os.path.basename(example).replace(".json", "")
            family = LABEL_DF[LABEL_DF['sha256'] == hhash][LABEL_FIELD].iloc[0]
            y.append(LABEL_MAP[family])

            # filtering report
            with open(example, encoding='utf-8') as f:
                sample = json.load(f)
            sample = sample["behavior"]['summary']
            normalized_sample = {}
            for field in FIELDS[run_type]:
                normalizer_f = CAPA_NORMALIZERS[field]
                normalized_sample[field] = [normalizer_f(x) for x in sample[field]]
            X_raw.append(normalized_sample)


        # TEMP: model selection 80/20%, TBD later -- based on date as in Avast paper ?
        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # =========== preprocessing X_train
        logging.warning(" [!] Tokenizing and encoding train data!")
        class_counts_in_train = Counter(y_train)
        logging.warning(f" [!] Size: {len(x_train_raw)} | Class counts in test:\n{class_counts_in_train}")
        if TOKENIZER_TYPE == "bpe":
            tokenizer = JSONTokenizerBPE(
                vocab_size=VOCAB,
                seq_len=SEQ_LEN,
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=[],
            )
        else:
            tokenizer = JSONTokenizerNaive(
                vocab_size=VOCAB,
                seq_len=SEQ_LEN,
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=[],
                counter_dump=True
            )

        logging.warning(" [*] Initializing tokenizer training...")
        tokenizer_model_prefix = os.path.join(datafolders[run_type], f"tokenizer_{VOCAB}")
        tokenizer.train(
            X_raw,
            model_prefix=tokenizer_model_prefix
        )
        x_train = tokenizer.encode(x_train_raw)

        logging.warning("[!] Tokenizing and encoding test data!")
        class_counts_in_test = Counter(y_train)
        logging.warning(f" [!] Size: {len(x_test_raw)} | Class counts in test:\n{class_counts_in_test}")
        if TOKENIZER_TYPE == "bpe":
            tokenizer = JSONTokenizerBPE(
                vocab_size=VOCAB,
                seq_len=SEQ_LEN,
                model_path=tokenizer_model_prefix+".model",
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=[]
            )
        else:
            tokenizer = JSONTokenizerNaive(
                vocab_size=VOCAB,
                seq_len=SEQ_LEN,
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=[]
            )
            tokenizer.load_vocab(tokenizer_model_prefix+"_vocab.json")
        x_test = tokenizer.encode(x_test_raw)

        # dump X_train, X_test, y_train and y_test
        np.save(os.path.join(datafolders[run_type], f"x_train_{SUFFIX}.npy"), x_train)
        np.save(os.path.join(datafolders[run_type], f"x_test_{SUFFIX}.npy"), x_test)
        np.save(os.path.join(datafolders[run_type], f"y_train_{SUFFIX}.npy"), y_train)
        np.save(os.path.join(datafolders[run_type], f"y_test_{SUFFIX}.npy"), y_test)

        # ============= DEFINE MODEL =============
        with open(os.path.join(datafolders[run_type], f"tokenizer_{VOCAB}_vocab.json")) as f:
            nebula_vocab = json.load(f)
        models[run_type]['class'] = TransformerEncoderChunks
        models[run_type]['config'] = {
            "vocab_size": len(nebula_vocab),
            "maxlen": SEQ_LEN,
            "chunk_size": 64,
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": 10, # binary classification
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
            "loss_function": CrossEntropyLoss(),
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
        logging.warning(f" [!!!] Starting CV over {run_type}!")
        
        x_train = np.load(os.path.join(datafolders[run_type], f"x_train_{SUFFIX}.npy"))
        y_train = np.load(os.path.join(datafolders[run_type], f"y_train_{SUFFIX}.npy"))
        
        model_class = models[run_type]['class']
        model_config = models[run_type]['config']
        
        if LIMIT:
            x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED)
            x_train = x_train[:LIMIT]
            y_train = y_train[:LIMIT]
            
        set_random_seed(RANDOM_SEED)
        cv_outfolder = os.path.join(out_folder_root, f"cv_{run_type}_lim{LIMIT}_r{RANDOM_SEED}_t{TIME_BUDGET}")
        if os.path.exists(cv_outfolder):
            logging.warning(f" [!] CV output folder {cv_outfolder} already exists, skipping!")
            continue
        cv = CrossValidation(
            model_trainer_class,
            model_trainer_config,
            model_class,
            model_config,
            output_folder_root=cv_outfolder,
            false_positive_rates=[]
        )
        cv.run_folds(
            x_train, 
            y_train, 
            epochs=None, # time budget specified
            folds=3, 
            random_state=RANDOM_SEED
        )
        cv.dump_metrics(prefix=f"{run_type}")
        cv.log_avg_metrics()

        del cv, x_train, y_train
        clear_cuda_cache()
