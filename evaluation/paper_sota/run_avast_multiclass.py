import os
REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

import sys
sys.path.extend([REPO_ROOT, "."])

import time
import json
import logging
import numpy as np
from tqdm import tqdm
from pandas import read_csv, to_datetime
from collections import defaultdict
from sklearn.utils import shuffle

from nebula.preprocessing.wrappers import (
    preprocess_nebula_avast,
    preprocess_neurlux
)

from nebula import ModelTrainer
from nebula.models.neurlux import NeurLuxModel
from nebula.models.quovadis import QuoVadisModel
from nebula.models import TransformerEncoderChunks
from nebula.models.dmds import DMDSGatedCNN
from nebula.evaluation import CrossValidation
from nebula.misc import set_random_seed, clear_cuda_cache

from torch import cuda
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW

LIMIT = None
TIME_BUDGET = 5 # minutes
INFOLDER = None # if data is processed already

NEBULA_VOCAB = 50000
NEURLUX_VOCAB = 10000

RANDOM_SEED = 1763
SEQ_LEN = 512
TRAIN_TEST_SPLIT_DATE = '2019-08-01'

# x
folder = os.path.join(REPO_ROOT, r"data\data_raw\Avast\Public_Avast_CTU_CAPEv2_Dataset_Small\public_small_reports")
EXAMPLE_PATHS = [os.path.join(folder, x) for x in os.listdir(folder)[:LIMIT]]
# y
label_file = os.path.join(REPO_ROOT, r"data\data_raw\Avast\Public_Avast_CTU_CAPEv2_Dataset_Small\public_labels.csv")
LABEL_FIELD = 'classification_family'
LABEL_TABLE = read_csv(label_file)
LABEL_MAP = dict(zip(
    sorted(LABEL_TABLE[LABEL_FIELD].unique()),
    list(range(LABEL_TABLE[LABEL_FIELD].nunique()))
))

if __name__ == "__main__":
    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_avast_multiclass_{int(time.time())}"
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
    datafolders['nebula_bpe'] = os.path.join(out_folder_root, f"nebula_bpe_avast_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['nebula_wht'] = os.path.join(out_folder_root, f"nebula_wht_avast_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['neurlux'] = os.path.join(out_folder_root, f"neurlux_avast_vocab_{NEURLUX_VOCAB}_seqlen_{SEQ_LEN}")
    
    # =========== reading data
    capa_normalizer = {
        "resolved_apis": lambda x: x.lower(),
        "mutexes": lambda x: x.lower()
    }

    X_raw_train = []
    X_raw_test = []
    y_train = []
    y_test = []
    for example in tqdm(EXAMPLE_PATHS):
        hhash = os.path.basename(example).replace(".json", "")
        sample_data = LABEL_TABLE[LABEL_TABLE['sha256'] == hhash].iloc[0]
        family = sample_data[LABEL_FIELD]
        
        with open(example, encoding='utf-8') as f:
            sample = json.load(f)
        sample = sample["behavior"]['summary']
        normalized_sample = {field: [capa_normalizer[field](x) for x in sample[field]] for field in capa_normalizer}
        if to_datetime(sample_data['date']) < to_datetime(TRAIN_TEST_SPLIT_DATE):
            X_raw_train.append(normalized_sample)
            y_train.append(LABEL_MAP[family])
        else:
            X_raw_test.append(normalized_sample)
            y_test.append(LABEL_MAP[family])
    
    y_train = np.array(y_train, dtype=np.int8)
    y_test = np.array(y_test, dtype=np.int8)

    # =========== 'nebula' bpe preprocessing
    preprocess_nebula_avast(
        X_raw_train,
        y_train,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula_bpe'],
        tokenizer_type='bpe'
    )
    preprocess_nebula_avast(
        X_raw_test,
        y_test,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula_bpe'],
        tokenizer_type='bpe',
        tokenizer_model=os.path.join(datafolders['nebula_bpe'], f"tokenizer_{NEBULA_VOCAB}.model"),
    )
    # =========== 'nebula' whitespace preprocessing
    preprocess_nebula_avast(
        X_raw_train,
        y_train,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula_wht'],
        tokenizer_type='whitespace'
    )
    preprocess_nebula_avast(
        X_raw_test,
        y_test,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula_wht'],
        tokenizer_type='whitespace',
        tokenizer_model=os.path.join(datafolders['nebula_wht'], f"tokenizer_{NEBULA_VOCAB}_vocab.json"),
    )

    # =========== 'neurlux' & 'avast' preprocessing
    _, neurlux_vocab_file = preprocess_neurlux(
        X_raw_train,
        y=y_train,
        outfolder=datafolders['neurlux'],
        limit=LIMIT,
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN,
        in_memory=True
    )
    _ = preprocess_neurlux(
        X_raw_test,
        y=y_test,
        vocab_file=neurlux_vocab_file,
        outfolder=datafolders['neurlux'],
        limit=LIMIT,
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN,
        in_memory=True
    )

    # ============= DEFINE MODELS =============
    models = defaultdict(dict)
    # count unique elements in y
    n_unique_y = len(set(y_train))
    logging.warning(f" [!] Multiclass classification with {n_unique_y} classes")

    with open(neurlux_vocab_file) as f:
        neurlux_vocab = json.load(f)
    models['neurlux']['class'] = NeurLuxModel
    models['neurlux']['config'] = {
        "embedding_dim": 256,
        "vocab_size": len(neurlux_vocab),
        "seq_len": SEQ_LEN,
        "num_classes": n_unique_y
    }
    
    with open(os.path.join(datafolders['nebula_bpe'], f"tokenizer_{NEBULA_VOCAB}_vocab.json")) as f:
        nebula_vocab = json.load(f)
    models['nebula_bpe']['class'] = TransformerEncoderChunks
    models['nebula_bpe']['config'] = {
        "vocab_size": len(nebula_vocab),
        "maxlen": SEQ_LEN,
        "chunk_size": 64,
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": n_unique_y,
        "classifier_head": [64],
        "layerNorm": False,
        "dropout": 0.3,
        "mean_over_sequence": False,
        "norm_first": True
    }

    with open(os.path.join(datafolders['nebula_wht'], f"tokenizer_{NEBULA_VOCAB}_vocab.json")) as f:
        nebula_vocab = json.load(f)
    models['nebula_wht']['class'] = TransformerEncoderChunks
    models['nebula_wht']['config'] = {
        "vocab_size": len(nebula_vocab),
        "maxlen": SEQ_LEN,
        "chunk_size": 64,
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": n_unique_y,
        "classifier_head": [64],
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
    for run_name in models.keys():
        cv_outfolder = os.path.join(out_folder_root, f"cv_{run_name}_lim{LIMIT}_r{RANDOM_SEED}_t{TIME_BUDGET}")
        if os.path.exists(cv_outfolder):
            logging.warning(f" [!] Skipping... CV output folder for run {run_name} already exists: {cv_outfolder}")
            continue
        
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
            random_state=RANDOM_SEED,
            false_positive_rates=[]
        )
        cv.dump_metrics(prefix=f"{run_name}")
        cv.log_avg_metrics()

        del cv, x_train, y_train
        clear_cuda_cache()
