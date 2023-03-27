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
from tqdm import tqdm

from nebula.preprocessing.wrappers import (
    parse_cruparamer_xmlsample,
    preprocess_nebula_cruparamer,
    preprocess_quovadis_cruparamer,
    preprocess_neurlux,
    preprocess_dmds_cruparamer
)

from nebula import ModelTrainer
from nebula.models.neurlux import NeurLuxModel
from nebula.models.quovadis import QuoVadisModel
from nebula.models import TransformerEncoderChunks
from nebula.models.dmds import DMDSGatedCNN
from nebula.evaluation import CrossValidation
from nebula.misc import set_random_seed, clear_cuda_cache

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

LIMIT = None

NEBULA_VOCAB = 50000
NEURLUX_VOCAB = 10000
QUO_VADIS_TOP_API = 600

TIME_BUDGET = 5 # minutes
FOLDS = 3

RANDOM_SEED = 1763
INFOLDER = "out_cruparamer" # if data is processed already

SEQ_LEN = 512

DATA_DIR = os.path.join(REPO_ROOT, "data", "data_raw", "CruParamer")

CRUPARAMER_TRAIN_1 = os.path.join(DATA_DIR, "Datacon2019-Malicious-Code-DataSet-Stage1", "train", "black")
CRUPARAMER_TRAIN_0 = os.path.join(DATA_DIR, "Datacon2019-Malicious-Code-DataSet-Stage1", "train", "white")
CRUPARAMER_TEST = os.path.join(DATA_DIR, "Datacon2019-Malicious-Code-DataSet-Stage1", "test")

if __name__ == "__main__":
    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_cruparamer_{FOLDS}_folds_{int(time.time())}"
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
    datafolders['nebula'] = os.path.join(out_folder_root, f"nebula_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['neurlux'] = os.path.join(out_folder_root, f"neurlux_vocab_{NEURLUX_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['quovadis'] = os.path.join(out_folder_root, f"quovadis_vocab_{QUO_VADIS_TOP_API}_seqlen_{SEQ_LEN}")
    datafolders['dmds'] = os.path.join(out_folder_root, f"dmds_vocab_seqlen_{SEQ_LEN}")
    
    # =========== PARSING RAW REPORTS IN MEMORY ==============
    logging.warning(" [*] Parsing malicious reports...")
    malicious_reports = []
    malicious_reports_api_only = []
    dmds_reports_malicious = []
    for sample in tqdm(os.listdir(CRUPARAMER_TRAIN_1)[:LIMIT]):
        sample_fullpath = os.path.join(CRUPARAMER_TRAIN_1, sample)
        report_full, report_apis_only = parse_cruparamer_xmlsample(sample_fullpath)
        malicious_reports.append(" ".join(report_full))
        dmds_reports_malicious.append(report_full)
        malicious_reports_api_only.append(report_apis_only)
    
    logging.warning(" [*] Parsing benign reports...")
    benign_reports = []
    benign_reports_api_only = []
    dmds_reports_benign = []
    for sample in tqdm(os.listdir(CRUPARAMER_TRAIN_0)[:LIMIT]):
        sample_fullpath = os.path.join(CRUPARAMER_TRAIN_0, sample)
        report_full, report_apis_only = parse_cruparamer_xmlsample(sample_fullpath)
        benign_reports.append(" ".join(report_full))
        dmds_reports_benign.append(report_full)
        benign_reports_api_only.append(report_apis_only)
    
    y = np.array([1] * len(malicious_reports) + [0] * len(benign_reports), dtype=np.int8)
    reports = malicious_reports + benign_reports
    reports_apis_only = malicious_reports_api_only + benign_reports_api_only
    dmds_reports = dmds_reports_malicious + dmds_reports_benign

    logging.warning(" [*] Parsing test reports...")
    reports_test = []
    reports_apis_only_test = []
    dmds_reports_test = []
    for sample in tqdm(os.listdir(CRUPARAMER_TEST)[:LIMIT]):
        sample_fullpath = os.path.join(CRUPARAMER_TEST, sample)
        report_full, report_apis_only = parse_cruparamer_xmlsample(sample_fullpath)
        reports_test.append(" ".join(report_full))
        dmds_reports_test.append(report_full)
        reports_apis_only_test.append(report_apis_only)
    
    y_test = np.array([1] * len(reports_test), dtype=np.int8)

    # =========== 'dmds' & 'cruparamer' preprocessing
    preprocess_dmds_cruparamer(
        dmds_reports,
        y=y,
        limit=LIMIT,
        outfolder=datafolders['dmds'],
        seq_len=SEQ_LEN,
    )
    preprocess_dmds_cruparamer(
        dmds_reports_test,
        y=y_test,
        limit=LIMIT,
        outfolder=datafolders['dmds'],
        seq_len=SEQ_LEN,
        suffix="test"
    )

    # =========== 'nebula' & 'cruparamer' preprocessing
    logging.warning(" [*] Preprocessing reports Nebula style...")
    _ = preprocess_nebula_cruparamer(
        events=reports,
        y=y,
        limit=LIMIT,
        outfolder=datafolders['nebula'],
        seq_len=SEQ_LEN,
        vocab_size=NEBULA_VOCAB,
    )
    nebula_tokenizer = os.path.join(datafolders['nebula'], f"tokenizer_{NEBULA_VOCAB}.model")
    _ = preprocess_nebula_cruparamer(
        events=reports_test,
        y=y_test,
        limit=LIMIT,
        outfolder=datafolders['nebula'],
        seq_len=SEQ_LEN,
        vocab_size=NEBULA_VOCAB,
        tokenizer_model=nebula_tokenizer
    )

    # =========== 'neurlux' & 'cruparamer' preprocessing
    _, neurlux_vocab_file = preprocess_neurlux(
        reports,
        y,
        outfolder=datafolders['neurlux'],
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN,
        limit=LIMIT,
        in_memory=True
    )
    _ = preprocess_neurlux(
        reports_test,
        y_test,
        vocab_file=neurlux_vocab_file,
        outfolder=datafolders['neurlux'],
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN,
        limit=LIMIT,
        in_memory=True
    )

    # ============ 'quo vadis' & 'cruparamer' preprocessing
    _, quovadis_vocab_file = preprocess_quovadis_cruparamer(
        reports_apis_only,
        y,
        outfolder=datafolders['quovadis'],
        seq_len=SEQ_LEN,
        top_api=QUO_VADIS_TOP_API,
        limit=LIMIT
    )
    _ = preprocess_quovadis_cruparamer(
        reports_apis_only_test,
        y_test,
        vocab_file=quovadis_vocab_file,
        outfolder=datafolders['quovadis'],
        seq_len=SEQ_LEN,
        top_api=QUO_VADIS_TOP_API,
        limit=LIMIT
    )

    # ============= DEFINE MODELS =============
    models = defaultdict(dict)

    with open(neurlux_vocab_file) as f:
        neurlux_vocab = json.load(f)
    models['neurlux']['class'] = NeurLuxModel
    models['neurlux']['config'] = {
        "embedding_dim": 256,
        "vocab_size": len(neurlux_vocab),
        "seq_len": SEQ_LEN,
    }

    models['quovadis']['class'] = QuoVadisModel
    models['quovadis']['config'] = {
        "vocab": quovadis_vocab_file,
        "seq_len": SEQ_LEN
    }
    
    with open(os.path.join(datafolders['nebula'], f"tokenizer_{NEBULA_VOCAB}_vocab.json")) as f:
        nebula_vocab = json.load(f)
    models['nebula']['class'] = TransformerEncoderChunks
    models['nebula']['config'] = {
        "vocab_size": len(nebula_vocab),
        "maxlen": SEQ_LEN,
        "chunk_size": 64,
        "dModel": 64,  # embedding & transformer dimension
        "nHeads": 8,  # number of heads in nn.MultiheadAttention
        "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
        "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "numClasses": 1, # binary classification
        "hiddenNeurons": [64],
        "layerNorm": False,
        "dropout": 0.3,
        "mean_over_sequence": False,
        "norm_first": True
    }
    
    models['dmds']['class'] = DMDSGatedCNN
    models['dmds']['config'] = {
        "ndim": 98,
        "seq_len": SEQ_LEN,
    }

    device = "cuda" if cuda.is_available() else "cpu"
    model_trainer_class = ModelTrainer
    model_trainer_config = {
        "device": device,
        "model": None, # will be set later within CrossValidation class
        "loss_function": BCEWithLogitsLoss(),
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
            folds=FOLDS, 
            random_state=RANDOM_SEED
        )
        cv.dump_metrics(prefix=f"{run_name}")
        cv.log_avg_metrics()

        del cv, x_train, y_train
        clear_cuda_cache()
