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

from nebula.models.neurlux import NeurLuxModel
from nebula.models import TransformerEncoderChunks
from nebula.misc import set_random_seed, clear_cuda_cache

from nebula.lit_utils import LitTrainerWrapper
from nebula.misc import fix_random_seed

from torch import cuda
from torch.nn import CrossEntropyLoss

# TEST
# LIMIT = 1000
# INFOLDER = None # if data is processed already
# NEBULA_VOCAB = 500
# NEURLUX_VOCAB = 500

# PROD
LIMIT = None
INFOLDER = os.path.join(REPO_ROOT, "evaluation\paper_sota\out_family_downscale_avast_PROD")
RUN_PREFIX = "dropout0.3_"
NEBULA_VOCAB = 50000
NEURLUX_VOCAB = 10000

RANDOM_SEED = 1763
SEQ_LEN = 512
TRAIN_TEST_SPLIT_DATE = '2019-08-01'

# x
folder = os.path.join(REPO_ROOT, r"..\..\Data\Avast-CTU\Public_Avast_CTU_CAPEv2_Dataset_Small\public_small_reports")
EXAMPLE_PATHS = [os.path.join(folder, x) for x in os.listdir(folder)[:LIMIT]]
# y
label_file = os.path.join(REPO_ROOT, r"..\..\Data\Avast-CTU\Public_Avast_CTU_CAPEv2_Dataset_Small\public_labels.csv")
LABEL_FIELD = 'classification_family'
LABEL_TABLE = read_csv(label_file)
LABEL_MAP = dict(zip(
    sorted(LABEL_TABLE[LABEL_FIELD].unique()),
    list(range(LABEL_TABLE[LABEL_FIELD].nunique()))
))
AVAST_LABEL_IDX_TO_FAMILY = {v: k for k, v in LABEL_MAP.items()}

if __name__ == "__main__":

    fix_random_seed(RANDOM_SEED)

    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_family_downscale_avast_{int(time.time())}"
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
    datafolders['nebulabpe'] = os.path.join(out_folder_root, f"data_nebulabpe_avast_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['nebulawht'] = os.path.join(out_folder_root, f"data_nebulawht_avast_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['neurlux'] = os.path.join(out_folder_root, f"data_neurlux_avast_vocab_{NEURLUX_VOCAB}_seqlen_{SEQ_LEN}")
    
    # =========== reading data
    if os.path.exists(datafolders['nebulabpe']) and os.path.exists(datafolders['nebulawht']) and os.path.exists(datafolders['neurlux']):
        logging.warning(f" [!] Data already exists in {datafolders['nebulabpe']}, {datafolders['nebulawht']}, {datafolders['neurlux']}. Skipping data preprocessing...")
        y_train = np.load(os.path.join(datafolders['nebulabpe'], "y_train_full.npy"))
        y_test = np.load(os.path.join(datafolders['nebulabpe'], "y_test_full.npy"))
    else:
        capa_normalizer = {
            "resolved_apis": lambda x: x.lower(),
            "mutexes": lambda x: x.lower()
        }

        X_raw_train = []
        X_raw_test = []
        y_train = []
        y_test = []
        logging.warning(f" [*] Reading data from {len(EXAMPLE_PATHS)} samples...")
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
            outfolder=datafolders['nebulabpe'],
            tokenizer_type='bpe'
        )
        preprocess_nebula_avast(
            X_raw_test,
            y_test,
            limit=LIMIT,
            vocab_size=NEBULA_VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders['nebulabpe'],
            tokenizer_type='bpe',
            tokenizer_model=os.path.join(datafolders['nebulabpe'], f"tokenizer_{NEBULA_VOCAB}.model"),
        )
        # =========== 'nebula' whitespace preprocessing
        preprocess_nebula_avast(
            X_raw_train,
            y_train,
            limit=LIMIT,
            vocab_size=NEBULA_VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders['nebulawht'],
            tokenizer_type='whitespace'
        )
        preprocess_nebula_avast(
            X_raw_test,
            y_test,
            limit=LIMIT,
            vocab_size=NEBULA_VOCAB,
            seq_len=SEQ_LEN,
            outfolder=datafolders['nebulawht'],
            tokenizer_type='whitespace',
            tokenizer_model=os.path.join(datafolders['nebulawht'], f"tokenizer_{NEBULA_VOCAB}_vocab.json"),
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

    device = "cuda" if cuda.is_available() else "cpu"
    n_unique_y = len(set(y_train))

    # =============== SUBSAMPLED FAMILY LOOP =======================
    for nr_of_families in range(3, n_unique_y+1):
        logging.warning(f" [*] Multiclass classification with {nr_of_families} classes...")

        set_random_seed(RANDOM_SEED)
        # subsample random number_of_families labels from y_train and
        subsampled_labels = np.random.choice(np.unique(y_train), size=nr_of_families, replace=False)
        subsampled_families = [AVAST_LABEL_IDX_TO_FAMILY[x] for x in subsampled_labels]
        logging.warning(f" [!] Sampled labels: {subsampled_labels} | Families: {subsampled_families}")

        # use this to map y_subsampled to range(0, nr_of_families)
        label_mapping = {label: index for index, label in enumerate(subsampled_labels)}

        # ============= DEFINE MODELS =============
        models = defaultdict(dict)

        # with open(os.path.join(datafolders['neurlux'], f"vocab_{NEURLUX_VOCAB}.json")) as f:
        #     neurlux_vocab = json.load(f)
        # models['neurlux']['class'] = NeurLuxModel
        # models['neurlux']['config'] = {
        #     "vocab_size": len(neurlux_vocab),
        #     "seq_len": SEQ_LEN,
        #     "num_classes": nr_of_families
        # }

        with open(os.path.join(datafolders['nebulabpe'], f"tokenizer_{NEBULA_VOCAB}_vocab.json")) as f:
            nebula_vocab = json.load(f)
        models['nebulabpe']['class'] = TransformerEncoderChunks
        models['nebulabpe']['config'] = {
            "vocab_size": len(nebula_vocab),
            "maxlen": SEQ_LEN,
            "chunk_size": 64,
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": nr_of_families,
            "classifier_head": [64],
            "layerNorm": False,
            "dropout": 0.3,
            "pooling": "mean",
            "norm_first": True
        }

        with open(os.path.join(datafolders['nebulawht'], f"tokenizer_{NEBULA_VOCAB}_vocab.json")) as f:
            nebula_vocab = json.load(f)
        models['nebulawht']['class'] = TransformerEncoderChunks
        models['nebulawht']['config'] = {
            "vocab_size": len(nebula_vocab),
            "maxlen": SEQ_LEN,
            "chunk_size": 64,
            "dModel": 64,  # embedding & transformer dimension
            "nHeads": 8,  # number of heads in nn.MultiheadAttention
            "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
            "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            "numClasses": nr_of_families,
            "classifier_head": [64],
            "layerNorm": False,
            "dropout": 0.5,
            "pooling": "mean",
            "norm_first": True
        }

        # ============= TRAINING LOOP =============
        for run_name in models.keys():
            set_random_seed(RANDOM_SEED)

            out_folder = os.path.join(out_folder_root, f"training_{RUN_PREFIX}{run_name}")
            family_folder = os.path.join(out_folder, f"{run_name}_nr_families_{nr_of_families}_csv")
            if os.path.exists(family_folder):
                logging.warning(f" [!] Skipping... Output folder for run {run_name} already exists: {family_folder}")
                continue
            
            # =============== DATA ===============
            suffix = LIMIT if LIMIT else "full"
            x_train = np.load(os.path.join(datafolders[run_name], f"x_train_{suffix}.npy"))
            y_train = np.load(os.path.join(datafolders[run_name], f"y_train_{suffix}.npy"))
            x_test = np.load(os.path.join(datafolders[run_name], f"x_test_{suffix}.npy"))
            y_test = np.load(os.path.join(datafolders[run_name], f"y_test_{suffix}.npy"))
            
            if LIMIT:
                x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED)
                x_train = x_train[:LIMIT]
                y_train = y_train[:LIMIT]
                x_test, y_test = shuffle(x_test, y_test, random_state=RANDOM_SEED)
                x_test = x_test[:LIMIT]
                y_test = y_test[:LIMIT]

            # preserve only indices in x_train / y_train with these labels
            x_train_indices = np.isin(y_train, subsampled_labels)
            x_train_subsample = x_train[x_train_indices]
            y_train_subsample = y_train[x_train_indices]
            y_train_subsample_mapped = np.array([label_mapping[label] for label in y_train_subsample])

            x_test_indices = np.isin(y_test, subsampled_labels)
            x_test_subsample = x_test[x_test_indices]
            y_test_subsample = y_test[x_test_indices]
            y_test_subsample_mapped = np.array([label_mapping[label] for label in y_test_subsample])

            logging.warning(f" [!] Sizes of full sets: train -- {x_train.shape}, test -- {x_test.shape}")
            logging.warning(f" [!] Sizes of subsampled sets: train -- {x_train_subsample.shape}, test -- {x_test_subsample.shape}")
            
            # ============= TRAINING =============
            logging.warning(f" [!!!] Starting training: {run_name}!")
            model_class = models[run_name]['class']
            model_config = models[run_name]['config']
            model = model_class(**model_config)
                            
            lit_trainer = LitTrainerWrapper(
                pytorch_model=model,
                name=f"{run_name}_nr_families_{nr_of_families}",
                log_folder=out_folder,
                epochs=20,
                device=device,
                log_every_n_steps=10,
                scheduler="onecycle",
                batch_size=128,
                dataloader_workers=4,
                gradient_clip_val=1.0,
                loss=CrossEntropyLoss(),
                out_classes=nr_of_families
            )
            train_loader = lit_trainer.create_dataloader(x_train_subsample, y_train_subsample_mapped)
            test_loader = lit_trainer.create_dataloader(x_test_subsample, y_test_subsample_mapped, shuffle=False)
            lit_trainer.train_lit_model(train_loader=train_loader, val_loader=test_loader)

            del x_train_subsample, y_train_subsample, x_test_subsample, y_test_subsample
            del y_train_subsample_mapped, y_test_subsample_mapped
            del x_train, x_test
            clear_cuda_cache()
        
        del models, subsampled_labels, subsampled_families
