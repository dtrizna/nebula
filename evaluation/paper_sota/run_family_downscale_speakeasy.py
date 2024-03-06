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

from nebula.preprocessing.wrappers import (
    preprocess_nebula_speakeasy,
    preprocess_quovadis_speakeasy,
    preprocess_neurlux,
    preprocess_dmds_speakeasy
)

from nebula.models.neurlux import NeurLuxModel
from nebula.models.quovadis import QuoVadisModel
from nebula.models import TransformerEncoderChunks
from nebula.models.dmds import DMDSGatedCNN
from nebula.misc import set_random_seed, clear_cuda_cache
from nebula.constants import SPEAKEASY_LABELMAP

from torch import cuda
from torch.nn import CrossEntropyLoss
from nebula.lit_utils import LitTrainerWrapper
from nebula.misc import fix_random_seed

LABEL_SAMPLING_ORDER = ["clean", "backdoor", "ransomware", "trojan", "coinminer", "rat", "dropper", "keylogger"]
EPOCHS = 20
RUN_PREFIX = f"ordered_ep{EPOCHS}_"

LIMIT = None
INFOLDER = "out_family_downscale_speakeasy_PROD" # if data is processed already

NEBULA_VOCAB = 50000
NEURLUX_VOCAB = 10000
QUO_VADIS_TOP_API = 600

RANDOM_SEED = 1763
SEQ_LEN = 512
SPEAKEASY_TRAINSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_trainset")
SPEAKEASY_TESTSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_testset")
SPEAKEASY_LABEL_IDX_TO_FAMILY = {v: k for k, v in SPEAKEASY_LABELMAP.items()}

if __name__ == "__main__":
    
    fix_random_seed(RANDOM_SEED)

    if INFOLDER:
        out_folder_root = INFOLDER
    else:
        out_folder_root = f"out_family_downscale_speakeasy_{int(time.time())}"
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
    datafolders['nebulabpe'] = os.path.join(out_folder_root, f"data_nebula_speakeasy_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['nebulawht'] = os.path.join(out_folder_root, f"data_nebulawht_speakeasy_vocab_{NEBULA_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['neurlux'] = os.path.join(out_folder_root, f"data_neurlux_speakeasy_vocab_{NEURLUX_VOCAB}_seqlen_{SEQ_LEN}")
    datafolders['quovadis'] = os.path.join(out_folder_root, f"data_quovadis_speakeasy_vocab_{QUO_VADIS_TOP_API}_seqlen_{SEQ_LEN}")
    datafolders['dmds'] = os.path.join(out_folder_root, f"data_dmds_speakeasy_vocab_seqlen_{SEQ_LEN}")
    
    # =========== 'nebula' & 'speakeasy' preprocessing
    logging.warning("Preprocessing 'nebula' & 'speakeasy'...")
    _, y_train, y_paths_train, = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TRAINSET_PATH,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebulabpe'],
        multiclass=True
    )
    _, y_test, y_paths_test = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TESTSET_PATH,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebulabpe'],
        tokenizer_model=os.path.join(datafolders['nebulabpe'], f"tokenizer_{NEBULA_VOCAB}.model"),
        multiclass=True
    )
    
    # =========== 'nebula' & 'speakeasy' preprocessing
    _, y_train, y_paths_train, = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TRAINSET_PATH,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebulawht'],
        multiclass=True,
        tokenizer_type="whitespace"
    )
    _, y_test, y_paths_test = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TESTSET_PATH,
        limit=LIMIT,
        vocab_size=NEBULA_VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebulawht'],
        tokenizer_model=os.path.join(datafolders['nebulawht'], f"tokenizer_{NEBULA_VOCAB}_vocab.json"),
        multiclass=True,
        tokenizer_type="whitespace"
    )

    # ============= 'dmds' & 'speakeasy' preprocessing
    preprocess_dmds_speakeasy(
        y_paths_train,
        y=y_train,
        seq_len=SEQ_LEN,
        outfolder=datafolders['dmds'],
        suffix="train",
        limit=LIMIT,
    )
    preprocess_dmds_speakeasy(
        y_paths_test,
        y=y_test,
        seq_len=SEQ_LEN,
        outfolder=datafolders['dmds'],
        suffix="test",
        limit=LIMIT,
    )

    # =========== 'neurlux' & 'speakeasy' preprocessing
    _, neurlux_vocab_file = preprocess_neurlux(
        y_paths_train,
        y=y_train,
        outfolder=datafolders['neurlux'],
        limit=LIMIT,
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN

    )
    _ = preprocess_neurlux(
        y_paths_test,
        y=y_test,
        vocab_file=neurlux_vocab_file,
        outfolder=datafolders['neurlux'],
        limit=LIMIT,
        vocab_size=NEURLUX_VOCAB,
        seq_len=SEQ_LEN
    )

    # ============ 'quo vadis' & 'speakeasy' preprocessing
    _, quovadis_vocab_file = preprocess_quovadis_speakeasy(
        y_paths_train,
        y=y_train,
        seq_len=SEQ_LEN,
        top_api=QUO_VADIS_TOP_API,
        outfolder=datafolders['quovadis'],
        limit=LIMIT,
    )
    _ = preprocess_quovadis_speakeasy(
        y_paths_test,
        y=y_test,
        vocab_file=quovadis_vocab_file,
        seq_len=SEQ_LEN,
        top_api=QUO_VADIS_TOP_API,
        outfolder=datafolders['quovadis'],
        limit=LIMIT,
    )

    device = "cuda" if cuda.is_available() else "cpu"
    n_unique_y = len(set(y_train))

    # =============== SUBSAMPLED FAMILY LOOP =======================
    for nr_of_families in range(3, n_unique_y+1):
        logging.warning(f" [*] Multiclass classification with {nr_of_families} classes...")

        set_random_seed(RANDOM_SEED)
        
        # subsample random number_of_families labels from y_train and
        # subsampled_labels = np.random.choice(np.unique(y_train), size=nr_of_families, replace=False)
        # subsampled_families = [SPEAKEASY_LABEL_IDX_TO_FAMILY[x] for x in subsampled_labels]

        # try with predefined order
        subsampled_families = LABEL_SAMPLING_ORDER[:nr_of_families]
        subsampled_labels = [SPEAKEASY_LABELMAP[family] for family in subsampled_families]

        logging.warning(f" [!] Sampled labels: {subsampled_labels} | Families: {subsampled_families}")

        # use this to map y_subsampled to range(0, nr_of_families)
        label_mapping = {label: index for index, label in enumerate(subsampled_labels)}

        # ============= DEFINE MODELS =============
        models = defaultdict(dict)

        with open(neurlux_vocab_file) as f:
            neurlux_vocab = json.load(f)
        models['neurlux']['class'] = NeurLuxModel
        models['neurlux']['config'] = {
            "vocab_size": len(neurlux_vocab),
            "seq_len": SEQ_LEN,
            "num_classes": nr_of_families
        }

        # models['quovadis']['class'] = QuoVadisModel
        # models['quovadis']['config'] = {
        #     "vocab": quovadis_vocab_file,
        #     "seq_len": SEQ_LEN,
        #     "num_classes": nr_of_families
        # }

        # models['dmds']['class'] = DMDSGatedCNN
        # models['dmds']['config'] = {
        #     "seq_len": SEQ_LEN,
        #     "num_classes": nr_of_families,
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
            "dropout": 0.3,
            "pooling": "mean",
            "norm_first": True
        }
        
        # ============= TRAINING LOOP =============
        for run_name in models.keys():
            set_random_seed(RANDOM_SEED)

            out_folder = os.path.join(out_folder_root, f"{RUN_PREFIX}{run_name}")
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
                epochs=EPOCHS,
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

            # get f1 scores
            # y_train_pred = lit_trainer.predict_lit_model(train_loader)
            # f1_train = f1_score(y_train_subsample_mapped, y_train_pred, average='macro')
            # logging.warning(f"[!] Train F1: {f1_train*100:.2f}%")

            # y_test_pred = lit_trainer.predict_lit_model(test_loader)
            # f1_test = f1_score(y_test_subsample_mapped, y_test_pred, average='macro')
            # logging.warning(f"[!] Test F1: {f1_test*100:.2f}%")

            del x_train_subsample, y_train_subsample, x_test_subsample, y_test_subsample
            del y_train_subsample_mapped, y_test_subsample_mapped
            del x_train, x_test
            clear_cuda_cache()
        
        del models, subsampled_labels, subsampled_families
