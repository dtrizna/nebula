import os
import sys
import json

import logging
logging.basicConfig(level=logging.INFO)

# correct path to repository root
if os.path.basename(os.getcwd()) == "scripts":
    REPOSITORY_ROOT = os.path.join(os.getcwd(), "..")
else:
    REPOSITORY_ROOT = os.getcwd()
sys.path.append(REPOSITORY_ROOT)

from nebula.misc import fix_random_seed
fix_random_seed(0)

# TAKE A PE SAMPLE
PE = "path_to_exe"
TOKENIZER = "whitespace" # supports: ["bpe", "whitespace", "wordpunct"]

# ===================
# PREPROCESSING
# ===================
from nebula import PEDynamicFeatureExtractor
extractor = PEDynamicFeatureExtractor()


# 0. EMULATE IT -- SKIP IF YOU HAVE JSON REPORT ALREADY !!!
# emulation_report = extractor.emulate(path=PE)


# 1. FILTER AND NORMALIZE IT
emulation_report = os.path.join(REPOSITORY_ROOT, r"emulation\reportSample_EntryPoint_ransomware.json")
logging.info(f" [*] Filtering and normalizing report...")
filtered_report = extractor.filter_and_normalize_report(emulation_report)


# 2. TOKENIZE IT
from nebula.preprocessing import JSONTokenizerBPE, JSONTokenizerNaive

logging.info(f" [*] Initializing tokenizer...")
if TOKENIZER in ["whitespace", "wordpunct"]:
    with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects", "speakeasy_whitespace_50000_vocab.json")) as f:
        vocab = json.load(f)

    tokenizer = JSONTokenizerNaive(
        vocab_size=len(vocab),
        seq_len=512,
        vocab=vocab,
        type=TOKENIZER
    )

if TOKENIZER == "bpe":
    pretrained_bpe_model = os.path.join(REPOSITORY_ROOT, "nebula", "objects", "speakeasy_BPE_50000_sentencepiece.model")
    with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects", "speakeasy_BPE_50000_sentencepiece_vocab.json")) as f:
        vocab = json.load(f)

    tokenizer = JSONTokenizerBPE(
        vocab_size=len(vocab),
        seq_len=512,
        model_path=pretrained_bpe_model
    )
logging.info(f" [!] Tokenizer ready.")


# 3. ENCODE IT
logging.info(f" [*] Tokenizing and encoding report..."	)
if TOKENIZER in ["whitespace", "wordpunct"]:
    tokenized_report = tokenizer.tokenize(filtered_report)
    encoded_report = tokenizer.encode(
        tokenized_report,
        tokenize=False,
        pad=True
    )
if TOKENIZER == "bpe":
    tokenized_report = tokenizer.tokenize(filtered_report)
    encoded_report = tokenizer.encode(filtered_report, pad=True)

from torch import Tensor
x = Tensor(encoded_report).long()
logging.info(f" [!] Report encoded with shapes: {x.shape}")

# ===================
# MODELING
# ===================

logging.info(f" [*] Loading model...")
from nebula.models import TransformerEncoderChunks
model_config = {
    "vocab_size": len(vocab),
    "maxlen": 512,
    "chunk_size": 64, # input splitting to chunks
    "dModel": 64,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64], # classifier ffnn dims
    "layerNorm": False,
    "dropout": 0.3,
    "mean_over_sequence": False,
    "norm_first": True
}
model = TransformerEncoderChunks(**model_config)
if TOKENIZER == "bpe":
    model_path = os.path.join(REPOSITORY_ROOT, r"nebula\objects\speakeasy_BPE_50000_torch.model")
if TOKENIZER in ["whitespace", "wordpunct"]:
    model_path = os.path.join(REPOSITORY_ROOT, r"nebula\objects\speakeasy_whitespace_50000_torch.model")
from torch import load
model.load_state_dict(load(model_path))
logging.info(f" [!] Model ready.")

logit = model(x)
from torch import sigmoid
prob = sigmoid(logit)
print(f"\n[!!!] Probability of being malicious: {prob.item():.3f} | Logit: {logit.item():.3f}")

TRAIN_SAMPLE = False
if TRAIN_SAMPLE:
    # ===================
    # OPTIONAL !!!!!
    # ===================
    # TRAINING
    # ===================
    from torch import cuda
    from torch.optim import AdamW
    from torch.nn import BCEWithLogitsLoss
    from nebula import ModelTrainer

    TIME_BUDGET = 5 # minutes
    device = "cuda" if cuda.is_available() else "cpu"
    model_trainer_config = {
        "device": device,
        "model": model,
        "loss_function": BCEWithLogitsLoss(),
        "optimizer_class": AdamW,
        "optimizer_config": {"lr": 3e-4},
        "optim_scheduler": None,
        "optim_step_budget": None,
        "outputFolder": "./out",
        "batchSize": 96,
        "verbosity_n_batches": 100,
        "clip_grad_norm": 1.0,
        "n_batches_grad_update": 1,
        "time_budget": int(TIME_BUDGET*60) if TIME_BUDGET else None,
    }

    # 3. TRAIN
    model_trainer = ModelTrainer(**model_trainer_config)
    # model_trainer.train(x, y)