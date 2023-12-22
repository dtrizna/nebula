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
PE = r"C:\Windows\System32\calc.exe"
TOKENIZER = "bpe" # supports: ["bpe", "whitespace", "wordpunct"]

# ===================
# PREPROCESSING
# ===================
from nebula import PEDynamicFeatureExtractor
extractor = PEDynamicFeatureExtractor()


# 0. EMULATE IT -- SKIP IF YOU HAVE JSON REPORT ALREADY !!!
if os.path.exists(PE):
    logging.info(f" [*] Emulating {PE}...")
    emulation_report = extractor.emulate(path=PE)


# 1. FILTER AND NORMALIZE IT
# emulation_report = os.path.join(REPOSITORY_ROOT, r"emulation\report_example_entrypoint_ransomware.json")
logging.info(f" [*] Filtering and normalizing report...")
filtered_report = extractor.filter_and_normalize_report(emulation_report)


# 2. TOKENIZE IT
from nebula.preprocessing import JSONTokenizerBPE, JSONTokenizerNaive

logging.info(f" [*] Initializing tokenizer...")
if TOKENIZER in ["whitespace", "wordpunct"]:
    with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects", "whitespace_50000_vocab.json")) as f:
        vocab = json.load(f)

    tokenizer = JSONTokenizerNaive(
        vocab_size=len(vocab),
        seq_len=512,
        vocab=vocab,
        type=TOKENIZER
    )

if TOKENIZER == "bpe":
    pretrained_bpe_model = os.path.join(REPOSITORY_ROOT, "nebula", "objects", "bpe_50000_sentencepiece.model")
    with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects", "bpe_50000_vocab.json")) as f:
        vocab = json.load(f)

    tokenizer = JSONTokenizerBPE(
        vocab_size=len(vocab),
        seq_len=512,
        model_path=pretrained_bpe_model,
        vocab=vocab
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
    model_path = os.path.join(REPOSITORY_ROOT, r"nebula\objects\bpe_50000_torch.model")
if TOKENIZER in ["whitespace", "wordpunct"]:
    model_path = os.path.join(REPOSITORY_ROOT, r"nebula\objects\whitespace_50000_torch.model")
from torch import load
model.load_state_dict(load(model_path))
logging.info(f" [!] Model ready.")

logit = model(x)
from torch import sigmoid
prob = sigmoid(logit)
print(f"\n[!!!] Probability of being malicious: {prob.item():.3f} | Logit: {logit.item():.3f}")
