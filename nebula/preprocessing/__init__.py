from .tokenization import JSONFilter, JSONTokenizerBPE, JSONTokenizerNaive
from .pe import PEStaticFeatureExtractor, PEDynamicFeatureExtractor
from .normalization import normalizeTableIP, normalizeTablePath

import json

import os
REPOSITORY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
from torch import Tensor

def load_tokenizer(type="whitespace"):
    assert type in ["whitespace", "bpe"]
    
    if type == "whitespace":
        with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects",
                "whitespace_50000_vocab.json")) as f:
            vocab = json.load(f)
    
        tokenizer = JSONTokenizerNaive(
            vocab_size=len(vocab),
            seq_len=512,
            vocab=vocab
        )
    if type == "bpe":
        with open(os.path.join(REPOSITORY_ROOT, "nebula", "objects",
                "bpe_50000_sentencepiece_vocab.json")) as f:
            vocab = json.load(f)
        
        tokenizer = JSONTokenizerBPE(
            vocab_size=len(vocab),
            seq_len=512,
            model_path=os.path.join(REPOSITORY_ROOT, "nebula", "objects",
                        "bpe_50000_sentencepiece.model")
        )
    return tokenizer


def tokenize_sample(report_path, type="whitespace", encode=True):
    extractor = PEDynamicFeatureExtractor()
    filtered_report = extractor.filter_and_normalize_report(report_path)
    tokenizer = load_tokenizer(type=type)
    tokenized_report = tokenizer.tokenize(filtered_report)
    if encode:
        encoded_report = tokenizer.encode(tokenized_report, pad=True, tokenize=False)
        x = Tensor(encoded_report).long()
        return x
    return tokenized_report
