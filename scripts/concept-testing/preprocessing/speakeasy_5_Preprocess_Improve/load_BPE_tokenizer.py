import os
import json
import sys
REPO_ROOT = "."
sys.path.append(REPO_ROOT)
from nebula import JSONTokenizerBPE

tokenizer = JSONTokenizerBPE(
    model_path=os.path.join(REPO_ROOT, r"nebula\objects\bpe_50000_sentencepiece.model"),
)

path = os.path.join(REPO_ROOT, r"data\data_filtered\speakeasy_trainset_BPE_10k\speakeasy_VocabSize_10000_tokenizer.vocab")
tokenizer.load_vocab(vocab=path)
v = tokenizer.vocab
with open(os.path.join(REPO_ROOT, r"nebula\objects\bpe_10000_vocab.json"), "w") as f:
    json.dump(v, f, indent=4)
