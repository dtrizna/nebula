# Project Nebula

<center><img src="img/eso1205ec.jpg" width="400"></center>

## Description

This repository is part of the [Nebula: Self-Attention for Dynamic Malware Analysis](https://arxiv.org/abs/2310.10664) project.

All Nebula and alternative dynamic malware analysis models are under `nebula/models` directory.

Examples of usage are under `scripts/` directory.

Dataset used for experiments and pretraining downloadable from [kaggle.com/datasets/dmitrijstrizna/quo-vadis-malware-emulation](https://www.kaggle.com/datasets/dmitrijstrizna/quo-vadis-malware-emulation).

### Installation

Model code and Nebula pretrained objects are available as `pip` package:

```bash
pip install git+https://github.com/dtrizna/nebula
```

### Usage Example

```python
from nebula import Nebula

# 0. MODEL SETUP
nebula = Nebula(
    vocab_size = 50000, # pre-trained only for 50k
    seq_len = 512, # pre-trained only for 512
    tokenizer = "bpe" # supports: ["bpe", "whitespace"],
)

# 1. EMULATE IT: SKIP IF YOU HAVE JSON REPORTS ALREADY
pe = r"C:\Windows\System32\calc.exe"
report = nebula.dynamic_analysis_pe_file(pe)

# 2. PREPROCESS EMULATED JSON REPORT AS ARRAY
x_arr = nebula.preprocess(report)

# 3. PASS THROUGH PYTORCH MODEL
prob = nebula.predict_proba(x_arr)

print(f"\n[!] Probability of being malicious: {prob:.3f}")
```

Running this:

```bash
> python3 scripts\nebula_pe_to_preds.py

INFO:root: [!] Successfully loaded pre-trained tokenizer model!
INFO:root: [!] Loaded vocab from <REDACTED>\nebula\objects\bpe_50000_vocab.json
INFO:root: [!] Tokenizer ready!
INFO:root: [!] Model ready!

[!] Probability of being malicious: 0.001
```

## Pre-training with Self-Supervised Learning (SSL)

Nebula is capable of learning from unlabeled data using self-supervised learning (SSL) techniques. Extensive evaluation of SSL efficiency and API level interface is a subject of future work.

### Masked Language Model

Implementation is under `nebula.lit_pretraining.MaskedLanguageModelTrainer` class.

### Auto-Regressive Language Model

Implementation is under `nebula.lit_pretraining.AutoRegressiveModelTrainer` class.
