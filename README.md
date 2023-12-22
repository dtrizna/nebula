# Project Nebula

<center><img src="img/eso1205ec.jpg" width="400"></center>

## Description

This repository is part of the [Nebula: Self-Attention for Dynamic Malware Analysis](https://arxiv.org/abs/2310.10664) project.

All Nebula and alternative dynamic malware analysis models are under `nebula/models` directory.

Examples of usage are under `scripts/` directory.

### Installation

```bash
pip install git+https://github.com/dtrizna/nebula
```

### Usage Example

```python
from nebula import Nebula

# model setup
TOKENIZER = "bpe" # supports: ["bpe", "whitespace"]
nebula = Nebula(
    vocab_size = 50000, # pre-trained only for 50k
    seq_len = 512, # pre-trained only for 512
    tokenizer = TOKENIZER,
)

# 0. EMULATE IT: SKIP IF YOU HAVE JSON REPORTS ALREADY
PE = r"C:\Windows\System32\calc.exe"
report = nebula.dynamic_analysis_pe_file(PE)
# 1. PREPROCESS EMULATED JSON REPORT AS ARRAY
x_arr = nebula.preprocess(report)
# 2. PASS THROUGH PYTORCH MODEL
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

## Pre-training with self-supervised techniques

### Masked Language Model

Implementation is under `nebula.pretraining.MaskedLanguageModel` class.

Evaluation done using `nebula.evaluation.SelfSupervisedPretraining` class that implements Cybersecurity Evaluation Framework for semisupervised learning (CEF-SSL) framework, introduced by Apruzzese et al. in <https://arxiv.org/pdf/2205.08944.pdf>. It suggests to perform data splits $N$ times, where $\mathbb{U}$ is used for pre-training, $\mathbb{L}$ for downstream task, and $\mathbb{F}$ for final evaluation.
