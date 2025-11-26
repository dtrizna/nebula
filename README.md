# Project Nebula

<center><img src="img/eso1205ec.jpg" width="400"></center>

## Description

This repository is part of the [Nebula: Self-Attention for Dynamic Malware Analysis](https://ieeexplore.ieee.org/document/10551436) publication.  
Pre-print available on [arxiv](https://arxiv.org/abs/2310.10664).

### Repository Structure

- All Nebula and alternative dynamic malware analysis models are under `nebula/models` directory.
- Examples of usage are under `scripts/` directory.
- Code related to extracting emualtion traces from raw PE samples is under `emulation/`.

### Datasets

Dataset used for experiments and pretraining downloadable from [huggingface.co/datasets/dtrizna/quovadis-speakeasy](https://huggingface.co/datasets/dtrizna/quovadis-speakeasy).

Additionally, while not used in this project directly, EMBER feature vectors for the same malware samples are available on huggingface as well: [huggingface.co/datasets/dtrizna/quovadis-ember](https://huggingface.co/datasets/dtrizna/quovadis-ember).

This should allow to perform research on static and dynamic detection methodology cross-analyses.

### Citation

If you find this code or data valuable, please cite us:

```
@ARTICLE{10551436,
  author={Trizna, Dmitrijs and Demetrio, Luca and Biggio, Battista and Roli, Fabio},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Nebula: Self-Attention for Dynamic Malware Analysis}, 
  year={2024},
  volume={19},
  number={},
  pages={6155-6167},
  keywords={Malware;Feature extraction;Data models;Analytical models;Long short term memory;Task analysis;Encoding;Malware;transformers;dynamic analysis;convolutional neural networks},
  doi={10.1109/TIFS.2024.3409083}}


@inproceedings{10.1145/3560830.3563726,
author = {Trizna, Dmitrijs},
title = {Quo Vadis: Hybrid Machine Learning Meta-Model Based on Contextual and Behavioral Malware Representations},
year = {2022},
isbn = {9781450398800},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3560830.3563726},
doi = {10.1145/3560830.3563726},
booktitle = {Proceedings of the 15th ACM Workshop on Artificial Intelligence and Security},
pages = {127–136},
numpages = {10},
keywords = {reverse engineering, neural networks, malware, emulation, convolutions},
location = {Los Angeles, CA, USA},
series = {AISec'22}
}
```

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
