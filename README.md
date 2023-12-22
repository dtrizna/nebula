# Project Nebula

<center><img src="img/eso1205ec.jpg" width="400"></center>

## Description

This repository is part of the [Nebula: Self-Attention for Dynamic Malware Analysis](https://arxiv.org/abs/2310.10664) project.

All Nebula and alternative dynamic malware analysis models are under `nebula/models` directory.

Examples of usage are under `scripts/` directory.

All experimental setup and results are under `evaluation/` directory, specifically:

- `paper_ablation/` contains the code for the ablation studies;
- `paper_sota/` contains the code for the comparison with state-of-the-art models;
- `explanation/` contains the code for the explainability experiments of the model;
- `adversarial/` contains the code for the adversarial robustness experiments;
- `language_modeling/` has experiments with masked language modeling.

### Installation

```bash
pip install git+https://github.com/dtrizna/nebula
```



## Pre-training with self-supervised techniques

### Masked Language Model

Implementation is under `nebula.pretraining.MaskedLanguageModel` class.

Evaluation done using `nebula.evaluation.SelfSupervisedPretraining` class that implements Cybersecurity Evaluation Framework for semisupervised learning (CEF-SSL) framework, introduced by Apruzzese et al. in <https://arxiv.org/pdf/2205.08944.pdf>. It suggests to perform data splits $N$ times, where $\mathbb{U}$ is used for pre-training, $\mathbb{L}$ for downstream task, and $\mathbb{F}$ for final evaluation.
