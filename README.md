# Project Nebula

<center><img src="https://cdn.eso.org/images/screen/eso1205ec.jpg" width="400"></center>

## Description

Behavioral intrusion detection system that is capable of self-supervised learning from unlabeled data. **Work in progress.**

Quasi-functional implementation:

- Portable Executable (PE) classifier under `nebula.PEHybridClassifier` class. It uses:
  - (1) 1D Convolutional Neural Network (CNN) for dynamic PE analysis. Behavior is obtained using *speakeasy* emulator (see `nebula.preprocessing.PEDynamicFeatureExtractor`), with analysis of:
    - sequence of API calls, their arguments and return values;
    - File system modifications;
    - Registry modifications;
    - Network connections.
  - (2) Fully connected neural network (FFNN) for analysis of static features as discussed in EMBER paper (Anderson and Roth, <https://arxiv.org/abs/1804.04637>).

```python
from nebula import PEHybridClassifier

model = PEHybridClassifier()
pe = r"C:\windows\syswow64\xcopy.exe"
staticFeatures, dynamicFeatures = model.preprocess(pe)
logits = model(staticFeatures, dynamicFeatures)
```

### Installation

```bash
pip install git+https://github.com/dtrizna/nebula
```

## PE classifier configuration evaluations

Evaluation code is available in `dev` branch, under `evaluation`.

<center><img src="img\_crossValidationPlots\_modelArchitectureComparison.png" width=900>

<img src="img\_crossValidationPlots\_vocabularySizeComparison.png" width=800>

<img src="img\_crossValidationPlots\_fullVocabSizeMaxLenHeatmap_fpr_0_001.png" width=800>

<img src="img\_crossValidationPlots\_PreProcessingComparison.png" width=800></center>

## Pre-training with self-supervised techniques

### Masked Language Model

Implementation is under `nebula.pretraining.MaskedLanguageModel` class.

Evaluation done using `nebula.pretraining.SelfSupervisedPretraining` class that implements Cybersecurity Evaluation Framework for semisupervised learning (CEF-SSL) framework, introduced by Apruzzese et al. in <https://arxiv.org/pdf/2205.08944.pdf>. It suggests to perform data splits $N$ times as follows, where $\mathbb{U}$ is used for pre-training, $\mathbb{L}$ for downstream task, and $\mathbb{F}$ for final evaluation:

<center><img src="img\_maskedLanguageModelPlots\datasetSplit.png" width=300></center>

Results with even brief pre-training (5 epochs on single GPU laptop) are unclear. On trainig set (i.e. $\mathbb{L}$) metrics are improved:

<center><img src="img\_maskedLanguageModelPlots\unlabeledDataSize_0.9_preTrain_5_downStream_2_nSplits_5_1672239000_trainSetStats.png" width=800></center>

Yet no significant benefit on test set (i.e. $\mathbb{F}$) are observed:

<center><img src="img\_maskedLanguageModelPlots\unlabeledDataSize_0.8_preTrain_10_downStream_3_nSplits_5_1672232495_testSetStats.png" width=800></center>

Pre-training was brief, done on a single laptop, with futher tests required.
