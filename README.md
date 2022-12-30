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
    - Networ connections.
  - (2) Fully connected neural network (FFNN) for analysis of static features as discussed in EMBER paper (Anderson and Roth, <https://arxiv.org/abs/1804.04637>).

```python
model = PEHybridClassifier(
    vocabFile=vocabFile,
    speakeasyConfig=speakeasyConfigFile
)
pe = r"C:\windows\syswow64\xcopy.exe"
staticFeatures, dynamicFeatures = model.preprocess(pe)
logits = model(staticFeatures, dynamicFeatures)
```

## PE classifier configuration evaluations

<center><img src="evaluation\_crossValidationPlots\_modelArchitectureComparison.png" width=700>

<img src="evaluation\_crossValidationPlots\_vocabularySizeComparison.png" width=700>

<img src="evaluation\_crossValidationPlots\_fullVocabSizeMaxLenHeatmap_fpr_0_001.png" width=700>

<img src="evaluation\_crossValidationPlots\_PreProcessingComparison.png" width=700></center>

## Pre-training with self-supervised techniques

### Masked Language Model

Implementation is under `nebula.pretraining.MaskedLanguageModel` class.

Evaluation done using `nebula.pretraining.SelfSupervisedPretraining` class that implements Cybersecurity Evaluation Framework for semisupervised learning (CEF-SSL) framework, introduced by Apruzzese et al. in <https://arxiv.org/pdf/2205.08944.pdf>. It suggests to perform data splits $N$ times as follows, where $\mathbb{U}$ is used for pre-training, $\mathbb{L}$ for downstream task, and $\mathbb{F}$ for final evaluation:

<center><img src="work\speakeasy_3_SSL_Pretraining\datasetSplit.png" width=300></center>

Results with even brief pre-training (5 epochs on single GPU laptop) are unclear. On trainig set (i.e. $\mathbb{L}$) metrics are improved:

<center><img src="evaluation\_maskedLanguageModelPlots\unlabeledDataSize_0.9_preTrain_5_downStream_2_nSplits_5_1672239000_trainSetStats.png" width=700></center>

Yet no significant benefit on test set (i.e. $\mathbb{F}$) are observed:

<center><img src="evaluation\_maskedLanguageModelPlots\unlabeledDataSize_0.8_preTrain_10_downStream_3_nSplits_5_1672232495_testSetStats.png" width=700></center>

Pre-training was brief, done on a single laptop, with futher tests required.

## Future work

### Short ToDo list

1. Build malware classifier class with convenient API
   - ~~Add ember feature extraction to sequential emulation model.~~
   - Provide it as `pip` package.
   - Download raw PE data, build dataset, and train classifier.
2. Try efficient long-sequence attention models:
   - Reformer
   - Start transformer
3. Pre-training routines on unlabeled data according to CEF-SSL framework:
   - naive methods:
     - expectation minimization (?)
   - language modeling:
     - ~~MLM~~ (see `nebula/pretraining/MaskedLanguageModel` class)
     - GPT like $p(x_{t+1}|x_{t}\ ...\ x_{0})$
   - embedding pretraining:
     - CBOW embedding langauge modeling
     - fastText
   - evaluate:
     - ~~pre-trained vs non-pretrained: TPR, F1, LOSS~~
       - different pretraining methods for these
     - pre-training epochs
     - unlabeled/labeled data ratio
4. 3D dataset `(batch_size, seq_len, features)` -- first learn representation for event (`features`), then for sequence of events (`seq_len`)

### Detailed ToDo list

#### `SpeakEasy` dynamic analysis modelling

- ~~assess preprocessing: filtering and normalization~~
- ~~generate filtered dataset suitable for SSL pre-training~~
- ~~add clean samples from windows syswow64 folder~~
- ~~perform tokenization and encoding~~
- ~~prepare extractor and tokenizer classes~~
- ~~preprocess train and test sets~~
- try different models with labeled dataset:
  - ~~1D-CNN + Linear~~
  - ~~Transformer~~
  - ~~LSTM~~
  - ~~1D-CNN + LSTM~~
  - Reformer (!)
  - star transformer (?)
  - 1D-CNN-Transformer (?)
- cross-validate models to find best hyperparameters
  - ~~1D CNN + Linear~~
  - ~~Transformer~~
  - 1D CNN + LSTM
  - Reformer
- cross-validate preprocessing to find best data structure
  - ~~Plot vocabSize and maxLen as heatmap!~~
  - ~~clean dataset (w/o api_seq_len==1 of ordinal APIs)~~ DONE: Training set size dropped from `91096` to `76126`.

    > NOTE: Excluded `MSVBVM60.ordinal_100` which was in 15192 training files, including 874 benignware files.  

    > NOTE2: There are lots of another examples with 1 or 2 API calls only -- especially in clean folder -- decided to keep them:

    ```text
       pc                  api_name         args ret_val

    0  0x407c96  KERNEL32.GetCommandLineA   []  0x45f0

    0  0x41808b  OLEAUT32.GetModuleHandleA   [0x418190, 0x400000, 0x0, 0x0]     0x0
    1  0x41809f    OLEAUT32.GetProcAddress  [0x0, 0x418181, 0x0, 0x1212000]     0x0

    0  0x40119a         kernel32.ShowConsoleCursor  [0x4037d3, 0x40153e, 0x0, 0x0]     0x0
    1  0x40128f  kernel32.GetUserDefaultUILanguage                              []  0xffff}
    ```

  - more verbose dataset -- include additional fields
    - ~~include API call 'args'~~
    - ~~try extra preProcessing strategies:~~
      - ~~include error: (1) preprocess dataset (2) run cross validation~~
      - ~~include dns: (1) preprocess dataset (2) run cross validation~~
    Results reported under `_speakeasyPreProcessingResults.ipynb` notebook.
  - rewrite preprocessing:
    - extract separate `JSONParser` class from `PEDynamicExtractor` class that will be suitable for `auditd` dataset
    - input for `JSONParser` should be list of normalized fields, e.g.: `[api.name, api.args, ...]`. It should be able to find where JSON has list of values, pass it `json_normalize` and then filter all consequent fields
  - 3D dataset: `(batch_size, seq_len, features)` where each sequence represent:
    - API call tokens: name, ret_val, args
    - network request tokens
    - featurized registry modification
    - featurized file modification

    > NOTE: models should be adjusted to parse 3D structure. If transformer, might need positional encoding twice.

    > NOTE2! This will build ground for future work on loging data (auditd, sysmon).

  - consider in-memory data processing
    > NOTE: Need to rerun SpeakEasy on raw PEs to get this data!
- Pre-training routines on unlabeled data according to CEF-SSL framework:
  - naive methods like expectation minimization
  - pretraining embeddings:
    - fastText
    - language modelling with `nn.Embedding()`
  - language modeling:
    - MLM
    - GPT like $p(x_{t+1}|x_{t}\ ...\ x_{0})$

### `auditd` dataset and dowstream tasks

- ~~test normalization and JSON filter speeds~~ (DONE, see `tests/auditd_1_Preprocessing/_auditd_json_benchmarks.ipynb` and `tests/speakeasy_1_Preprocessing/_speakeasyTicks.ipynb`)
  - ~~any ways to optimize pandas involvement?~~
    - ~~replace pandas whatsoever -- do plain `json` or `koalas`?~~
    - ~~what takes the most time -- loading? groupby?~~

- reverse shell dataset with `process.title` content only:
  - generate augmented reverse shells as malicious samples: (a) train -- one set of reverse shell binaries (b) val -- another binaries
  - random split of unique legitimate commands to train/val
  
  > NOTE: Be sure that generated commands resemble as close as possible to auditd telemetry (i.e. normalization)!

- `auditd` JSON dataset:
  - write classes for preprocessing
  - prepare samples for Cartesian bootstraping
  - build train and test datasets
  - sequence processing -- based on: (a) time (b) nr. of events ?
  
### Contextual embeddings

- ~~write pipeline for fastText training~~ (DONE, see `tests/modeling/fastText/_auditd_d_fasttext_full.ipynb`)
  - ~~anomaly detection on red team host~~ (DONE, see `tests/modeling/fastText/_auditd_c_fasttext.ipynb`)

- cross validation of `fasttext`:
  - `fasttext` as embeddings with models instead of `nn.Embedding()` - tokenization & embeddings setup:
    - for SpeakEasy reports:
      - custom whitespace + embeddings + model
      - custom whitespace + fastText + model
    - for `auditd` data (e.g. reverse shell dataset):
      - sentencePiece + embeddings + model
      - sentencePiece + fastText + model
        > Note: is it possible to define custom tokenizer for fastText?
  - with default hyperparameters:
    - embedding weights setup:
      - with trainable from scratch, random initialization
      - pre-trained `fasttext` model, frozen (not updated)
      - pre-trained `fasttext` model, but updated during supervised phase
  - fasttext `MAX_LINE_SIZE` analysis?
    - is it a problem at all? test on speakeasy reports
  - `get_word_vector()` (sequence of vectors) vs `get_sentence_vector()`
  - hyperparameter tests: epochs, dimension, ngrams, etc.

- `fasttext` intrinstic evaluation: `get_nearest_neighbors()` | `get_analogies()`  
  see: <https://fasttext.cc/docs/en/unsupervised-tutorial.html#nearest-neighbor-queries>

### Dataset improvements

- *Speakeasy reports*:
  - ~~with Quo.vadis dataset~~
  - with SOREL (?)
- *Windows logs*:
  - using public repos with Windows telemetry:  
    - <https://github.com/OTRF/Security-Datasets>
    - <https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES>
    - <https://github.com/NextronSystems/evtx-baseline>
  - `attackbot` data on windows?
- *Linux logs*:
  - using logs from pentest machines?
  - simulate your own TTPs in validation environments?
    - automate it?
