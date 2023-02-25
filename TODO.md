# Future Work

## Next ToDo list

- Transformer engineering:
  - Sliding window model, use: `work\speakeasy_5_Preprocess_Improve\sliding_window.py`
  - ~~Normalization -- pre or post?~~
    - RMSNorm?
    > NOTE: In <https://arxiv.org/pdf/2212.14034.pdf> they've seen no benefits from replacing LayerNorm with RMSNorm.
  - Optimization:
    - ~~AdamW~~
    - ~~weight decay 0.1~~
    - ~~gradient clipping 1.0~~
    - Learning rate schedules:
      - Take into account training budget (micro-batches / epochs / time) and adjust accordingly
      - ~~Cosine (used in "LLaMA: Open and Efficient Foundation Language Models" paper)~~
      - ~~Triangular <https://arxiv.org/pdf/2212.14034.pdf> -- code with time budget as follows: <https://github.com/JonasGeiping/cramming/blob/50bd06a65a4cd4a3dd6ee9ecce1809e1a9085374/cramming/backend/optimizers/schedulers.py#L323>~~
  - micro-batch summarization for gradient updates
  - Non-linearity:
    - GELU --> in <https://arxiv.org/pdf/2212.14034.pdf> they've seen no benefits from GELU over ReLU.
    - SwiGLU?
  - Embeddings:
    - Rotary --> in <https://arxiv.org/pdf/2212.14034.pdf> they've seen small benefits from rotary embeddings, but are slower to train.
    - scaled sinusoidal?

- Plots:
  - Lineplot loss vs. nr. of tokens (unique?)
  - Performance vs vocab size

## Short ToDo list

0. Paper:
   - Try BETH dataset
   - Try adversarial setup -- some attacks?

1. Build malware classifier class with convenient API
   - ~~Add ember feature extraction to sequential emulation model.~~
   - ~~Provide it as `pip` package.~~ Can be installed as: `python -m pip install git+https://github.com/dtrizna/nebula`
     - ~~Add vocab and speakeasy config files to package.~~
     - ~~Optimize JSON parsing.~~ Done in `JSONParser` class.
   - Add pre-trained models to package (?)
     - Organize a dedicated server suitable for: (a) storing malware dataset; (b) GPU training
     - Download raw PE data, build dataset, and train classifier.
2. Try attention models:
   - ~~input chunking to windows~~:
     - ~~flat architecture~~
   - ~~Transformer: re-run tests after fix of positional embeddings~~
   - ~~Reformer~~ Implemented `nebula.models.reformer.ReformerLM` class. Takes ~80min per epoch with 50k samples on GPU if sample length 2048.  
     - ~~Try lower learning rate -- training seems to hit plateau and fluctuate strongly after ~500 batches.~~ Loss still doesn't fall below ~0.13-0.2, even with scheduling of LR to $n\times10^{-7}$ scale.
     - ~~Try Reformer with comparable param sizes~~ Previous runs had them in similar ranges.
   - Star transformer (?)
   - Longformer (?)
3. Pre-training routines on unlabeled data according to CEF-SSL framework:
   - naive embedding methods:
     - CBOW embedding langauge modeling
     - fastText
   - language modeling:
     - ~~MLM~~ (see `nebula.pretraining.MaskedLanguageModel` class)
       - ~~Mask Every Epoch -- try variable N masking epochs;~~
       ~~> NOTE: Bug with `mask_every_epoch=True` and learning rate scheduler?~~
     - **GPT like $p(x_{t+1}|x_{t}\ ...\ x_{0})$**  
     - denoising autoencoder (see BART, etc.)
   - evaluate:
     - ~~pre-trained vs non-pretrained: TPR, F1, LOSS~~
     - ~~pre-training epochs~~ Done: see `evaluation/MaskedLanguageModeling/pretrain_epochs_analysis*` folders.
     - ~~unlabeled/labeled data ratio~~ Done: see `evaluation/MaskedLanguageModeling/uSize*` folders.
4. Data prep:
   - ~~try 50k vocab size for all promising models~~ DONE: see results under `evaluation/crossvalidation/_modelSelection_50k` folder.
   - 3D data : `(batch_size, seq_len, features)` -- first learn representation for event (`features`), then for sequence of events (`seq_len`)
5. ~~Evaluate preprocessing with BPE tokenizer instead of JSON cleanup + whitespace~~  
   DONE: See `nebula.preprocess.JSONTokenizerBPE` and tests under `evaluation/crossvalidation/_modelSelection_BPE`
6. **`auditd`/`windows` labeled dataset collection/preparation**
7. Other directions to consider:
   - Multiclass performance (?)
8. Else:
    - ~~Experiments with same downstream task dataset size, and variable pre-training size.~~
    - ~~Cross-Validation bug with low FP~~
    - ~~Visualize attention weights.~~
    - ~~Compare different Transformer setups (regular, windowed, reformer) with each other and with NeurLux / Ember.~~ Ember skipped -- not relevant?.
      - ~~Run windowed CV over 3 folds~~
      - ~~Create config for NeurLux -- (a) model (b) data~~
      - ~~Finalize ember data preprocessing~~
      - ~~complete TODOs in `_speakeasy_cv_final.py`~~

## Detailed ToDo list

### `SpeakEasy` dynamic analysis modelling

- ~~assess preprocessing: filtering and normalization~~
- ~~generate filtered dataset suitable for SSL pre-training~~
- ~~add clean samples from windows syswow64 folder~~
- ~~perform tokenization and encoding~~
- ~~prepare extractor and tokenizer classes~~
- ~~preprocess train and test sets~~
- try different models with labeled dataset:
  - ~~1D-CNN + Linear~~
    - try different padding configuration
  - ~~Transformer~~
  - ~~LSTM~~
  - ~~1D-CNN + LSTM~~
    - try 3D structure, with 1D-CNN embeddings
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
    - ~~extract separate `JSONParser` class from `PEDynamicExtractor` class that will be suitable for `auditd` dataset~~
    - ~~input for `JSONParser` should be list of normalized fields, e.g.: `[api.name, api.args, ...]`. It should be able to find where JSON has list of values, pass it `json_normalize` and then filter all consequent fields~~
  - 3D dataset: `(batch_size, seq_len, features)` where each sequence represent:
    - API call tokens: name, ret_val, args
    - network request tokens
    - featurized registry modification
    - featurized file modification

    > NOTE: models should be adjusted to parse 3D structure. If transformer, might need positional encoding twice.

    > NOTE2! This will build ground for future work on loging data (auditd, sysmon).

  - consider in-memory data processing
    > NOTE: Need to rerun SpeakEasy on raw PEs to get this data!

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
