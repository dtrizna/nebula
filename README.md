# Project Nebula

<!-- scaled image from web -->
<center><img src="https://cdn.eso.org/images/screen/eso1205ec.jpg" width="400"></center>

## Modelling

- `modeling/fastText`: code for self-supervised embeddings (`fastText`) from system logs
- `modeling/traditionNLP`: code for traditional NLP modeling of bash process titles
- `modeling/tokenizer`: code for tokenizing bash process titles with sentencePiece
- `modeling/mlm`: code for masked language modelling CNN-Transformer model

## To Be Done

### `SpeakEasy` dynamic analysis modelling

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
  - Reformer
  - 1D-CNN-Transformer (?)
  - star transformer (?)
- cross-validate models to find best hyperparameters
  - ~~1D CNN + Linear~~
  - ~~Transformer~~
  - 1D CNN + LSTM
  > NOTE: Plot vocabSize and maxLen CV as heatmap!
- cross-validate preprocessing to find best data structure
  - clean dataset (w/o api_seq_len==1 of ordinal APIs)
    > NOTE: `MSVBVM60.ordinal_100` is in 15192 files, including 874 benignware files.
    > NOTE2: There are lots of another examples with 1 or 2 API calls only -- especially in clean folder:

    ```text
       pc                  api_name         args ret_val

    0  0x407c96  KERNEL32.GetCommandLineA   []  0x45f0

    0  0x41808b  OLEAUT32.GetModuleHandleA   [0x418190, 0x400000, 0x0, 0x0]     0x0
    1  0x41809f    OLEAUT32.GetProcAddress  [0x0, 0x418181, 0x0, 0x1212000]     0x0

    0  0x40119a         kernel32.ShowConsoleCursor  [0x4037d3, 0x40153e, 0x0, 0x0]     0x0
    1  0x40128f  kernel32.GetUserDefaultUILanguage                              []  0xffff}
    ```

  - more verbose dataset -- include additional fields
  - 3D dataset: (batch_size, seq_len, features) where each sequence represent:
    - API call tokens: name, ret_val, args
    - network request tokens
    - featurized registry modification
    - featurized file modification
    > NOTE: models should be adjusted to parse 3D structure, if transformer, might need positional encoding twice
  - consider in-memory data processing
    > NOTE: need to rerun SpeakEasy on raw PEs to get this data
- try pre-training models with SSL: (1) MLM (2) GPT-style LM

### `auditd` **commandLine** work - dataset and dowstream tasks

- reverse shell dataset:
  - generate augmented reverse shells as malicious samples: (a) train -- one set of reverse shell binaries (b) val -- another binaries
  - random split of unique legitimate commands to train/val
    > NOTE: Be sure that both generated commands resemble as close as possible to auditd telemetry!

- `auditd` dataset:
  - prepare samples for Cartesian bootstraping
  - write classes for preprocessing
  - build train and test datasets
  - sequence processing -- based on: (a) time (b) nr. of events ?
  
### Contextual embeddings

- adjust preprocessing:
  - remove artificial separator
  - include only the minimal set of fields
  - `fasttext`: any chance to use sentencePiece tokenizer instead of default one?

- `fasttext` intrinstic evaluation: `get_nearest_neighbors()` | `get_analogies()`  
  see: <https://fasttext.cc/docs/en/unsupervised-tutorial.html#nearest-neighbor-queries>

- downstream tasks for malicious classification
  - *Speakeasy reports*: define pipeline for emulation logs from Quo.Vadis
  - *Windows logs*:
    - using public repos with Windows telemetry:  
  (a) mordor (redirects here: <https://github.com/OTRF/Security-Datasets>)  
  (b) <https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES>  
  (c) <https://github.com/NextronSystems/evtx-baseline>  
    - ask for `attackbot` data on windows?
  - *Linux logs*:
    - using logs from pentest machines?
    - simulate your own TTPs in validation environments?
      - automate it?

### A/B tests

- preprocessing for MLM training:
  - minimalistic setup -- preserves only the most important fields
  - raw JSON

- tokenization & embeddings setup:
  - wordpunct + embeddings + transformer
  - sentencePiece + embeddings + transformer
  - wordpunct + fastText + transformer
  - sentencePiece + fastText + transformer
    > Note: is it possible to define custom tokenizer for fastText?

- `fasttext` with default hyperparameters:
  - sequence of vectors (`get_word_vector()`) vs `get_sentence_vector()`
  - modeling -- embeddings:
    - with trainable from scratch, random initialization
    - pre-trained `fasttext` model, frozen (not updated)
    - pre-trained `fasttext` model, but updated during supervised phase
  - hyperparameter tests: epochs, dimension, ngrams, etc.
  - fasttext `MAX_LINE_SIZE` analysis?
    - is it a problem at all? if yes -- limit size:
      - chunk smaller windows -- 5 min vs 1 min
      - chunk the same window to multiple inputs

- sequence modelling:
  - LSTM
  - 1D-CNN + LSTM
  - tranformer
  - 1D-CNN + transformer
  - reformer
      > NOTE!  
      > (a) use the same time for training, e.g. one day  
      > (b) plot mean epoch time with boxplots

## Done

- test speeds (DONE in `_auditd_b_preprocess.ipynb`)
  - ~~any ways to optimize pandas involvement?~~
    - ~~replace pandas whatsoever -- do plain `json` or `koalas`?~~
    - ~~what takes the most time -- loading? groupby?~~
- write pipeline for fastText training
  - check tokenization
    - ~~Will it be a problem that title after tokenization will screw number of tokens in single event?~~ YES
    - ~~Try with different column separator, e.g. `\s,` pr `\s,\s` (if `,` makes incorrect tokens)?~~  
      See `utils.contants.FIELD_SEPARATOR`, settled on `" , "`
  intrinstic evalution:
  - network events tokens are close by?
  - ~~anomaly detection on red team host~~ see `_auditd_c_fasttext.ipynb`
- ~~consider unification of pre-processing to represent same/similar data as Sysmon/Speakeasy/(what Win telemetry uses Vanquish)?~~  
  Sort of done -- removed all non-essential stuff, left only fields, that can be replicated in case of Sysmon. See `utils.contants.AUDITD_FIELDS`
- ~~check what telemetry we have on Windows machines~~  
  Standard structure, but unusual EventIDs...
