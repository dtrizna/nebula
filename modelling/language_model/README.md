# To Be Done

- intrinstic evaluation: `get_nearest_neighbors()` | `get_analogies()`  
  see: <https://fasttext.cc/docs/en/unsupervised-tutorial.html#nearest-neighbor-queries>
- ~~check what telemetry we have on Windows machines~~
- downstream tasks for malicious classification
  - using public repos with Windows telemetry:  
  (a) mordor (redirects here: <https://github.com/OTRF/Security-Datasets>)  
  (b) <https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES>  
  (c) <https://github.com/NextronSystems/evtx-baseline>  
  - using pentest machines and auditd
  - ask for `attackbot` data on windows?
- A/B tests with default settings:
  - separator - " , " | "\t"
  - MAX_LINE_SIZE tests?
    - chunk smaller windows
    - chunk same window to multiple events
    - is it a problem that one hosts length is larger that `MAX_LINE_SIZE` in fastText
  - (TF-IDF | HashingVectorizer) vs FastText
  - sequence of vectors (get_word_vector()) vs get_sentence_vector()
  - fasttext hyperparameter tests: epochs, dimension, ngrams, etc.
  - modeling -- embeddings:
    - with trainable Embeddings from random initialization OR
    - pre-trained frozen Embeddings
    - pre-trained, but updated during supervised phase
  - modeling -- model:
    - LSTM
    - attention
    - plot mean epoch time with boxplots

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
