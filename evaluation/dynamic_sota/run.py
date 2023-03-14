import os
REPO_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

import sys
sys.path.extend([REPO_ROOT, "."])

import time
import json
import logging
import numpy as np
from collections import defaultdict

from nebula.models.neurlux import NeurLuxModel, NeurLuxPreprocessor
from nebula.models.quovadis import QuoVadisModel
from nebula.preprocessing import JSONTokenizerBPE, PEDynamicFeatureExtractor
from nebula.preprocessing.normalization import read_and_filter_json_folders
from nebula.models.quovadis import report_to_apiseq
from nebula.models import TransformerEncoderChunks


LIMIT = 10 # None

# speakeasy
VOCAB=5000 # 50000
SEQ_LEN=512
QUO_VADIS_TOP_API = 600
SPEAKEASY_TRAINSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_trainset")
SPEAKEASY_TESTSET_PATH = os.path.join(REPO_ROOT, "data", "data_raw", "windows_emulation_testset")

def preprocess_nebula_speakeasy(
        folder, 
        limit=None,
        outfolder=None,
        dump_paths=True,
        # filter settings
        record_fields=[
            'file_access.event',
            'file_access.path',
            'network_events.traffic.server',
            'network_events.traffic.port',
            'registry_access.event',
            'registry_access.path',
            'apis.api_name',
            'apis.args',
            'apis.ret_val',
        ],
        record_limits = {
            "network_events.traffic": 256
        },
        # tokenizer settings
        tokenizer_model=None,
        json_cleanup_symbols = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"],
        stopwords = ['api_name', 'args', 'ret_val', 'event', 'path', 'open_flags', 'access_flags', 'size', 'server', 'proto', 'port', 'method'],
        vocab_size = 50000,
        seq_len = 512,
):
    suffix = "test" if tokenizer_model else "train"
    if outfolder is None:
        outfolder = f"nebula_speakeasy_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    # filter and normalize reports
    extractor = PEDynamicFeatureExtractor(
        speakeasyRecordFields=record_fields,
        recordLimits=record_limits
    )
    subfolders = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith("report_")]
    events, y, y_filepaths = read_and_filter_json_folders(
        subFolders=subfolders,
        filter_function=extractor.filter_and_normalize_report,
        benign_folders=["report_clean", "report_windows_syswow64"],
        limit=limit
    )
    y = np.array(y, dtype=np.int8)
    np.save(os.path.join(outfolder, f"y_{suffix}.npy"), y)
    logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}.npy')}")
    
    if dump_paths:
        with open(os.path.join(outfolder, f"y_names_{suffix}.json"), "w") as f:
            json.dump(y_filepaths, f, indent=4)
        logging.warning(f" [!] Saved Y names as {os.path.join(outfolder, f'y_names_{suffix}.json')}")

    # train tokenizer
    if not tokenizer_model:
        tokenizer = JSONTokenizerBPE(
            patternCleanup=json_cleanup_symbols,
            stopwords=stopwords
        )
        logging.warning(" [*] Initializing tokenizer training...")
        tokenizer.train(
            events,
            vocab_size=vocab_size,
            model_prefix=os.path.join(outfolder, f"tokenizer_{vocab_size}")
        )
    else:
        tokenizer = JSONTokenizerBPE(
            model_path=tokenizer_model,
            patternCleanup=json_cleanup_symbols,
            stopwords=stopwords
        )
    
    logging.warning(" [*] Encoding and padding...")
    encoded = tokenizer.encode(events)
    x = tokenizer.pad_sequences(encoded, sequenceLength=seq_len)
    np.save(os.path.join(outfolder, f"x_{suffix}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}.npy')}")

    return x, y, y_filepaths


def preprocess_neurlux_speakeasy(
        reports_raw,
        y=None,
        outfolder=None,
        vocab_size=50000,
        seq_len=512,
        vocab_file=None
):
    suffix = "test" if vocab_file else "train"
    if outfolder is None:
        outfolder = f"neurlux_speakeasy_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    
    neurlux_preprocessor = NeurLuxPreprocessor(
        vocab_size=vocab_size,
        max_length=seq_len
    )
    if vocab_file:
        logging.warning(f" [*] Loading NeurLux preprocessor from {vocab_file}...")
        neurlux_preprocessor.load_vocab(vocab_file)
    else:
        logging.warning(f" [*] Training NeurLux preprocessor on {len(reports_raw)} reports...")
        neurlux_preprocessor.train(reports_train_raw)
        vocab_file = neurlux_preprocessor.dump_vocab(outfolder)
    
    x = neurlux_preprocessor.preprocess_sequence(reports_raw)
    x = neurlux_preprocessor.pad_sequence(x)
    np.save(os.path.join(outfolder, f"x_{suffix}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}.npy')}")
    if y is not None:
        assert len(reports_raw) == y.shape[0], "X and Y must have the same length"
        np.save(os.path.join(outfolder, f"y_{suffix}.npy"), y)
        logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}.npy')}")

    return x, vocab_file


def preprocess_quovadis_speakeasy(
    reports_raw,
    y=None,
    vocab_file=None,
    outfolder=None,
    seq_len=150,
):
    suffix = "test" if vocab_file else "train"
    if outfolder is None:
        outfolder = f"neurlux_speakeasy_{int(time.time())}"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)

    if vocab_file:
        with open(vocab_file) as f:
            vocab = json.load(f)
        quovadis_model = QuoVadisModel(
            vocab=vocab,
            seq_len=seq_len
        )
        api_sequences = []
        for report in reports_raw:
            api_sequences.append(report_to_apiseq(report))
    else:
        quovadis_model = QuoVadisModel(
            seq_len=seq_len
        )
        logging.warning(" [*] Building Quo Vadis vocabulary...")
        api_sequences = quovadis_model.build_vocab(reports_train_raw, top_api=QUO_VADIS_TOP_API)
        vocab_file = quovadis_model.dump_vocab(outfolder)
    
    x = quovadis_model.apisequences_to_arr(api_sequences)
    np.save(os.path.join(outfolder, f"x_{suffix}.npy"), x)
    logging.warning(f" [!] Saved X as {os.path.join(outfolder, f'x_{suffix}.npy')}")
    if y is not None:
        assert len(reports_raw) == y.shape[0], "X and Y must have the same length"
        np.save(os.path.join(outfolder, f"y_{suffix}.npy"), y)
        logging.warning(f" [!] Saved Y as {os.path.join(outfolder, f'y_{suffix}.npy')}")

    return x, vocab_file



if __name__ == "__main__":
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # READ DATA
    # IN - DIRECTORY
    # OUT - ARR

    # =========== set out logging to both file and stdout
    out = f"out_{int(time.time())}"
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out, "log.txt")),
            logging.StreamHandler()
        ]
    )
    
    datafolders = defaultdict(dict)

    # =========== 'nebula' & 'speakeasy' preprocessing
    datafolders['nebula']['speakeasy'] = os.path.join(out, f"nebula_speakeasy_vocab_{VOCAB}_seqlen_{SEQ_LEN}")
    _, y_train, y_paths_train, = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TRAINSET_PATH,
        limit=LIMIT,
        vocab_size=VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula']['speakeasy'],
    )
    _, y_test, y_paths_test = preprocess_nebula_speakeasy(
        folder=SPEAKEASY_TESTSET_PATH,
        limit=LIMIT,
        vocab_size=VOCAB,
        seq_len=SEQ_LEN,
        outfolder=datafolders['nebula']['speakeasy'],
        tokenizer_model=os.path.join(datafolders['nebula']['speakeasy'], f"tokenizer_{VOCAB}.model"),
    )

    # =========== get raw reports for further processing
    reports_train_raw = []
    for file in y_paths_train:
        with open(file) as f:
            reports_train_raw.append(json.load(f))

    reports_test_raw = []
    for file in y_paths_test:
        with open(file) as f:
            reports_test_raw.append(json.load(f))

    # =========== 'neurlux' & 'speakeasy' preprocessing
    datafolders['neurlux']['speakeasy'] = os.path.join(out, f"neurlux_speakeasy_vocab_{VOCAB}_seqlen_{SEQ_LEN}")
    
    _, neurlux_vocab_file = preprocess_neurlux_speakeasy(
        reports_train_raw, 
        y=y_train,
        outfolder=datafolders['neurlux']['speakeasy']
    )

    _ = preprocess_neurlux_speakeasy(
        reports_test_raw,
        y=y_test,
        vocab_file=neurlux_vocab_file,
        outfolder=datafolders['neurlux']['speakeasy']
    )

    # ============ 'quo vadis' & 'speakeasy' preprocessing
    datafolders['quovadis']['speakeasy'] = os.path.join(out, f"quovadis_speakeasy_vocab_{VOCAB}_seqlen_{SEQ_LEN}")
    
    _, quovadis_vocab_file = preprocess_quovadis_speakeasy(
        reports_train_raw,
        y=y_train,
        seq_len=SEQ_LEN,
        outfolder=datafolders['quovadis']['speakeasy']
    )
    _ = preprocess_quovadis_speakeasy(
        reports_test_raw,
        y=y_test,
        vocab_file=quovadis_vocab_file,
        seq_len=SEQ_LEN,
        outfolder=datafolders['quovadis']['speakeasy']
    )

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # CREATE MODEL
    # OUT - MODEL TRAINER OBJECT


    # TRAINING LOOP
    # IN - MODEL TRAINER OBJECT, DATA
    # OUT - TRAINED MODEL, CV DATA AS NPZ


    # EVALUATE RESULTS
    # IN - MODEL TRAINER OBJECT, DATA
    # OUT - ROC, F1, AUC, PRECISION, RECALL -- MEAN OVER VALIDATIONS & TEST SETS