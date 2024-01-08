import os
import time
import logging
import orjson
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])
from nebula.misc import get_path
from nebula import PEDynamicFeatureExtractor, JSONTokenizerBPE

# SCRIPT CONFIG

LIMIT = None
VOCABS = {8192: "8k", 16384: "16k", 32768: "32k"}
MAX_SEQ_LENGTHS = [512]

# from nebula.constants import *
# PREPROCESSING CONFIG AS DEFINED IN nebula.constants

SPEAKEASY_RECORD_FIELDS = [
    'file_access.event',
    'file_access.path',
    'network_events.traffic.server',
    'network_events.traffic.port',
    'registry_access.event',
    'registry_access.path',
    'apis.api_name',
    'apis.args',
    'apis.ret_val',
]

SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256}

JSON_CLEANUP_SYMBOLS = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"]

SPEAKEASY_TOKEN_STOPWORDS = ['api_name', 'args', 'ret_val', 'event', 'path', 'open_flags', 'access_flags', 'size', 'server', 'proto', 'port', 'method']

# DATA CONFIG
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")
EMULATION_TRAINSET_PATH = os.path.join(REPO_ROOT, "..", "..", "Data", "Nebula_Kaggle", "windows_emulation_trainset")
EMULATION_TESTSET_PATH = os.path.join(REPO_ROOT,  "..", "..", "Data", "Nebula_Kaggle", "windows_emulation_testset")

BENIGN_FOLDERS = ["report_clean", "report_windows_syswow64"]

# =========================

def main(limit=None):
    logging.warning("Initialized ...")
    extractor = PEDynamicFeatureExtractor(
        speakeasyRecordFields=SPEAKEASY_RECORD_FIELDS,
        recordLimits=SPEAKEASY_RECORD_LIMITS
    )
    # === PARSING PE FILES === 
    subFoldersTrain = [os.path.join(EMULATION_TRAINSET_PATH, x) for x in os.listdir(EMULATION_TRAINSET_PATH) if x.startswith("report_")]
    eventsTrain, yTrain, yHashesTrain = readAndFilterFolders(
        subFoldersTrain, 
        extractor.filter_and_normalize_report, 
        limit=limit)

    subFoldersTest = [os.path.join(EMULATION_TESTSET_PATH, x) for x in os.listdir(EMULATION_TESTSET_PATH) if x.startswith("report_")]
    eventsTest, yTest, yHashesTest = readAndFilterFolders(
        subFoldersTest,
        extractor.filter_and_normalize_report,
        limit=limit)

    # ======= ENCODING REPORTS WITH DIFFERENT MAX SEQ LENGTHS & VOCAB SIZES =====

    for vocab_size in VOCABS.keys():
        OUTFOLDER_SUFFIX = f"_BPE_{VOCABS[vocab_size]}_lim_{limit}"
        
        TRAIN_OUT_FOLDER = os.path.join(REPO_ROOT, "data", "data_filtered", f"speakeasy_trainset{OUTFOLDER_SUFFIX}")
        os.makedirs(TRAIN_OUT_FOLDER, exist_ok=True)
        
        TEST_OUT_FOLDER = os.path.join(REPO_ROOT, "data", "data_filtered", f"speakeasy_testset{OUTFOLDER_SUFFIX}")
        os.makedirs(TEST_OUT_FOLDER, exist_ok=True)

        # HANDLE Ys
        with open(os.path.join(TRAIN_OUT_FOLDER, "speakeasy_yHashes.json"), "w") as f:
            json.dump(yHashesTrain, f, indent=4)
        with open(os.path.join(TEST_OUT_FOLDER, "speakeasy_yHashes.json"), "w") as f:
            json.dump(yHashesTest, f, indent=4)
        np.save(os.path.join(TRAIN_OUT_FOLDER, f"speakeasy_y.npy"), np.array(yTrain, dtype=np.int8))
        np.save(os.path.join(TEST_OUT_FOLDER, f"speakeasy_y.npy"), np.array(yTest, dtype=np.int8))


        # HANDLE Xs
        for maxSeqLen in MAX_SEQ_LENGTHS: 
            tokenizer = JSONTokenizerBPE(
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=SPEAKEASY_TOKEN_STOPWORDS,
                vocab_size=vocab_size,
                seq_len=maxSeqLen
            )

            filePrefix = f"speakeasy_vocab_size_{vocab_size}"
            if os.path.exists(os.path.join(TRAIN_OUT_FOLDER, f"{filePrefix}_x.npy")) and \
                os.path.exists(os.path.join(TEST_OUT_FOLDER, f"{filePrefix}_x.npy")):
                logging.warning(f" [!] Skipping {filePrefix} because files already exist")
                continue

            # training tokenizer -- building vocabulary on train set
            logging.warning(f"Training tokenizer with vocab size: {vocab_size}...")
            tokenizer.train(
                eventsTrain,
                model_prefix = os.path.join(TRAIN_OUT_FOLDER, f"{filePrefix}_tokenizer"),
                removeTrainFiles=False
            )
            # encoding
            logging.warning(f"Encoding...")
            eventsEncodedTrain = tokenizer.encode(eventsTrain, pad=True)
            eventsEncodedTest = tokenizer.encode(eventsTest, pad=True)
            
            # padding -- done in encode() above
            # logging.warning(f"Padding with maxLen={maxSeqLen}...")
            # eventsEncodedPaddedTrain = tokenizer.pad_sequences(eventsEncodedTrain)
            # eventsEncodedPaddedTest = tokenizer.pad_sequences(eventsEncodedTest)
            
            # saving processed arrays
            logging.warning(f"Saving files with prefix: {filePrefix}_maxlen_{maxSeqLen}")
            np.save(os.path.join(TRAIN_OUT_FOLDER, f"{filePrefix}_maxlen_{maxSeqLen}_x.npy"), eventsEncodedTrain)
            np.save(os.path.join(TEST_OUT_FOLDER, f"{filePrefix}_maxlen_{maxSeqLen}_x.npy"), eventsEncodedTest)


def readAndFilterFolders(subFolders, parserFunction, limit=None):
    """
    Reads and filters all json files in subFolders. Returns a list of events and a list of y values.
    """
    events = []
    y = []
    yHashes = []
    for subFolder in subFolders:
        timenowStart = time.time()
        logging.warning(f"Filtering and normalizing {subFolder.strip()}...")

        files = [os.path.join(subFolder,x) for x in os.listdir(subFolder)[:limit] if x.endswith(".json")]
        for file in tqdm(files):
            with open(file, "r") as f:
                entryPoints = orjson.loads(f.read())

            jsonEventRecord = parserFunction(entryPoints)
            if jsonEventRecord:
                events.append(jsonEventRecord)
                
                hhash = os.path.basename(file).rstrip('.json')
                yHashes.append(hhash)
                if os.path.basename(subFolder) in BENIGN_FOLDERS:
                    y.append(0)
                else:
                    y.append(1)
        timenowEnd = time.time()
        logging.warning(f"Finished... Took: {timenowEnd - timenowStart:.2f}s")
    return events, y, yHashes


if __name__ == "__main__":

    logfolder = rf"{REPO_ROOT}\data\data_filtered\speakeasy_parsing_logs"
    os.makedirs(logfolder, exist_ok=True)
    LOGFILE = os.path.join(logfolder, f"PreProcessing_2n_Loop_{int(time.time())}.log")
    print(f'[!] Logging to {LOGFILE}')
    logging.basicConfig(
        filename=LOGFILE,
        level=logging.WARNING,
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )        
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ===============
    main(limit=LIMIT)
    #main()
