import os
import time
import logging
import orjson
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])
from nebula.misc import getRealPath
from nebula import PEDynamicFeatureExtractor, JSONTokenizer

# SCRIPT CONFIG

OUTFOLDER_SUFFIX = "_50k"
LOGFILE = f"PreProcessing{OUTFOLDER_SUFFIX}_{int(time.time())}.log"
VOCAB_SIZES = [50000]
MAX_SEQ_LENGTHS = [2048]

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

# exclude all speakeasy JSON keys from tokenized sequence
SPEAKEASY_TOKEN_STOPWORDS = ['api_name', 'args', 'ret_val', 'event', 'path', 'open_flags', 'access_flags', 'size', 'server', 'proto', 'port', 'method']

# DATA CONFIG
SCRIPT_PATH = getRealPath(type="script")
EMULATION_TRAINSET_PATH = SCRIPT_PATH + r"\data\data_raw\windows_emulation_trainset"
EMULATION_TESTSET_PATH = SCRIPT_PATH + r"\data\data_raw\windows_emulation_testset"

BENIGN_FOLDERS = ["report_clean", "report_windows_syswow64"]

# =========================

def main(limit=None, mode="readAndFilter"):

    extractor = PEDynamicFeatureExtractor(
        speakeasyRecordFields=SPEAKEASY_RECORD_FIELDS,
        recordLimits=SPEAKEASY_RECORD_LIMITS
    )

    tokenizer = JSONTokenizer(
        patternCleanup=JSON_CLEANUP_SYMBOLS,
        stopwords=SPEAKEASY_TOKEN_STOPWORDS
    )

    # === PARSING PE FILES ===

    # for PE path or PE bytes you can use
    #jsonEventRecords = extractor.emulate(path=path)
    #jsonEventRecords = extractor.emulate(data=bytez)

    # for JSON report use ('entry_points' only)
    parserFunction = extractor.filter_and_normalize_report
    # jsonEventRecords = parserFunction(entryPoints)

    # === TOKENIZATION FUNCTIONS ===

    tokenizerFunction = tokenizer.tokenize
    encodingFunction = tokenizer.convertTokenListToIds

    # ==== READING RAW SPEAKEASY REPORTS -- FILTERING AND NORMALIZING ====
    # FORMAT: [CLEAN_NORMALIZED_JSON_REPORT_1, CLEAN_NORMALIZED_JSON_REPORT_2, ...]

    logging.warning("Initialized ...")

    trainOutFolder = SCRIPT_PATH + rf"\data\data_filtered\speakeasy_trainset{OUTFOLDER_SUFFIX}"
    os.makedirs(trainOutFolder, exist_ok=True)
    testOutFolder = SCRIPT_PATH + rf"\data\data_filtered\speakeasy_testset{OUTFOLDER_SUFFIX}"
    os.makedirs(testOutFolder, exist_ok=True)
    
    if mode == "readAndFilter":
        subFoldersTrain = [os.path.join(EMULATION_TRAINSET_PATH, x) for x in os.listdir(EMULATION_TRAINSET_PATH) if x.startswith("report_")]
        eventsTrain, yTrain, yHashesTrain = readAndFilterFolders(
            subFoldersTrain, 
            parserFunction, 
            limit=limit)

        subFoldersTest = [os.path.join(EMULATION_TESTSET_PATH, x) for x in os.listdir(EMULATION_TESTSET_PATH) if x.startswith("report_")]
        eventsTest, yTest, yHashesTest = readAndFilterFolders(
            subFoldersTest,
            parserFunction,
            limit=limit)
    
        # ==== TOKENIZING REPORTS =====
        # FORMAT: [[EVENT1_TOKEN1, EVENT1_TOKEN2, ...], [EVENT2_TOKEN1, EVENT2_TOKEN2, ...], ...]

        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.warning(f"{timenow}: Tokenizing...")
        eventsTokenizedTrain = tokenizerFunction(eventsTrain)
        eventsTokenizedTest = tokenizerFunction(eventsTest)

        # dump tokenized datasets
        with open(f"{trainOutFolder}\\speakeasy_tokenized.json", "w") as f:
            json.dump(eventsTokenizedTrain, f, indent=4)
        with open(f"{testOutFolder}\\speakeasy_tokenized.json", "w") as f:
            json.dump(eventsTokenizedTest, f, indent=4)
    
        # dump yHashes
        with open(f"{trainOutFolder}\\speakeasy_yHashes.json", "w") as f:
            json.dump(yHashesTrain, f, indent=4)
        with open(f"{testOutFolder}\\speakeasy_yHashes.json", "w") as f:
            json.dump(yHashesTest, f, indent=4)

        # dump y
        np.save(f"{trainOutFolder}\\speakeasy_y.npy", np.array(yTrain, dtype=np.int8))
        np.save(f"{testOutFolder}\\speakeasy_y.npy", np.array(yTest, dtype=np.int8))

        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.warning(f"{timenow}: Dumped tokenized files to {trainOutFolder} and {testOutFolder}")

    elif mode == "load":
        with open(f"{trainOutFolder}\\speakeasy_tokenized.json", "r") as f:
            eventsTokenizedTrain = json.load(f)
        with open(f"{testOutFolder}\\speakeasy_tokenized.json", "r") as f:
            eventsTokenizedTest = json.load(f)
        with open(f"{trainOutFolder}\\speakeasy_yHashes.json", "r") as f:
            yHashesTrain = json.load(f)
        with open(f"{testOutFolder}\\speakeasy_yHashes.json", "r") as f:
            yHashesTest = json.load(f)
        yTrain = np.load(f"{trainOutFolder}\\speakeasy_y.npy", allow_pickle=True).tolist()
        yTest = np.load(f"{testOutFolder}\\speakeasy_y.npy", allow_pickle=True).tolist()

    else:
        raise ValueError("Invalid mode")

    # ======= TRAINING TOKENIZER WITH DIFFERENT VOCAB SIZES =======
    # ======= ENCODING REPORTS WITH DIFFERENT MAX SEQ LENGTHS =====

    for vocabSize in VOCAB_SIZES:
        for maxSeqLen in MAX_SEQ_LENGTHS: 

            filePrefix = f"speakeasy_VocabSize_{vocabSize}_maxLen_{maxSeqLen}"
            if os.path.exists(f"{trainOutFolder}\\{filePrefix}_x.npy") and \
                os.path.exists(f"{testOutFolder}\\{filePrefix}_x.npy"):
                logging.warning(f" [!] Skipping {filePrefix} because files already exist")
                continue

            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Starting Preprocessing: VocabSize: {vocabSize}, MaxSeqLen: {maxSeqLen}")

            # training tokenizer -- building vocabulary on train set
            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Building vocab with vocab size: {vocabSize}...")
            tokenizer.train(eventsTokenizedTrain, vocabSize=vocabSize)
            tokenizer.dumpTokenizerFiles(trainOutFolder, tokenListSequence=eventsTokenizedTrain)

            # encoding
            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Encoding...")
            eventsEncodedTrain = encodingFunction(eventsTokenizedTrain)
            eventsEncodedTest = encodingFunction(eventsTokenizedTest)
            
            # padding
            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Padding...")
            tokenizer.sequenceLength = maxSeqLen
            eventsEncodedPaddedTrain = tokenizer.padSequenceList(eventsEncodedTrain)
            eventsEncodedPaddedTest = tokenizer.padSequenceList(eventsEncodedTest)
            
            # saving processed arrays
            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Saving files with prefix: {filePrefix}")
            np.save(f"{trainOutFolder}\\{filePrefix}_x.npy", eventsEncodedPaddedTrain)
            np.save(f"{testOutFolder}\\{filePrefix}_x.npy", eventsEncodedPaddedTest)


def readAndFilterFolders(subFolders, parserFunction, limit=None):
    """
    Reads and filters all json files in subFolders. Returns a list of events and a list of y values.
    """
    events = []
    y = []
    yHashes = []
    for subFolder in subFolders:
        timenowStart = time.time()
        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.warning(f"{timenow}: Filtering and normalizing {subFolder.strip()}...")

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
        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        timenowEnd = time.time()
        logging.warning(f"{timenow}: Finished... Took: {timenowEnd - timenowStart:.2f}s")
    return events, y, yHashes

if __name__ == "__main__":

    outputRootFolder = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_parsing_logs"
    outputFolder = os.path.join(outputRootFolder)
    os.makedirs(outputFolder, exist_ok=True)

    if LOGFILE:
        logging.basicConfig(filename=os.path.join(outputFolder, LOGFILE), level=logging.WARNING)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ===============

    #main(limit=None, mode="load")
    main()
