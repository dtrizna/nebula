import os
import time
import logging
import orjson
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])
from nebula.misc import getRealPath, flattenList
from nebula.misc import dumpTokenizerFiles
from nebula import PEDynamicFeatureExtractor, JSONTokenizer

OUTFOLDER_SUFFIX = "_TEST" # ""

# PREPROCESSING CONFIG AS DEFINED IN nebula.constants
# from nebula.constants import *

SPEAKEASY_CONFIG = r"C:\Users\dtrizna\Code\nebula\emulation\_speakeasyConfig.json"

SPEAKEASY_RECORDS = ["registry_access", "file_access", "network_events.traffic", "apis"]

SPEAKEASY_RECORD_SUBFILTER = {
                                'apis': ['api_name', 'ret_val'],
                                'file_access': ['event', 'path'],
                                'network_events.traffic': ['server', 'port']
                            }

SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256}

RETURN_VALUES_TOKEEP = ['0x1', '0x0', '0xfeee0001', '0x46f0', '0x77000000', '0x4690', '0x90', '0x100', '0xfeee0004', '0x6', '0x10c', '-0x1', '0xfeee0002', '0xfeee0000', '0x54', '0x3', '0x10', '0xfeee0005', '0x2', '0xfeee0003', '0x7d90', '0xfeee0006', '0x4610', '0x45f0', '0x20', '0xffffffff', '0x4e4', '0x8810', '0x7e70', '0x7', '0x7000', '0xc000', '0xfeee0007', '0xcd', '0xf002', '0xf001', '0xf003', '0xfeee0008', '0xfeee0009', '0xfeee000b', '0xfeee000a', '0xfeee000c', '0xfeee0014', '0x47b0', '0xfeee000e', '0xfeee000d', '0xfeee000f', '0xfeee0015', '0xfeee0016', '0xfeee0010', '0xfeee0011', '0xfeee0013', '0xfeee0012', '0x4', '0xfeee0017', '0xfeee0018', '0xfeee0019', '0x8000', '0x7ec0', '0x400000', '0x1db10106', '0xfeee001a', '0xfeee001c', '0xfeee001b', '0x102', '0x5', '0xfeee0071', '0x8', '0x5265c14', '0x9000', '0x7de0', '0xc', '0x14', '0xfeee001d', '0x46d0', '0xfeee001e', '0xfeee001f', '0xfeee0020', '0x50000', '0xe', '0x8cc0', '0x4012ac', '0x12', '0xfeee0040', '0xfeee0022', '0xfeee0021', '0xfeee0023', '0xfeee0024', '0xfeee0025', '0x77d10000', '0xfeee0027', '0x2a', '0xfeee0026', '0x2c', '0xfeee007e', '0xfeee005d', '0xfeee0028', '0x78000000', '0x2e', '0xfeee007c']

JSON_CLEANUP_SYMBOLS = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"]

SPEAKEASY_TOKEN_STOPWORDS = flattenList([SPEAKEASY_RECORD_SUBFILTER[x] for x in SPEAKEASY_RECORD_SUBFILTER])

VOCAB_SIZES = [500, 1500, 1000, 2000]
MAX_SEQ_LENGTHS = [512, 1024, 2048]

SCRIPT_PATH = getRealPath(type="script")

EMULATION_TRAINSET_PATH = SCRIPT_PATH + r"\data\data_raw\windows_emulation_trainset"
EMULATION_TESTSET_PATH = SCRIPT_PATH + r"\data\data_raw\windows_emulation_testset"

BENIGN_FOLDERS = ["report_clean", "report_windows_syswow64"]

# =========================

def main(limit=None, mode="run", y=True):

    extractor = PEDynamicFeatureExtractor(
        speakeasyConfig=SPEAKEASY_CONFIG,
        speakeasyRecords=SPEAKEASY_RECORDS,
        recordSubFilter=SPEAKEASY_RECORD_SUBFILTER,
        recordLimits=SPEAKEASY_RECORD_LIMITS,
        returnValues=RETURN_VALUES_TOKEEP
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
    parserFunction = extractor.parseReportEntryPoints
    # jsonEventRecords = parserFunction(entryPoints)

    # === TOKENIZATION FUNCTIONS ===

    tokenizerFunction = tokenizer.tokenize
    encodingFunction = tokenizer.convertTokenListToIds

    # ==== ACTUAL PROCESSING ====

    logging.warning("Initialized ...")

    trainOutFolder = SCRIPT_PATH + rf"\data\data_filtered\speakeasy_trainset{OUTFOLDER_SUFFIX}"
    os.makedirs(trainOutFolder, exist_ok=True)
    testOutFolder = SCRIPT_PATH + rf"\data\data_filtered\speakeasy_testset{OUTFOLDER_SUFFIX}"
    os.makedirs(testOutFolder, exist_ok=True)
    
    if mode == "run":
        subFoldersTrain = [os.path.join(EMULATION_TRAINSET_PATH, x) for x in os.listdir(EMULATION_TRAINSET_PATH) if x.startswith("report_")]
        eventsTrain, yTrain = readAndFilterFolders(
            subFoldersTrain, 
            parserFunction, 
            limit=limit)

        subFoldersTest = [os.path.join(EMULATION_TESTSET_PATH, x) for x in os.listdir(EMULATION_TESTSET_PATH) if x.startswith("report_")]
        eventsTest, yTest = readAndFilterFolders(
            subFoldersTest,  
            parserFunction, 
            limit=limit)

        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.warning(f"{timenow}: Tokenizing...")
        eventsTokenizedTrain = tokenizerFunction(eventsTrain)
        eventsTokenizedTest = tokenizerFunction(eventsTest)

        with open(f"{trainOutFolder}\\speakeasy_tokenized.json", "w") as f:
            json.dump(eventsTokenizedTrain, f, indent=4)
        with open(f"{testOutFolder}\\speakeasy_tokenized.json", "w") as f:
            json.dump(eventsTokenizedTest, f, indent=4)

        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.warning(f"{timenow}: Dumped tokenized files to {trainOutFolder} and {testOutFolder}")

    elif mode == "load":
        with open(f"{trainOutFolder}\\speakeasy_tokenized.json", "r") as f:
            eventsTokenizedTrain = json.load(f)
        with open(f"{testOutFolder}\\speakeasy_tokenized.json", "r") as f:
            eventsTokenizedTest = json.load(f)

    for vocabSize in VOCAB_SIZES:
        for maxSeqLen in MAX_SEQ_LENGTHS:
            # if maxSeqLen < 2048:
            #     SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256, 'apis': 256}
            # else:
            #     SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256}
                
            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Starting Preprocessing: VocabSize: {vocabSize}, MaxSeqLen: {maxSeqLen}")

            buildVocab(eventsTokenizedTrain, tokenizer, vocabSize=vocabSize, outFolder=trainOutFolder)

            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Encoding...")
            eventsEncodedTrain = encodingFunction(eventsTokenizedTrain)
            eventsEncodedTest = encodingFunction(eventsTokenizedTest)
            
            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Padding...")
            eventsEncodedPaddedTrain = tokenizer.padSequenceList(eventsEncodedTrain, maxLen=maxSeqLen)
            eventsEncodedPaddedTest = tokenizer.padSequenceList(eventsEncodedTest, maxLen=maxSeqLen)
            

            timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            logging.warning(f"{timenow}: Saving files with prefix: speakeasy_VocabSize_{vocabSize}_maxLen_{maxSeqLen}")
            np.save(f"{trainOutFolder}\\speakeasy_VocabSize_{vocabSize}_maxLen_{maxSeqLen}_x.npy", eventsEncodedPaddedTrain)
            np.save(f"{testOutFolder}\\speakeasy_VocabSize_{vocabSize}_maxLen_{maxSeqLen}_x.npy", eventsEncodedPaddedTest)
            
            if y:
                yTrain = np.array(yTrain, dtype=np.int8)
                yTest = np.array(yTest, dtype=np.int8)
                np.save(f"{trainOutFolder}\\speakeasy_VocabSize_{vocabSize}_maxLen_{maxSeqLen}_y.npy", yTrain)
                np.save(f"{testOutFolder}\\speakeasy_VocabSize_{vocabSize}_maxLen_{maxSeqLen}_y.npy", yTest)


def buildVocab(eventsTokenized, tokenizer, vocabSize, outFolder):
    
    timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    logging.warning(f"{timenow}: Building vocab with vocab size: {vocabSize}...")
    tokenizer.buildVocab(eventsTokenized, vocabSize=vocabSize)
    dumpTokenizerFiles(tokenizer, outFolder, vocabSize)


def readAndFilterFolders(subFolders, parserFunction, limit=None):
    events = []
    y = []
    for subFolder in subFolders:

        timenowStart = time.time()
        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        logging.warning(f"{timenow}: Filtering and normalizing {subFolder.strip()}...")

        files = [os.path.join(subFolder,x) for x in os.listdir(subFolder)[:limit] if x.endswith(".json")]
        for file in tqdm(files):
            with open(file, "r") as f:
                entryPoints = orjson.loads(f.read())

            jsonEventRecords = parserFunction(entryPoints)
            events.append(jsonEventRecords)

            if os.path.basename(subFolder) in BENIGN_FOLDERS:
                y.append(0)
            else:
                y.append(1)
            
        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        timenowEnd = time.time()
        logging.warning(f"{timenow}: Finished... Took: {timenowEnd - timenowStart:.2f}s")
    return events, y
    

if __name__ == "__main__":

    outputRootFolder = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_parsing_logs"
    outputFolder = os.path.join(outputRootFolder)
    os.makedirs(outputFolder, exist_ok=True)

    logFile = f"PreProcessing_{int(time.time())}.log"

    if logFile:
        logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # ===============

    #main(limit=None, mode="load", y=False)
    main()
