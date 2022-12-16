import os
import time
import pickle
import orjson
import numpy as np

import sys
sys.path.extend([".", ".."])
from nebula.misc import getRealPath
from nebula import PEDynamicFeatureExtractor, JSONTokenizer
from nebula.constants import *

# PREPROCESSING CONFIG AS DEFINED IN nebula.constants

SPEAKEASY_CONFIG = r"C:\Users\dtrizna\Code\nebula\emulation\_speakeasyConfig.json"

SPEAKEASY_RECORDS = ["apis", "registry_access", "file_access", 'network_events.traffic']

SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256}

SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC = {'apis': ['api_name', 'ret_val'],
                                    'file_access': ['event', 'path'],
                                    'network_events.traffic': ['server', 'port']}

RETURN_VALUES_TOKEEP = ['0x1', '0x0', '0xfeee0001', '0x46f0', '0x77000000', '0x4690', '0x90', '0x100', '0xfeee0004', '0x6', '0x10c', '-0x1', '0xfeee0002', '0xfeee0000', '0x54', '0x3', '0x10', '0xfeee0005', '0x2', '0xfeee0003', '0x7d90', '0xfeee0006', '0x4610', '0x45f0', '0x20', '0xffffffff', '0x4e4', '0x8810', '0x7e70', '0x7', '0x7000', '0xc000', '0xfeee0007', '0xcd', '0xf002', '0xf001', '0xf003', '0xfeee0008', '0xfeee0009', '0xfeee000b', '0xfeee000a', '0xfeee000c', '0xfeee0014', '0x47b0', '0xfeee000e', '0xfeee000d', '0xfeee000f', '0xfeee0015', '0xfeee0016', '0xfeee0010', '0xfeee0011', '0xfeee0013', '0xfeee0012', '0x4', '0xfeee0017', '0xfeee0018', '0xfeee0019', '0x8000', '0x7ec0', '0x400000', '0x1db10106', '0xfeee001a', '0xfeee001c', '0xfeee001b', '0x102', '0x5', '0xfeee0071', '0x8', '0x5265c14', '0x9000', '0x7de0', '0xc', '0x14', '0xfeee001d', '0x46d0', '0xfeee001e', '0xfeee001f', '0xfeee0020', '0x50000', '0xe', '0x8cc0', '0x4012ac', '0x12', '0xfeee0040', '0xfeee0022', '0xfeee0021', '0xfeee0023', '0xfeee0024', '0xfeee0025', '0x77d10000', '0xfeee0027', '0x2a', '0xfeee0026', '0x2c', '0xfeee007e', '0xfeee005d', '0xfeee0028', '0x78000000', '0x2e', '0xfeee007c']

JSON_CLEANUP_SYMBOLS = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"]

SPEAKEASY_TOKEN_STOPWORDS = flattenList([SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC[x] for x in SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC])

VOCAB_SIZE = 10000
MAX_SEQ_LEN = 2048

# =========================

def plotCounterCountsLineplot(counter, outfile):
    import matplotlib.pyplot as plt
    import numpy as np
    counts = [x[1] for x in counter.most_common()]
    
    plt.figure(figsize=(12,6))
    plt.plot(np.arange(len(counts)), counts)
    plt.yscale("log")
    # add ticks and grid to plot
    plt.grid(which="both")
    # save to file
    plt.savefig(outfile)

def dumpTokenizerFiles(tokenizer, outFolder):
    file = f"{outFolder}\\speakeasy_VocabSize_{VOCAB_SIZE}.pkl"
    tokenizer.dumpVocab(file)
    print("Dumped vocab to {}".format(file))
    
    file = f"{outFolder}\\speakeasy_VocabSize_{VOCAB_SIZE}_counter.pkl"
    print("Dumped vocab counter to {}".format(file))
    with open(file, "wb") as f:
        pickle.dump(tokenizer.counter, f)

    file = f"{outFolder}\\speakeasy_VocabSize_{VOCAB_SIZE}_counter_plot.png"
    plotCounterCountsLineplot(tokenizer.counter, file)
    print("Dumped vocab counter plot to {}".format(file))

def parseDatasetFolders(subFolders, outFolder, limit=None):
    events = []
    y = []
    for subFolder in subFolders:

        timenowStart = time.time()
        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(f"{timenow}: Filtering and normalizing {subFolder}...")

        files = [f"{subFolder}\\{x}" for x in os.listdir(subFolder)[:limit] if x.endswith(".json")]
        l = len(files)
        for i,file in enumerate(files):
            print(f"{subFolder:>20}: {i+1}/{l} {' '*30}", end="\r")
            
            with open(file, "r") as f:
                entryPoints = orjson.loads(f.read())

            # for PE path or PE bytes you can use
            #jsonEventRecords = extractor.emulate(path=path)
            #jsonEventRecords = extractor.emulate(data=bytez)

            # for 'entry_points' from json report use
            jsonEventRecords = extractor.parseReportEntryPoints(entryPoints)
            events.append(jsonEventRecords)

            if subFolder in BENIGN_FOLDERS:
                y.append(0)
            else:
                y.append(1)
            
        timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        timenowEnd = time.time()
        print(f"{timenow}: Finished... Took: {timenowEnd - timenowStart:.2f}s", " "*100)
    
    timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f"{timenow}: Tokenizing...")
    eventsTokenized = tokenizer.tokenize(events)

    timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f"{timenow}: Building vocab...")
    tokenizer.buildVocab(eventsTokenized, vocabSize=VOCAB_SIZE)

    dumpTokenizerFiles(tokenizer, outFolder)

    timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f"{timenow}: Encoding...")
    eventsEncoded = tokenizer.convertTokenListToIds(eventsTokenized)

    timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f"{timenow}: Padding...")
    eventsEncodedPadded = tokenizer.padSequenceList(eventsEncoded, maxLen=MAX_SEQ_LEN)
    y = np.array(y, dtype=np.int8)
    print(eventsEncodedPadded.shape)
    print(y.shape)
    np.save(f"{outFolder}\\speakeasy_VocabSize_{VOCAB_SIZE}_maxLen_{MAX_SEQ_LEN}.npy", eventsEncodedPadded)
    np.save(f"{outFolder}\\speakeasy_VocabSize_{VOCAB_SIZE}_maxLen_{MAX_SEQ_LEN}_y.npy", y)


if __name__ == "__main__":

    extractor = PEDynamicFeatureExtractor(
        speakeasyConfig=SPEAKEASY_CONFIG,
        speakeasyRecords=SPEAKEASY_RECORDS,
        recordSubFilter=SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC,
        recordLimits=SPEAKEASY_RECORD_LIMITS,
        returnValues=RETURN_VALUES_TOKEEP
    )

    tokenizer = JSONTokenizer(
        patternCleanup=JSON_CLEANUP_SYMBOLS,
        stopwords=SPEAKEASY_TOKEN_STOPWORDS
    )

    PATH = getRealPath(type="script")
    BENIGN_FOLDERS = ["report_clean", "report_windows_syswow64"]
    print("Initialized ...")

    limit = None 

    EMULATION_TRAINSET_PATH = PATH + r"\..\data\data_raw\windows_emulation_trainset"
    subFoldersTrain = [os.path.join(EMULATION_TRAINSET_PATH, x) for x in os.listdir(EMULATION_TRAINSET_PATH) if x.startswith("report_")]
    trainOutFolder = PATH + r"\..\data\data_filtered\speakeasy_trainset"
    os.makedirs(trainOutFolder, exist_ok=True)
    parseDatasetFolders(subFoldersTrain, trainOutFolder, limit=limit)

    EMULATION_TESTSET_PATH = PATH + r"\..\data\data_raw\windows_emulation_testset"
    subFoldersTest = [os.path.join(EMULATION_TESTSET_PATH, x) for x in os.listdir(EMULATION_TESTSET_PATH) if x.startswith("report_")]
    testOutFolder = PATH + r"\..\data\data_filtered\speakeasy_testset"
    os.makedirs(testOutFolder, exist_ok=True)
    parseDatasetFolders(subFoldersTest, testOutFolder, limit=limit)
