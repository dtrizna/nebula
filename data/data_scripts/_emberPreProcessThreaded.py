import json
import os
import py7zr
import logging
import numpy as np
from tqdm import tqdm
import threading
import time

import sys
sys.path.extend([".", "../.."])
from nebula.misc import getRealPath
from nebula.ember import PEFeatureExtractor

SCRIPT_PATH = getRealPath(type="script")
#SCRIPT_PATH = getRealPath(type="notebook")

def getHashDirectory(h):
    emulationTrainsetPath = rf"{SCRIPT_PATH}\..\data_raw\windows_emulation_trainset"
    for root, dirs, files in os.walk(emulationTrainsetPath):
        if h+".json" in files or h+".err" in files:
            return root.split("\\")[-1].split("_")[1]
    return None

def getArhiveObjects(arhivePath):
    archives = {}
    for files in os.listdir(arhivePath):
        if files.endswith(".7z"):
            archives[files.rstrip(".7z")] = py7zr.SevenZipFile(
                os.path.join(arhivePath, files), 
                password='infected',
                mode='r'
            )
    return archives

def writeEmberFeatureVector(pe, extractor, outPath, idx):
    try:
        vector = extractor.feature_vector(pe)
        #logging.warning(f"[+] Ember feature vector extracted for {idx}")
    except Exception as e:
        logging.error(f"[-] Error while extracting ember features: {e}")
        vector = np.zeros(shape=(2381,), dtype=np.float32)
    
    np.save(f"{outPath}\\{idx}.npy", vector)

if __name__ == "__main__":
    yHashesFiles = rf"{SCRIPT_PATH}\..\data_filtered\speakeasy_trainset\speakeasy_yHashes.json"
    yHahses = json.load(open(yHashesFiles, "r"))

    archivePath = rf"{SCRIPT_PATH}\..\data_raw\windows_raw_pe_trainset"
    archives = getArhiveObjects(archivePath)
    outPath = rf"{SCRIPT_PATH}\..\data_filtered\ember"
    os.makedirs(outPath, exist_ok=True)
    
    # check if some array already exists
    if [x for x in os.listdir(outPath) if x.endswith(".npy") and x.startswith("x_train_")]:
        existingArrName = os.listdir(outPath)[0]
        existingIdx = int(existingArrName.split("_")[-1].split(".")[0])
    else:
        existingIdx = 0
    print(f"Existing array: {existingArrName} with processed samples index {existingIdx}")
    
    extractor = PEFeatureExtractor(print_feature_warning=False)
    #arr = np.empty(shape=(len(yHahses), 2381), dtype=np.float32)
    for i, sampleHash in enumerate(tqdm(yHahses)):
        # skip if done
        if i <= existingIdx:
            continue
        
        directory = getHashDirectory(sampleHash)
        if directory:
            sample = f"PeX86Exe/{directory}/{sampleHash}"
        
            z = archives[directory]
            # check if sample is in archive
            if sample in z.getnames():
                peDict = z.read(targets=[sample])
                pe = peDict[sample].read()
            else:
                logging.error(f"[-] {sampleHash} should be in archive {directory}, but not found")
        else: # sample not found in any archive, should be syswow
            sample = os.path.join("c:\\", "windows", "syswow64", sampleHash + ".exe")
            if os.path.exists(sample):
                with open(sample, "rb") as f:
                    pe = f.read()
            else:
                logging.error(f"[-] {sampleHash} should be in syswow64, but not found")
        
        while threading.active_count() > 100:
            time.sleep(0.1)
        t = threading.Thread(target=writeEmberFeatureVector, args=(pe, extractor, outPath, i))
        t.start()