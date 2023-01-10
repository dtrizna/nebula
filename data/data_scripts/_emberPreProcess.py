import json
import os
import py7zr
import logging
import numpy as np
from tqdm import tqdm

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

if __name__ == "__main__":
    yHashesFiles = rf"{SCRIPT_PATH}\..\data_filtered\speakeasy_trainset\speakeasy_yHashes.json"
    archivePath = rf"{SCRIPT_PATH}\..\data_raw\windows_raw_pe_trainset"
    outPath = rf"{SCRIPT_PATH}\..\data_filtered\ember"
    os.makedirs(outPath, exist_ok=True)
    
    # check if some array already exists
    if [x for x in os.listdir(outPath) if x.endswith(".npy")]:
        existingArrName = os.listdir(outPath)[0]
        existingIdx = int(existingArrName.split("_")[-1].split(".")[0])
    else:
        existingIdx = 0
    print(f"Existing array: {existingArrName} with processed samples index {existingIdx}")
    
    extractor = PEFeatureExtractor(print_feature_warning=False)
    yHahses = json.load(open(yHashesFiles, "r"))
    arr = np.empty(shape=(len(yHahses), 2381), dtype=np.float32)
    for i, sampleHash in enumerate(tqdm(yHahses)):
        # skip if done
        if i <= existingIdx:
            continue
        
        directory = getHashDirectory(sampleHash)
        archive = directory + ".7z"
        sample = f"PeX86Exe/{directory}/{sampleHash}"
        
        with py7zr.SevenZipFile(os.path.join(archivePath, archive), mode='r', password='infected') as z:
            # check if sample is in archive
            if sample in z.getnames():
                peDict = z.read(targets=[sample])
                pe = peDict[sample].read()        
            else:
                logging.error(f"{sampleHash} not found in archive {archive}")
                import pdb;pdb.set_trace()
        
        vector = extractor.feature_vector(pe)
        arr[yHahses.index(sampleHash)] = vector
    
        np.save(os.path.join(f"{outPath}", f"x_train_{i}.npy"), arr)
        # delete file with previous index
        if i > 0:
            os.remove(os.path.join(f"{outPath}", f"x_train_{i-1}.npy"))