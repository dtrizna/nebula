import json
import os
import py7zr
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

import sys
sys.path.extend([".", "../.."])
from nebula.misc import getRealPath
from nebula.ember import PEFeatureExtractor
from nebula.constants import STATIC_FEATURE_VECTOR_LENGTH

SCRIPT_PATH = getRealPath(type="script")
#SCRIPT_PATH = getRealPath(type="notebook")

def getHashDirectory(h, dataset):
    emulationSetPath = rf"{SCRIPT_PATH}\..\data_raw\windows_emulation_{dataset}"
    for root, dirs, files in os.walk(emulationSetPath):
        if h+".json" in files or h+".err" in files:
            return root.split("\\")[-1].split("_")[1]

if __name__ == "__main__":
    dataset = "trainset"
    yHashesFiles = rf"{SCRIPT_PATH}\..\data_filtered\speakeasy_{dataset}\speakeasy_yHashes.json"
    archivePath = rf"{SCRIPT_PATH}\..\data_raw\windows_raw_pe_{dataset}"
    outPath = rf"{SCRIPT_PATH}\..\data_filtered\ember"
    os.makedirs(outPath, exist_ok=True)
    extractor = PEFeatureExtractor(print_feature_warning=False)
    yHashes = json.load(open(yHashesFiles, "r"))
    
    # check if some array already exists
    files = [x for x in os.listdir(outPath) if (x.endswith(".npy") and x.startswith(f"x_{dataset}"))]
    assert len(files) == 1, "More than one array found in output directory"
    if files:
        existingArrName = files[0]
        existingIdx = int(existingArrName.split("_")[-1].split(".")[0])
        arr = np.load(os.path.join(outPath, existingArrName))
    else:
        existingArrName = None
        existingIdx = 0
        arr = np.empty(shape=(len(yHashes), STATIC_FEATURE_VECTOR_LENGTH), dtype=np.float32)
    assert arr.shape == (len(yHashes), STATIC_FEATURE_VECTOR_LENGTH)
    print(f"[!] Existing array: {existingArrName} with processed samples index {existingIdx}")
    
    batch = 100
    i = 0
    print("[*] Collecting samples from archives in batches of", batch)
    while i < len(yHashes):
        samples = defaultdict(list)
        for i, sampleHash in enumerate(tqdm(yHashes)):
            # skip if done
            if i <= existingIdx:
                continue
            
            directory = getHashDirectory(sampleHash, dataset=dataset)
            archive = directory + ".7z"
            sample = f"PeX86Exe/{directory}/{sampleHash}"
            samples[archive].append(sample)

            # batch collected
            if i == existingIdx + batch:
                existingIdx = i
                break

        for archive in samples:
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f" {now}: [*] Extracting {len(samples[archive])} samples from {archive}...")
            with py7zr.SevenZipFile(os.path.join(archivePath, archive), mode='r', password='infected') as z:
                # check if sample is in archive
                peDict = z.read(targets=samples[archive])
                peBytes = [peDict[sample].read() for sample in peDict]
                # make peBytes a dict with same keys as peDict
                peBytes = dict(zip(peDict.keys(), peBytes))
            
            # add timestamp to log
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f" {now}: [*] Extracting Ember features for {batch} samples from {archive}...")
            for pe in tqdm(peBytes):
                vector = extractor.feature_vector(peBytes[pe])
                peHash = pe.split("/")[-1]
                arr[yHashes.index(peHash)] = vector
        
        np.save(os.path.join(f"{outPath}", f"x_{dataset}_{i}.npy"), arr)
        del samples
        # delete file with previous index
        try:
            os.remove(os.path.join(f"{outPath}", f"x_{dataset}_{i-batch}.npy"))
        except FileNotFoundError:
            pass