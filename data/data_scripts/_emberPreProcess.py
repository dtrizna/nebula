import json
import os
import py7zr
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import subprocess

from _lzma import LZMAError
BROKEN_FILES = []

import sys
sys.path.extend([".", "../.."])
from nebula.misc import get_path
from nebula.ember import PEFeatureExtractor
from nebula.constants import STATIC_FEATURE_VECTOR_LENGTH

SCRIPT_PATH = get_path(type="script")
#SCRIPT_PATH = getRealPath(type="notebook")

def getHashDirectory(h, dataset):
    emulationSetPath = os.path.join(SCRIPT_PATH, "..", "data_raw", f"windows_emulation_{dataset}")
    for root, dirs, files in os.walk(emulationSetPath):
        if h+".json" in files or h+".err" in files:
            return root.split("\\")[-1].split("_")[1]

def extractSamplesWithPy7zr(archiveName, archive, samples):
    print(f"\n{now}: [*] Extracting {len(samples[archive])} samples from {archive} using py7zr...")
    with py7zr.SevenZipFile(archiveName, mode='r', password='infected') as z:
        peDict = z.read(targets=samples[archive])
        peBytes = [peDict[sample].read() for sample in peDict]
        # make peBytes a dict with same keys as peDict
        peBytes = dict(zip(peDict.keys(), peBytes))
    return peBytes

def extractBrokenFile(archivePath, archive, errorFile, peBytes):
    cmd = f"7z e -pinfected -y -o{archivePath} {os.path.join(archivePath, archive)} \"{errorFile}\""
    # execute command with suprocess, but hide output
    try:
        subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] {errorFile} could not be extracted even with 7z, added to broken files, and skipped")
        BROKEN_FILES.append(errorFile)
        return peBytes
    extractedPath = os.path.join(archivePath, errorFile.split("/")[-1])
    if os.path.exists(extractedPath):
        print(f"[!] {errorFile} extracted successfully")
        with open(extractedPath, "rb") as f:
            peBytes.update({errorFile: f.read()})
        print(f"[!] {errorFile} added to peBytes")
        os.remove(extractedPath)
        print(f"[!] {errorFile} deleted")
    else:
        raise e
    return peBytes

def extractSamplesWith7z(archivePath, archive, samples):
    print(f"{now}: [*] Extracting {len(samples[archive])} samples from {archive} using 7z...")
    peBytes = {}
    for sample in samples[archive]:
        peBytes = extractBrokenFile(archivePath, archive, sample, peBytes)
    return peBytes


if __name__ == "__main__":
    dataset = "trainset"
    yHashesFiles = os.path.join(SCRIPT_PATH, "..", "data_filtered", f"speakeasy_{dataset}", "speakeasy_yHashes.json")
    archivePath = os.path.join(SCRIPT_PATH, "..", "data_raw", f"windows_raw_pe_{dataset}")
    outPath = os.path.join(SCRIPT_PATH, "..", "data_filtered", "ember")
    os.makedirs(outPath, exist_ok=True)
    extractor = PEFeatureExtractor(print_feature_warning=False)
    print("[*] Loading yHashes from", yHashesFiles)
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
    
    # sanity check
    print(arr[existingIdx-50:existingIdx+3,0:3])

    batch = 10
    i = 0
    py7zr_failed = False
    print(f"[*] Collecting samples with batch size: {batch}")
    while i < len(yHashes):
        samples = defaultdict(list)
        for i, sampleHash in enumerate(yHashes):
            print(f"Reading hash location: {i}/{len(yHashes)}", end="\r")
            if i <= existingIdx:
                continue # skip if done

            
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
            try:
                if py7zr_failed != archive: 
                    # py7zr is faster for batch extraction
                    peBytes = extractSamplesWithPy7zr(os.path.join(archivePath, archive), archive, samples)
                else:
                    peBytes = extractSamplesWith7z(archivePath, archive, samples)
            except (py7zr.exceptions.CrcError, LZMAError) as e:
                print(f"[-] '{e}': extration failed, falling back to 7z")
                # one (or more) of the PEs is corrupted, in that case py7zr fails to extract it
                # using cmdline 7z to extract them, which do not cares
                peBytes = extractSamplesWith7z(archivePath, archive, samples)
                py7zr_failed = archive # py7zr will fail for next X (?) files in this archive
            
            # add timestamp to log
            now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f" {now}: [*] Extracting Ember features for {batch} samples from {archive}...")
            for pe in tqdm(peBytes):
                vector = extractor.feature_vector(peBytes[pe])
                peHash = pe.split("/")[-1]
                arr[yHashes.index(peHash)] = vector
            
            # dump BROKEN_FILES
            if BROKEN_FILES:
                # read existing broken files and apped new ones
                if os.path.exists(os.path.join(outPath, f"broken_files.json")):
                    with open(os.path.join(outPath, f"broken_files.json"), "r") as f:
                        old = json.load(f)
                    BROKEN_FILES = sorted((set(old + BROKEN_FILES)))
                with open(os.path.join(outPath, f"broken_files.json"), "w") as f:
                    json.dump(BROKEN_FILES, f, indent=4)
                print(f"[!] Updated broken files list, current list length: {len(BROKEN_FILES)}")
                BROKEN_FILES = []
                
        
        np.save(os.path.join(f"{outPath}", f"x_{dataset}_{i}.npy"), arr)
        del samples
        # delete file with previous index
        try:
            os.remove(os.path.join(f"{outPath}", f"x_{dataset}_{i-batch}.npy"))
        except FileNotFoundError:
            pass
        if i % 100 == 0: # reset py7zr failed flag after some time
            py7zr_failed = False