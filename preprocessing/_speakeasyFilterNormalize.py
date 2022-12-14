import os
import time
import sys
sys.path.append(".")
sys.path.append("..")
from src.misc import getRealPath
from src.filters import getRecordsFromFile
from src.normalization import normalizeTableIP, normalizeTablePath, joinSpeakEasyRecordsToJSON

from utils.constants import SPEAKEASY_RECORDS, SPEAKEASY_SUBFILTER_MINIMALISTIC

PATH = getRealPath(type="script")
EMULATION_DATASET_PATH = PATH + "\\..\\data\\data_raw\\windows_emulationDataset\\"
subFolders = [x for x in os.listdir(EMULATION_DATASET_PATH) if x.startswith("report_")]
LIMIT = None

PUT_TO_FOLDER = PATH + "\\..\\data\\data_filtered\\speakeasy_trainset\\"

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
timenowStart = time.time()
print(f"{timenow}: Filtering and normalizing speakeasy dataset...")

for subFolder in subFolders:
    fullPath = f"{EMULATION_DATASET_PATH}{subFolder}"
    files = [f"{fullPath}\\{x}" for x in os.listdir(fullPath)[:LIMIT] if x.endswith(".json")]
    if subFolder == 'report_backdoor':
        files = [f"{fullPath}\\fb7fc834c884cd1596e49b5a78cba4efa14fbd66ce6c12f160d8ef647c87ecf4.json"] + files	
    l = len(files)
    for i,file in enumerate(files):
        print(f"{subFolder:>20}: {i+1}/{l} {' '*30}", end="\r")
        
        # read and filter
        recordDict = getRecordsFromFile(files[0], SPEAKEASY_RECORDS)
        
        # normalize 
        recordDict['network_events.traffic'] = normalizeTableIP(recordDict['network_events.traffic'], col='server')
        recordDict['file_access'] = normalizeTablePath(recordDict['file_access'], col='path')
        
        # parse back to JSON
        jsonEvent = joinSpeakEasyRecordsToJSON(recordDict, subFilter=SPEAKEASY_SUBFILTER_MINIMALISTIC)
        
        outputFolder = f"{PUT_TO_FOLDER}{subFolder}"
        os.makedirs(outputFolder, exist_ok=True)
        hashFileName = file.split("\\")[-1]
        with open(f"{outputFolder}\\{hashFileName}", "w") as f:
            f.write(jsonEvent)

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
timenowEnd = time.time()
print(f"{timenow}: Finished... Took: {timenowEnd - timenowStart:.2f}s")