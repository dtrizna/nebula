import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from utils.preprocessing import auditdBuildSequenceFromTable, auditdReadAndFilterFile
from utils.misc import getScriptPath

if __name__ == "__main__":

    SCRIPT_PATH = getScriptPath()
    
    # input
    DATA_FOLDER = SCRIPT_PATH + "\\..\\..\\data\\auditd_msft_raw\\"
    LIMIT = None
    files = os.listdir(DATA_FOLDER)[:LIMIT]
    
    # output
    OUT_FILE = SCRIPT_PATH + "\\auditd_parsed.out"
    #OUT_FILE = SCRIPT_PATH + "\\auditd_parsed_noLengthLimit.out"
    open(OUT_FILE, 'w').close() # clears file if exists

    for i,file in enumerate(files):
        print(f"Processing file: {i+1}/{len(files)}", end="\r")
        
        fileFullPath = DATA_FOLDER + file
        df = auditdReadAndFilterFile(fileFullPath)

        # run preprocessing
        out = auditdBuildSequenceFromTable(df)
        #out = auditdBuildSequenceFromTable(df, lengthLimit=None)
        with open(OUT_FILE, "a", encoding='utf-8') as f:
            f.writelines(out)
