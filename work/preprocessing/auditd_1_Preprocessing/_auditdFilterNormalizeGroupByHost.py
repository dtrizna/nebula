import sys
sys.path.append('.')
sys.path.append('..')

from src.normalization import normalizeAuditdTable
from src.misc import getRealPath
from src.filters import readAndFilterFile
from src.constants import *
import time
import os
import pandas as pd

SCRIPT_PATH = getRealPath(type="script")

# input
DATA_FOLDER = SCRIPT_PATH + "\\..\\data\\"
READ_FROM_FOLDER = DATA_FOLDER + "auditd_msft_raw_benign\\"
#READ_FROM_FOLDER = DATA_FOLDER + "auditd_msft_raw_malicious\\Vanquish_RedTeam_Hosts\\"
#READ_FROM_FOLDER = DATA_FOLDER + "auditd_msft_raw_malicious\\ML_Pen_Tester_Hosts\\"
LIMIT = None
files = [x for x in os.listdir(READ_FROM_FOLDER)[:LIMIT] if x.endswith(".json")]
timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Processing {len(files)} files! Reading and filtering...")

# output
PUT_TO_FOLDER = DATA_FOLDER+"auditd_msft_groups_benign_filtered\\"
#PUT_TO_FOLDER = DATA_FOLDER+"auditd_msft_groups_malicious_filtered\\"
os.makedirs(PUT_TO_FOLDER, exist_ok=True)

out = []
for file in files:
    ldf = readAndFilterFile(READ_FROM_FOLDER+file, 
                            normalizedFields=AUDITD_FIELDS, 
                            filterDict={"rule.sidid": AUDITD_TYPES})
    out.append(ldf)
out = pd.concat(out)
out.fillna("(none)", inplace=True)

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Normalizing...")
df = normalizeAuditdTable(out.copy())

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Groupby loop...")
groupByCols = ['hostname']#, 'TimeStamp']
l = df.groupby(groupByCols).size().shape[0]
for i, (group, gdf) in enumerate(df.groupby(groupByCols)):
    file = group + ".json"
    print(f"{i+1}/{l} -- {file+' '*35}", end='\r')
    with open(PUT_TO_FOLDER+file, "w") as f:
        f.write(gdf.drop(columns=groupByCols).to_json(orient='records', indent=4, date_format='iso'))

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Finished!")
