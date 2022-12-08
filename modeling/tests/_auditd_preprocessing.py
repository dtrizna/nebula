import sys
sys.path.append('../..')

from utils.normalization import normalizeAuditdTable
from utils.misc import getScriptPath
from utils.preprocessing import readAndFilterFile
from utils.constants import *
import time
import os
import pandas as pd

SCRIPT_PATH = getScriptPath()

# input
DATA_FOLDER = SCRIPT_PATH + "\\..\\..\\data\\"
LIMIT = None
files = os.listdir(DATA_FOLDER+"auditd_msft_raw\\")[:LIMIT]
timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Processing {len(files)} files! Reading and filtering...")

out = []
for file in files:
    ldf = readAndFilterFile(DATA_FOLDER+"auditd_msft_raw\\"+file, normalizedFields=AUDITD_FIELDS, filterDict={"rule.sidid": AUDITD_TYPES})
    out.append(ldf)
out = pd.concat(out)
out.fillna("(none)", inplace=True)

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Normalizing...")
df = normalizeAuditdTable(out.copy())

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Groupby loop...")
groupByCols = ['hostname']#, 'process.ppid']
l = df.groupby(groupByCols).size().shape[0]
for i, (group, gdf) in enumerate(df.groupby(groupByCols)):
    #file = "_".join(group) + ".json"
    file = group + ".json"
    print(f"{i+1}/{l} -- {file+' '*35}", end='\r')
    with open(DATA_FOLDER+"auditd_msft_grouped\\"+file, "w") as f:
        f.write(gdf.drop(columns=groupByCols).to_json(orient='records', indent=4))

timenow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
print(f"{timenow}: Finished!")
