import re
from numpy import where
from pandas import to_datetime

from nebula.constants import VARIABLE_MAP, SPEAKEASY_LABELMAP

import orjson
import time
import logging
import os
from tqdm import tqdm

def normalizeTablePath(df, col="path"):
    if df.empty:
        return df
    ldf = df.copy()
    ldf[col] = ldf[col].apply(normalizeStringPath)
    ldf[col] = ldf[col].apply(normalizeStringHash)
    return ldf

def normalizeTableIP(df, col = 'auditd.summary.object.primary'):
    # NOTE: it is faster than using apply() on each row
    # however, applies only to fields that has explicitly IPs as values
    if df.empty:
        return df
    ldf = df.copy()
    ldf[col] = where(ldf[col].str.match(r'^127\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'), "<lopIP>", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'^169\.254\.169\.254'), "<imds>", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'(^10\.)|(^172\.1[6-9]\.)|(^172\.2[0-9]\.)|(^172\.3[0-1]\.)|(^192\.168\.)'), "<prvIP>", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'), "<pubIP>", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'^([0-9a-fA-F]{1,4}:){1,7}:'), "<IPv6>", ldf[col])
    return ldf

def normalizeStringHash(string):
    string = re.sub(r'[0-9a-fA-F]{64}', "<sha256>", string)
    string = re.sub(r'[0-9a-fA-F]{40}', "<sha1>", string)
    string = re.sub(r'[0-9a-fA-F]{32}', "<md5>", string)
    return string

def normalizeStringDomain(string,
                    domain_regex = r"\b([a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.)+",
                    top_level_domains_regex = r"(com|net|ai|us|uk|cz|gov)\b",
                    placeholder = "<domain>"):
    return re.sub(domain_regex+top_level_domains_regex, placeholder, string)

def normalizeStringIP(string):
    string = re.sub(r'127\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', "<lopIP>", string)
    string = re.sub(r'169\.254\.169\.254', "<imds>", string)
    string = re.sub(r'(10(\.(25[0-5]|2[0-4][0-9]|1[0-9]{1,2}|[0-9]{1,2})){3}|((172\.(1[6-9]|2[0-9]|3[01]))|192\.168)(\.(25[0-5]|2[0-4][0-9]|1[0-9]{1,2}|[0-9]{1,2})){2})', "<prvIP>", string)
    string = re.sub(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', "<pubIP>", string)
    #string = re.sub(r'([0-9a-fA-F]{1,4}:){1,7}:', "<IPv6>", string) # exclude IPv6 for now
    return string

def normalizeStringPath(path):
    """Function that takes a path string and returns a normalized version,
    but substituting: (1) drive letters with [drive] (2) network hosts with [net] 
                      (3) arbitrary, non-default usernames with [user] 
                      (4) environment variables with fullpath equivalent

    Args:
        path (str): String containing file-path, e.g. "C:\\users\\ieuser\\desktop\\my.exe"
        verbose (bool, optional): Whether to print file-path normalization steps. Defaults to False.

    Returns:
        str: Normalized path, e.g. "[drive]\\users\\[user]\\desktop\\my.exe"
    """
    # X. some paths have "*raw:", "*amsiprocess:", "script started by " auxiliary strings...
    path = path.lower().replace("*raw:","").replace("*amsiprocess:","").replace("a script started by ","").strip()
    
    # 1a. Normalize drive
    # "c:\\"" or "c:\"" will be [drive]\
    path = re.sub(r"\w:\\{1,2}", r"<drive>\\", path)
    
    # 2. Normalize network paths, i.e. need to change "\\host\" to "[net]\"
    # before that take care of "\;lanmanredirector" or "\;webdavredirector" paths
    path = re.sub(r"[\\]{1,2};((lanmanredirector|webdavredirector)\\;){0,1}\w\:[a-z0-9]{16}", "\\\\", path)
    # [\w\d\.\-]+ is DNS pattern, comes from RFC 1035 and:
    # https://docs.microsoft.com/en-us/troubleshoot/windows-server/identity/naming-conventions-for-computer-domain-site-ou#dns-host-names
    path = re.sub(r"\\\\[\w\d\.\-]+\\", r"<net>\\", path)
    
    # 1b. Normalize drive (2) - you can refer to path as "dir \user\ieuser\desktop" in windows
    # if starts with \, and not \\ (captures not-\\ as \1 group and preserves), then add [drive]
    path = re.sub(r"^\\([^\\])", r"<drive>\\\1", path)
    
    # 1c. Normalize "\\?\Volume{614d36cf-0000-0000-0000-10f915000000}\" format
    path = re.sub(r"\\[\.\?]\\volume\{[a-z0-9\-]{36}\}", r"<drive>", path)
    
    # 3. normalize non-default users
    default_users = ["administrator", "public", "default"]
    if "users\\" in path:
        # default user path, want to preserve them
        if not any([True if "users\\"+x+"\\" in path else False for x in default_users]):
            path = re.sub(r"users\\[^\\]+\\", r"users\\<user>\\", path)
    
    # 4. Normalize environment variables with actual paths
    for k,v in VARIABLE_MAP.items():
        path = path.replace(k, v)
    
    return path

def sortTime(df, col='TimeStamp'):
    df[col] = to_datetime(df.TimeStamp)
    df.sort_values(by=[col], inplace=True)    

def swapNotnaColumn(df, col_orig, col_replace):
    # for all events that has col_orig -- put that value in col_replace
    return where(df[col_orig].notna(), df[col_orig], df[col_replace])

def normalizeAuditdTable(df):
    df.fillna("<none>", inplace=True)
    # sort by TimeStamp and drop
    sortTime(df, col='TimeStamp')
    # process args replace process title
    df['process.title'] = swapNotnaColumn(df, "process.args", "process.title")
    df.drop(columns=['process.args'], inplace=True)    
    # ip, domain name, and hash normalization
    df = normalizeTableIP(df, col = 'auditd.summary.object.primary')
    df['process.title'] = df['process.title'].apply(normalizeStringIP)
    df['process.title'] = df['process.title'].apply(normalizeStringDomain)
    df['process.title'] = df['process.title'].apply(normalizeStringHash)
    # ppid preprocessing
    df['process.ppid'] = df['process.ppid'].apply(lambda x: x if x == "1" else "<pid>")
    return df

def read_and_filter_json_folders(subFolders, benign_folders, filter_function=None, limit=None, multiclass=False):
    """
    Reads and filters all json files in subFolders. Returns a list of events and a list of y values.
    """
    events, y, y_filepaths = [], [], []
    for subFolder in subFolders:
        timenowStart = time.time()
        logging.warning(f"Filtering and normalizing {subFolder.strip()}")

        files = [os.path.join(subFolder, x) for x in os.listdir(subFolder)[:limit] if x.endswith(".json")]
        for file in tqdm(files):
            with open(file, "r") as f:
                jsonEventRaw = orjson.loads(f.read())

            if filter_function is not None:
                jsonEventFiltered = filter_function(jsonEventRaw)
            else:
                jsonEventFiltered = jsonEventRaw
            if jsonEventFiltered:
                events.append(jsonEventFiltered)
                
                # if you want only hash of binary, uncomment this
                # file = os.path.basename(file).rstrip('.json')
                y_filepaths.append(file)
                if os.path.basename(subFolder) in benign_folders:
                    y.append(0)
                else:
                    if not multiclass:
                        y.append(1)
                    else:
                        malware_type = os.path.basename(subFolder).split("_")[1]
                        y.append(SPEAKEASY_LABELMAP[malware_type])
        timenowEnd = time.time()
        logging.warning(f"Finished... Took: {timenowEnd - timenowStart:.2f}s")
    return events, y, y_filepaths
