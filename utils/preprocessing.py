import re
import orjson
from numpy import where
from pandas import json_normalize, to_datetime, DataFrame
from .constants import AUDITD_FIELDS, AUDITD_TYPES, FIELD_SEPARATOR
from .misc import filterDictByKeys


def normalizeCmd(cmd,
                custom_domains=["skype.com", "microsoft.com", "azure.com", "example.com"], 
                placeholder="_IPADDRESS_"
                ):
    # normalize IP addresses
    cmd = re.sub(r"([0-9]{1,3}\.){3}[0-9]{1,3}", placeholder, cmd)
    # normalize localhosts
    cmd = cmd.replace("localhost", placeholder)
    for domain in custom_domains:
        cmd = cmd.replace(domain, placeholder)
    return cmd


def ipLabeler(ldf):
    col = 'auditd.summary.object.primary'
    ldf[col] = where(ldf[col].str.startswith('127.'), "(lopIP)", ldf[col])
    ldf[col] = where(ldf[col].str.match('169.254.169.254'), "(imds)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'(^10\.)|(^172\.1[6-9]\.)|(^172\.2[0-9]\.)|(^172\.3[0-1]\.)|(^192\.168\.)'), "(prvIP)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$'), "(pubIP)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'^([0-9a-fA-F]{1,4}:){1,7}:'), "(IPv6)", ldf[col])
    return ldf


def auditdReadAndFilterFile(filename,  
                            auditd_fields=AUDITD_FIELDS,
                            auditd_types=AUDITD_TYPES):
    
    with open(filename, "rb") as f:
        # [:-1] since last element of JSON is non event
        try:
            data = orjson.loads(f.read())[:-1]
        except orjson.JSONDecodeError as ex:
            print(f"File: {filename} -- JSONDecodeError:\n\t", ex)
            return DataFrame()

    # preserve only necessary fields
    data_filtered = [filterDictByKeys(x) for x in data]
    
    # need to again specify column names to inforce ordering
    df = json_normalize(data_filtered)[auditd_fields]

    # for all events that has process.args -- put that value in process.title
    df['process.title'] = where(df['process.args'].notna(), df['process.args'], df['process.title'])
    df.drop(columns=['process.args'], inplace=True)

    # filter only necessary event types and columns
    newdf = df[df['rule.sidid'].isin(auditd_types)].copy()
    return newdf


def auditdPreprocessTable(newdf, lengthLimit=20000):
    """
    input is a dataframe with 5min of data
    output is a string with data in fastText format
    length is the maximum length of the telemtry
          20 000 corresponds to ~80 percentile from 61k of hosts
          40 000 corresponds to ~90 percentile from 61k of hosts

    # CONSIDER:
        # my data has been already divided to 5min chunks
        # it might be needed to do chunking here based on TimeStamp if needed
    """
    # timestamp -- sorting and discaring (might want preserve diff for future?)
    newdf['TimeStamp'] = to_datetime(newdf.TimeStamp)
    newdf.sort_values(by=['TimeStamp'], inplace=True)
    newdf.drop(columns=['TimeStamp'], inplace=True)

    # dealing with NaN to avoid errors
    newdf.fillna("(none)", inplace=True)
    
    # ppid preprocessing
    newdf['process.ppid'] = newdf['process.ppid'].apply(lambda x: x if x == "1" else "(pid)")    
    
    # ip address replacement
    newdf = ipLabeler(newdf)
    
    # adds this as last column
    newdf['event.separator'] = "(sep)"

    out = []
    for _, groupDf in newdf.groupby('hostname'):    
        arr = groupDf.drop(columns=['hostname']).values.flatten()
        # fastText expects whitespace separated values in utf-8
        # I additionally introduce ',' since process.title has spaces
        host_telemetry = FIELD_SEPARATOR.join(arr).encode().decode('utf-8', 'ignore') 
        out.append(host_telemetry[:lengthLimit]+"\n")

    return out

