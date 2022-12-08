import re
from numpy import where
from pandas import to_datetime
from .constants import FIELD_SEPARATOR

def labelFieldIP(df, col = 'auditd.summary.object.primary'):
    # NOTE: 
    # it is faster than using apply() on each row
    # however, applies only to fields that has explicitly IPs as values
    ldf = df.copy()
    ldf[col] = where(ldf[col].str.match(r'^127\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'), "(lopIP)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'^169\.254\.169\.254'), "(imds)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'(^10\.)|(^172\.1[6-9]\.)|(^172\.2[0-9]\.)|(^172\.3[0-1]\.)|(^192\.168\.)'), "(prvIP)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'), "(pubIP)", ldf[col])
    ldf[col] = where(ldf[col].str.match(r'^([0-9a-fA-F]{1,4}:){1,7}:'), "(IPv6)", ldf[col])
    return ldf

def labelStringDomain(string,
                    domain_regex = r"\b([a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.)+",
                    top_level_domains_regex = r"(com|net|ai|us|uk|cz|gov)\b",
                    placeholder = "(domain)"):
    return re.sub(domain_regex+top_level_domains_regex, placeholder, string)

def labelStringIP(string):
    string = re.sub(r'127\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', "(lopIP)", string)
    string = re.sub(r'169\.254\.169\.254', "(imds)", string)
    string = re.sub(r'(10(\.(25[0-5]|2[0-4][0-9]|1[0-9]{1,2}|[0-9]{1,2})){3}|((172\.(1[6-9]|2[0-9]|3[01]))|192\.168)(\.(25[0-5]|2[0-4][0-9]|1[0-9]{1,2}|[0-9]{1,2})){2})', "(prvIP)", string)
    string = re.sub(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', "(pubIP)", string)
    #string = re.sub(r'([0-9a-fA-F]{1,4}:){1,7}:', "(IPv6)", string) # exclude IPv6 for now
    return string

def sortTime(df, col='TimeStamp'):
    df[col] = to_datetime(df.TimeStamp)
    df.sort_values(by=[col], inplace=True)

def swapNotnaColumn(df, col_orig, col_replace):
    # for all events that has col_orig -- put that value in col_replace
    return where(df[col_orig].notna(), df[col_orig], df[col_replace])

def normalizeAuditdTable(df):
    df.fillna("(none)", inplace=True)
    # sort by TimeStamp and drop
    sortTime(df, col= 'TimeStamp')
    df.drop(columns=['TimeStamp'], inplace=True)
    # process args replace process title
    df['process.title'] = swapNotnaColumn(df, "process.args", "process.title")
    df.drop(columns=['process.args'], inplace=True)    
    # ip and domain name normalization
    df = labelFieldIP(df, col = 'auditd.summary.object.primary')
    df['process.title'] = df['process.title'].apply(labelStringIP)
    df['process.title'] = df['process.title'].apply(labelStringDomain)
    # ppid preprocessing
    df['process.ppid'] = df['process.ppid'].apply(lambda x: x if x == "1" else "(pid)")
    # adds this as last column
    df['event.separator'] = "(sep)"
    return df

def groupAuditdSequences(newdf, groupByCols=['hostname'], lengthLimit=20000):
    """
    input is a dataframe with 5min of data
    output is a string with auditd data for a host
    length is the maximum length of the telemtry
          20 000 corresponds to ~80 percentile from 61k of hosts
          40 000 corresponds to ~90 percentile from 61k of hosts
    # CONSIDER:
        # my data has been already divided to 5min chunks
        # it might be needed to do chunking here based on TimeStamp if needed
    """
    newdf = normalizeAuditdTable(newdf.copy())

    out = []
    for _, groupDf in newdf.groupby(groupByCols):    
        arr = groupDf.drop(columns=groupByCols).values.flatten()
        # fastText expects whitespace separated values in utf-8
        # I additionally introduce ',' in FIELD_SEPARATOR since process.title has spaces
        host_telemetry = FIELD_SEPARATOR.join(arr).encode().decode('utf-8', 'ignore') 
        out.append(host_telemetry[:lengthLimit]+"\n")

    return out
