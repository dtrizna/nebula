import orjson
from pandas import json_normalize, DataFrame, concat
from .misc import filterDictByKeys
from .constants import SPEAKEASY_RECORDS

def readAndFilterEvent(jsonEvent,
              jsonType="normalized",
              normalizedFields=None,
              filterDict=None):
    """Takes str of JSON and returns a filtered DataFrame.
    NOTE: Per event parsing is ~50 times slower than reading events in bulk from file with readAndFilterFile()!

    Args:
        jsonEvent (str, bytes, dict): raw event in JSON format
        normalizedFields (dict): {"field": ["value1", "value2"]} to preserve from event
        jsonType (str): "normalized" or "nested"
        filterDict (list): list of normalized fields to preserve from event like 'process.title'
    """
    if jsonType not in ("normalized", "nested"):
        raise ValueError("readAndFilterEvent(): jsonType must be 'normalized' or 'nested'")

    if isinstance(jsonEvent, (str, bytes)):
        jsonEvent = orjson.loads(jsonEvent)

    # if event has normalized fields, can filter keys before json_normalize (faster)
    if normalizedFields and jsonType == "normalized":
        jsonEvent = filterDictByKeys(jsonEvent, key_list=normalizedFields)

    ldf = json_normalize(jsonEvent)

    # if event is nested (aka hierarchical), need to filter columns after normalization (slower)
    if normalizedFields and jsonType == "nested":
        # filter elemts in normalizedFields that are in ldf columns
        normalizedFields = [field for field in normalizedFields if field in ldf.columns]
        #normalizedFields = list(set(normalizedFields).intersection(set(ldf.columns)))
        ldf = ldf[normalizedFields].copy()

    if filterDict:
        for field, values in filterDict.items():
            ldf = ldf[ldf[field].isin(values)].copy()
    return ldf


def readAndFilterFile(jsonFile,  
                      jsonType="normalized",
                      normalizedFields=None,
                      filterDict=None,
                      fillna=True):
    """Reads JSON file and returns a filtered DataFrame.

    Args:
        jsonFile (str): Filepath to JSON file with normalized fields (e.g. 'process.title') 
                        where Events are in list format: [{event1}, {event2}, ...]
        jsonType (str): "normalized" or "nested"
        normalizedFields (list): list of normalized fields to preserve from event (e.g. 'process.title')
        filterDict (dict): {"normalizedField": ["fieldValue1", "fieldValue2"]} to preserve from event

    Returns:
        pd.DataFrame: table with filtered events
    """
    if jsonType not in ("normalized", "nested"):
        raise ValueError("readAndFilterEvent(): jsonType must be 'normalized' or 'nested'")

    with open(jsonFile, "rb") as f:
        try: # [:-1] since last element of JSON is non event
            data = orjson.loads(f.read())#[:-1]
        except orjson.JSONDecodeError as ex:
            print(f"File: {jsonFile} -- JSONDecodeError:\n\t", ex)
            return DataFrame()
    
    # if event has normalized fields, can filter keys before json_normalize (faster)
    if normalizedFields and jsonType == "normalized":
        data = [filterDictByKeys(x, key_list=normalizedFields) for x in data]

    df = json_normalize(data)
    
    # need to again specify column names to enforce ordering
    if normalizedFields and jsonType in ("nested", "hierachical"):
        # filter elemts in normalizedFields that are in ldf columns
        normalizedFields = [field for field in normalizedFields if field in df.columns]
        df = df[normalizedFields].copy()

    # filter only necessary event types and columns
    if filterDict:
        for field, values in filterDict.items():
            df = df[df[field].isin(values)].copy()

    if fillna:
        df.fillna(r"<none>", inplace=True)
    
    return df

def getRecordsFromReport(entryPoints, recordFields=SPEAKEASY_RECORDS):
    records = dict()
    for recordField in recordFields:
        recordList = [json_normalize(x, record_path=[recordField.split('.')]) for x in entryPoints if recordField.split('.')[0] in x]
        records[recordField] = concat(recordList) if recordList else DataFrame()
    return records

def getRecordsFromFile(jsonFile, recordFields = SPEAKEASY_RECORDS):
    """Reads JSON file and returns a filtered DataFrame.

    Args:
        jsonFile (str): Filepath to JSON file with normalized fields (e.g. 'process.title') 
                        where Events are in list format: [{event1}, {event2}, ...]
        recordFields (list): list of record fields to preserve from event (e.g. 'process.title')

    Returns:
        pd.DataFrame: table with filtered events
    """
    with open(jsonFile, "rb") as f:
        try: # [:-1] since last element of JSON is non event
            data = orjson.loads(f.read())#[:-1]
        except orjson.JSONDecodeError as ex:
            print(f"File: {jsonFile} -- JSONDecodeError:\n\t", ex)
            return DataFrame()
    
    records = dict()
    for recordField in recordFields:
        recordList = [json_normalize(x, record_path=[recordField.split('.')]) for x in data if recordField.split('.')[0] in x]
        records[recordField] = concat(recordList) if recordList else DataFrame()
    return records