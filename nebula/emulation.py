import os
import logging
from time import time
from pathlib import Path

import speakeasy
import re

from pandas import json_normalize, concat, DataFrame
from nebula.normalization import normalizeTableIP, normalizeTablePath
from nebula.constants import *
from nebula.misc import getAlphaNumChars

class PEDynamicFeatureExtractor(object):
    def __init__(self, 
                    speakeasyConfig=None, 
                    speakeasyRecords=SPEAKEASY_RECORDS,
                    recordSubFilter=SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC,
                    recordLimits=SPEAKEASY_RECORD_LIMITS
                ):
        
        self.speakeasyConfig = speakeasyConfig
        if self.speakeasyConfig and not os.path.exists(speakeasyConfig):
            raise Exception(f"Speakeasy config file not found: {self.speakeasyConfig}")
        
        self.speakeasyRecords = speakeasyRecords
        self.recordSubFilter = recordSubFilter
        self.recordLimits = recordLimits

    def _createErrorFile(self, errfile):
        # just creating an empty file to incdicate failure
        Path(errfile).touch()
    
    def _emulation(self, config, path, data):
        try:
            file = path if path else str(data[0:15])
            se = speakeasy.Speakeasy(config=config)
            if path:
                module = se.load_module(path=path)
            if data:
                module = se.load_module(data=data)
            se.run_module(module)
            return se.get_report()
        except SPEAKEASY_EXCEPTIONS as ex:
            logging.error(f" [-] Failed emulation of {file}\nException: {ex}\n")
            return None
        except Exception as ex:
            logging.error(f" [-] Failed emulation, general Exception: {file}\n{ex}\n")
            return None
    
    def emulate(self, path=None, data=None, emulationOutputFolder=None):
        
        if emulationOutputFolder:
            os.makedirs(emulationOutputFolder, exist_ok=True)
        
        if path is None and data is None:
            raise ValueError("Either 'file' or 'data' must be specified.")
        if path:
            if not os.path.exists(path):
                raise ValueError(f"File {path} does not exist.")
            self.sampleName = os.path.basename(path).replace(".exe", "")
        else:
            self.sampleName = f"{time()}"
        report = self._emulation(self.speakeasyConfig, path, data)
        
        if emulationOutputFolder:
            if report:
                with open(os.path.join(self.outputFolder, f"{self.sampleName}.json"), "w") as f:
                    f.write(report["entry_points"])
            else:
                self._createErrorFile(os.path.join(self.outputFolder, f"{self.sampleName}.err"))

        api_seq_len = sum([len(x["apis"]) for x in report["entry_points"]]) if report else 0
        if api_seq_len == 0:
            return None
        else:
            return self.parseReportEntryPoints(report["entry_points"])
    
    def parseReportEntryPoints(self, entryPoints):
        # clean up report
        recordDict = self.getRecordsFromReport(entryPoints)
        
        # filter out events with uninformative API sequences
        # i.e. emulation failed extract valuable info
        if recordDict['apis'].shape[0] == 1 and \
           recordDict['apis'].iloc[0].api_name == 'MSVBVM60.ordinal_100':
            return None

        # normalize
        recordDict['network_events.traffic'] = normalizeTableIP(recordDict['network_events.traffic'], col='server')
        recordDict['file_access'] = normalizeTablePath(recordDict['file_access'], col='path')
        
        # normalize args to exclude any non-alphanumeric characters
        if 'args' in recordDict['apis'].columns:
            # filter unicode '\uXXXX' values from args which is list of strings using re.sub
            recordDict['apis']['args'] = recordDict['apis']['args'].apply(lambda x: [getAlphaNumChars(y) for y in x])
    
        # limit verbose fields to a certain number of records
        if self.recordLimits:
            for field in self.recordLimits.keys():
                if field in recordDict.keys():
                    recordDict[field] = recordDict[field].head(self.recordLimits[field])
        # join 
        recordJson = self.joinRecordsToJSON(recordDict)
        return recordJson
    
    def getRecordsFromReport(self, entryPoints):
        records = dict()
        for recordField in self.speakeasyRecords:
            recordList = [json_normalize(x, record_path=[recordField.split('.')]) for x in entryPoints if recordField.split('.')[0] in x]
            records[recordField] = concat(recordList) if recordList else DataFrame()
        return records

    def joinRecordsToJSON(self, recordDict):
        jsonEvent = "{"
        for i, key in enumerate(recordDict.keys()):
            if recordDict[key].empty:
                continue
            if key in self.recordSubFilter.keys():
                jsonVal = recordDict[key][self.recordSubFilter[key]].to_json(orient='records', indent=4)
            else:
                jsonVal = recordDict[key].to_json(orient='records', indent=4)
            jsonEvent += f"\n\"{key}\":\n{jsonVal}"

            if i != len(recordDict.keys())-1:
                jsonEvent += ","

        if jsonEvent.endswith(","):
            jsonEvent = jsonEvent[:-1]
        jsonEvent += "}"
        return jsonEvent
