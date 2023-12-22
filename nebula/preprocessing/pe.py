import os
import json
import logging
from time import time
from pathlib import Path

import lief
import speakeasy

import nebula
from nebula.constants import *
from nebula.models.ember import PEFeatureExtractor
from nebula.misc import get_alphanum_chars

from .tokenization import JSONFilter
from .normalization import normalizeTableIP, normalizeTablePath


def is_pe_file(file_path):
    try:
        # Parse the file
        binary = lief.parse(file_path)

        # Check if it's PE
        if isinstance(binary, lief.PE.Binary):
            return True
    except (lief.bad_format, lief.bad_file, lief.pe_error, lief.parser_error):
        # Exceptions indicate it's likely not a PE or the file can't be parsed.
        pass
    return False


class PEStaticFeatureExtractor(object):
    def __init__(self):
        self.extractor = PEFeatureExtractor(print_feature_warning=False)
        
    def feature_vector(self, bytez):
        return self.extractor.feature_vector(bytez)


class PEDynamicFeatureExtractor(object):
    def __init__(self, 
                    speakeasyConfig=None, 
                    speakeasyRecordFields=SPEAKEASY_RECORD_FIELDS,
                    recordLimits=SPEAKEASY_RECORD_LIMITS,
                    emulationOutputFolder=None
                ):
        
        # setup speakseasy config
        if speakeasyConfig is None:
            speakeasyConfig = os.path.join(os.path.dirname(nebula.__file__), "objects", "speakeasy_config.json")
        if isinstance(speakeasyConfig, dict):
            self.speakeasyConfig = speakeasyConfig
        else:
            assert os.path.exists(speakeasyConfig), f"Speakeasy config file not found: {speakeasyConfig}"
            with open(speakeasyConfig, "r") as f:
                self.speakeasyConfig = json.load(f)
        
        self.recordLimits = recordLimits
        self.parser = JSONFilter(fields=speakeasyRecordFields)

        self.outputFolder = emulationOutputFolder
        if self.outputFolder:
            os.makedirs(emulationOutputFolder, exist_ok=True)

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
    
    def emulate(self, path=None, data=None):
        assert path or data, "Either 'file' or 'data' must be specified."
        if path:
            assert os.path.exists(path), f"File {path} does not exist."
            self.sampleName = os.path.basename(path).replace(".exe", "").replace(".dll", "")
        else:
            self.sampleName = f"{int(time())}"
        report = self._emulation(self.speakeasyConfig, path, data)
        
        if self.outputFolder:
            if report:
                with open(os.path.join(self.outputFolder, f"{self.sampleName}.json"), "w") as f:
                    json.dump(report, f, indent=4)
            else:
                self._createErrorFile(os.path.join(self.outputFolder, f"{self.sampleName}.err"))

        api_seq_len = sum([len(x["apis"]) for x in report["entry_points"]]) if report else 0
        return None if api_seq_len == 0 else self.filter_and_normalize_report(report["entry_points"])
    
    def filter_and_normalize_report(self, entryPoints):
        if isinstance(entryPoints, str) and os.path.exists(entryPoints):
            with open(entryPoints, "r") as f:
                entryPoints = json.load(f)
        # clean up report
        recordDict = self.parser.filter_and_concat(entryPoints)
        
        # filter out events with uninformative API sequences
        # i.e. emulation failed extract valuable info
        if 'apis' in recordDict and \
            recordDict['apis'].shape[0] == 1 and \
            recordDict['apis'].iloc[0].api_name == 'MSVBVM60.ordinal_100':
                return None

        # normalize
        if 'file_access' in recordDict:
            recordDict['file_access'] = normalizeTablePath(recordDict['file_access'], col='path')
        if 'network_events.traffic' in recordDict \
            and 'server' in recordDict['network_events.traffic'].columns:
                recordDict['network_events.traffic'] = normalizeTableIP(recordDict['network_events.traffic'], col='server')
        if 'network_events.dns' in recordDict \
            and 'query' in recordDict['network_events.dns'].columns:
            recordDict['network_events.dns']['query'] = recordDict['network_events.dns']['query'].apply(lambda x: ' '.join(x.split('.')))
        # normalize args to exclude any non-alphanumeric characters
        if 'args' in recordDict['apis'].columns:
            # filter unicode '\uXXXX' values from args which is list of strings using re.sub
            recordDict['apis']['args'] = recordDict['apis']['args'].apply(lambda x: [get_alphanum_chars(y) for y in x])
    
        # limit verbose fields to a certain number of records
        if self.recordLimits:
            for field in self.recordLimits.keys():
                if field in recordDict.keys():
                    recordDict[field] = recordDict[field].head(self.recordLimits[field])
        # join 
        recordJson = self.joinRecordsToJSON(recordDict)
        return recordJson
    
    @staticmethod
    def joinRecordsToJSON(recordDict):
        jsonEvent = "{"
        # sort in order to ensure consistent order of fields, put 'apis' at the end
        for i, key in enumerate(sorted(recordDict.keys(), reverse=True)):
            jsonVal = recordDict[key].to_json(orient='records')
            jsonEvent += f"\"{key}\":{jsonVal}"

            if i != len(recordDict.keys())-1:
                jsonEvent += ","

        if jsonEvent.endswith(","):
            jsonEvent = jsonEvent[:-1]
        jsonEvent += "}"
        return json.loads(jsonEvent)
