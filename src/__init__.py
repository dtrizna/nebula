import os
from collections import Counter
import numpy as np

from .constants import SPEAKEASY_RECORDS
from .filters import getRecordsFromReport
from .normalization import normalizeTableIP, normalizeTablePath, joinSpeakEasyRecordsToJSON

import speakeasy
from pefile import PEFormatError
from unicorn import UcError

import logging
from pathlib import Path


class DynamicFeatureExtractor():
    def __init__(self, outputFolder=None, speakeasyConfig=None):
        self.outputFolder = outputFolder
        self.speakeasyConfig = speakeasyConfig

    def _createErrorFile(errfile):
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
        
        except PEFormatError as ex:
            logging.error(f" [-] Failed emulation, PEFormatError: {file}\n{ex}\n")
            return None
        except UcError as ex:
            logging.error(f" [-] Failed emulation, UcError: {file}\n{ex}\n")
            return None
        except IndexError as ex:
            logging.error(f" [-] Failed emulation, IndexError: {file}\n{ex}\n")
            return None
        except speakeasy.errors.NotSupportedError as ex:
            logging.error(f" [-] Failed emulation, NotSupportedError: {file}\n{ex}\n")
            return None
        except speakeasy.errors.SpeakeasyError as ex:
            logging.error(f" [-] Failed emulation, SpeakEasyError: {file}\n{ex}\n")
            return None
        except Exception as ex:
            logging.error(f" [-] Failed emulation, general Exception: {file}\n{ex}\n")
            return None
    
    def emulate(self, path=None, data=None):
        if path is None and data is None:
            raise ValueError("Either 'file' or 'data' must be specified.")
        if path:
            if not os.path.exists(path):
                raise ValueError(f"File {path} does not exist.")
            self.sampleName = os.path.basename(path)
        self.path = path
        self.data = data
        report = self._emulation(self.speakeasyConfig, self.path, self.data)
        # TODO
        # if outputFolder
        if report:
            took = report["emulation_total_runtime"]
            api_seq_len = sum([len(x["apis"]) for x in report["entry_points"]])
        else:
            api_seq_len = 0

        if api_seq_len == 0:
            return None
        else:
            # 1 API call is sometimes already enough
            recordDict = getRecordsFromReport(report["entry_points"], recordFields=SPEAKEASY_RECORDS)
            # normalize 
            recordDict['network_events.traffic'] = normalizeTableIP(recordDict['network_events.traffic'], col='server')
            recordDict['file_access'] = normalizeTablePath(recordDict['file_access'], col='path')

            recordJson = joinSpeakEasyRecordsToJSON(recordDict)
            return recordJson