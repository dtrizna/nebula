import os
# TODO: think about more secure alternative to pickle
import pickle
from time import time
from nltk import WhitespaceTokenizer
from pandas import json_normalize, concat, DataFrame
import numpy as np

from .constants import *
from .normalization import normalizeTableIP, normalizeTablePath

import speakeasy
from pefile import PEFormatError
from unicorn import UcError

import logging
from pathlib import Path
from collections.abc import Iterable
from collections import Counter


class JSONTokenizer():
    def __init__(self, 
                patternCleanup=JSON_CLEANUP_SYMBOLS,
                stopwords = SPEAKEASY_TOKEN_STOPWORDS):
        self.tokenizer = WhitespaceTokenizer()
        self.patternCleanup = patternCleanup
        self.stopwords = stopwords
        
        self.specialTokens = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.pad_token_id = self.specialTokens[self.pad_token]
        self.unk_token_id = self.specialTokens[self.unk_token]
        self.mask_token_id = self.specialTokens[self.mask_token]

        self.vocab = None
        self.counter = None

    # methods related to tokenization
    def tokenizeEvent(self, jsonEvent):
        jsonEventClean = self.clearJsonEvent(jsonEvent)
        tokenizedJsonEvent = self.tokenizer.tokenize(jsonEventClean)
        if self.stopwords:
            tokenizedJsonEvent = [x for x in tokenizedJsonEvent if x not in self.stopwords]
        return tokenizedJsonEvent

    def tokenize(self, jsonInput):
        if isinstance(jsonInput, (str, bytes)):
            return self.tokenizeEvent(jsonInput)
        elif isinstance(jsonInput, Iterable):
            return [self.tokenizeEvent(x) for x in jsonInput]
        else:
            raise TypeError("tokenize(): Input must be a string, bytes, or Iterable!")
    
    def clearJsonEvent(self, jsonEvent):
        jsonEvent = str(jsonEvent).lower()
        for x in self.patternCleanup:
            jsonEvent = jsonEvent.replace(x, " ")
        return jsonEvent

    # methods related to vocab
    def buildVocab(self, tokenListSequence, vocabSize=10000):
        counter = Counter()
        for tokenList in tokenListSequence:
            counter.update(tokenList)
        
        idx = len(self.specialTokens)
        vocab = self.specialTokens
        vocab.update({x[0]:i+idx for i,x in enumerate(counter.most_common(vocabSize-idx))})
        self.vocab = vocab
        self.counter = counter
        self.vocabSize = len(self.vocab) if vocabSize > len(self.vocab) else vocabSize
        if vocabSize > len(self.vocab):
            msg = " Provided 'vocabSize' is larger than number of tokens in corpus:"
            msg += f" {vocabSize} > {len(self.vocab)}. "
            msg += f"'vocabSize' is set to {self.vocabSize} to represent tokens in corpus!"
            logging.warning(msg)
    
    def dumpVocab(self, vocabPickleFile):
        with open(vocabPickleFile, "wb") as f:
            pickle.dump(self.vocab, f)

    def loadVocab(self, vocabPickleFile):
        with open(vocabPickleFile, "rb") as f:
            self.vocab = pickle.load(f)
        self.vocabSize = len(self.vocab)

    # methods related to encoding
    def convertTokenListToIds(self, tokenList):
        tokenizedListEncoded = []
        for tokenized in tokenList:
            tokenizedEncoded = self.convertTokensToIds(tokenized)
            tokenizedListEncoded.append(tokenizedEncoded)
        return tokenizedListEncoded

    def convertTokensToIds(self, tokenList):
        if self.vocab:
            return [self.vocab[x] if x in self.vocab else self.vocab["<unk>"] for x in tokenList]
        else:
            raise Exception("Vocabulary not initialized! Use buildVocab() first or load it using loadVocab()!")
    
    def padSequence(self, encodedSequence, maxLen=None):
        self.maxLen = maxLen if maxLen else self.maxLen
        if len(encodedSequence) > self.maxLen:
            return encodedSequence[:self.maxLen]
        else:
            return encodedSequence + [self.pad_token_id] * (self.maxLen - len(encodedSequence))
    
    def padSequenceList(self, encodedSequenceList, maxLen=None):
        self.maxLen = maxLen if maxLen else self.maxLen
        return np.array([self.padSequence(x) for x in encodedSequenceList], dtype=np.int8)

    def encode(self, jsonInput, pad=True, maxLen=512):
        tokenized = self.tokenize(jsonInput)
        if isinstance(tokenized[0], Iterable):
            encoded = self.convertTokenListToIds(tokenized)
        else:
            encoded = [self.convertTokensToIds(tokenized)]
        # apply padding to each element in list
        if pad:
            self.maxLen = maxLen
            return self.padSequenceList(encoded)
        else:
            return encoded


class PEDynamicFeatureExtractor():
    def __init__(self, 
                    speakeasyConfig=None, 
                    speakeasyRecords=SPEAKEASY_RECORDS,
                    recordSubFilter=SPEAKEASY_RECORD_SUBFILTER_MINIMALISTIC,
                    recordLimits=SPEAKEASY_RECORD_LIMITS,
                    returnValues=RETURN_VALUES_TOKEEP
                ):
        
        self.speakeasyConfig = speakeasyConfig
        if self.speakeasyConfig and not os.path.exists(speakeasyConfig):
            raise Exception(f"Speakeasy config file not found: {self.speakeasyConfig}")
        
        self.speakeasyRecords = speakeasyRecords
        self.recordSubFilter = recordSubFilter
        self.recordLimits = recordLimits
        self.returnValues = returnValues

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
        
        # normalize
        recordDict['network_events.traffic'] = normalizeTableIP(recordDict['network_events.traffic'], col='server')
        recordDict['file_access'] = normalizeTablePath(recordDict['file_access'], col='path')
        retValMask = recordDict['apis']['ret_val'].isin(self.returnValues)
        recordDict['apis']['ret_val'][~retValMask] = "<ret_val>"
        
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