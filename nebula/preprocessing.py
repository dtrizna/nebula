import os
# TODO: think about more secure alternative to pickle
import pickle
import logging
import json
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path
from copy import deepcopy
from collections import Counter
from typing import Iterable
from nltk import WhitespaceTokenizer
from pandas import json_normalize, concat, DataFrame

from nebula.constants import *
from nebula.misc import getAlphaNumChars
from nebula.plots import plotCounterCountsLineplot, plotListElementLengths
from nebula.normalization import normalizeTableIP, normalizeTablePath
from nebula.ember import PEFeatureExtractor

import speakeasy

class JSONTokenizer(object):
    def __init__(self, 
                sequenceLength = 2048,
                patternCleanup = JSON_CLEANUP_SYMBOLS,
                stopwords = SPEAKEASY_TOKEN_STOPWORDS,
                specialTokens = ["<pad>", "<unk>", "<mask>"]):
        self.tokenizer = WhitespaceTokenizer()
        self.patternCleanup = patternCleanup
        self.stopwords = stopwords
        
        self.specialTokens = dict(zip(specialTokens, range(len(specialTokens))))
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.pad_token_id = self.specialTokens[self.pad_token]
        self.unk_token_id = self.specialTokens[self.unk_token]
        self.mask_token_id = self.specialTokens[self.mask_token]

        self.counter = None
        self.vocab = None
        self.reverseVocab = None
        self.vocabError = "Vocabulary not initialized! Use buildVocab() first or load it using loadVocab()!"

        self.sequenceLength = sequenceLength

    # methods related to tokenization
    def clearJsonEvent(self, jsonEvent):
        jsonEvent = str(jsonEvent).lower()
        for x in self.patternCleanup:
            jsonEvent = jsonEvent.replace(x, " ")
        return jsonEvent

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
    
    # methods related to vocab
    def buildVocab(self, tokenListSequence, vocabSize=2500):
        counter = Counter()
        for tokenList in tqdm(tokenListSequence):
            counter.update(tokenList)
        
        idx = len(self.specialTokens)
        vocab = deepcopy(self.specialTokens)
        vocab.update({x[0]:i+idx for i,x in enumerate(counter.most_common(vocabSize-idx))})

        self.vocab = vocab
        self.counter = counter
        self.vocabSize = len(self.vocab) if vocabSize > len(self.vocab) else vocabSize
        self.reverseVocab = {v:k for k,v in self.vocab.items()}
        if vocabSize > len(self.vocab):
            msg = " Provided 'vocabSize' is larger than number of tokens in corpus:"
            msg += f" {vocabSize} > {len(self.vocab)}. "
            msg += f"'vocabSize' is set to {self.vocabSize} to represent tokens in corpus!"
            logging.warning(msg)
    
    def train(self, tokenizedListSequence, vocabSize=2500):
        self.buildVocab(tokenizedListSequence, vocabSize)

    def dumpVocab(self, vocabPickleFile):
        with open(vocabPickleFile, "wb") as f:
            pickle.dump(self.vocab, f)

    def loadVocab(self, vocab):
        if isinstance(vocab, dict):
            self.vocab = vocab
        else:
            with open(vocab, "rb") as f:
                self.vocab = pickle.load(f)
        self.vocabSize = len(self.vocab)
        self.reverseVocab = {v:k for k,v in self.vocab.items()}

    def load_from_pretrained(self, vocab):
        self.loadVocab(vocab)

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
            raise Exception("convertTokensToIds(): " + self.vocabError)
    
    def padSequence(self, encodedSequence):
        if len(encodedSequence) > self.sequenceLength:
            return encodedSequence[:self.sequenceLength]
        else:
            return encodedSequence + [self.pad_token_id] * (self.sequenceLength - len(encodedSequence))
    
    def padSequenceList(self, encodedSequenceList):
        return np.array([self.padSequence(x) for x in encodedSequenceList], dtype=np.int32)

    def encode(self, jsonInput, pad=True):
        tokenized = self.tokenize(jsonInput)
        if isinstance(tokenized[0], str):
            # means we got a single example for encoding
            encoded = [self.convertTokensToIds(tokenized)]
        else:
            encoded = self.convertTokenListToIds(tokenized)
        # apply padding to each element in list
        if pad:
            return self.padSequenceList(encoded)
        else:
            return encoded
    
    def decode(self, encodedSequence):
        if self.vocab and self.reverseVocab:
            decodedSequence = []
            for x in encodedSequence:
                if x == self.pad_token_id:
                    break
                elif x in self.reverseVocab:
                    decodedSequence.append(self.reverseVocab[x])
                else:
                    decodedSequence.append(self.unk_token)
            return decodedSequence
        else:
            raise Exception("detokenize(): " + self.vocabError)

    def dumpTokenizerFiles(self, outFolder, tokenListSequence=None):
        file = f"{outFolder}\\speakeasy_VocabSize_{self.vocabSize}.pkl"
        self.dumpVocab(file)
        logging.warning("Dumped vocab to {}".format(file))
        
        file = f"{outFolder}\\speakeasy_counter.pkl"
        logging.warning("Dumped vocab counter to {}".format(file))
        with open(file, "wb") as f:
            pickle.dump(self.counter, f)

        file = f"{outFolder}\\speakeasy_counter_plot.png"
        plotCounterCountsLineplot(self.counter, outfile=file)
        logging.warning("Dumped vocab counter plot to {}".format(file))

        if tokenListSequence:
            _ = plotListElementLengths(tokenListSequence, outfile=f"{outFolder}\\speakeasy_tokenListLengths.png")


class PEStaticFeatureExtractor(object):
    def __init__(self):
        self.extractor = PEFeatureExtractor(print_feature_warning=False)
        
    def feature_vector(self, bytez):
        return self.extractor.feature_vector(bytez)


class PEDynamicFeatureExtractor(object):
    def __init__(self, 
                    speakeasyConfig=None, 
                    speakeasyRecords=SPEAKEASY_RECORDS,
                    recordSubFilter=SPEAKEASY_RECORD_SUBFILTER_OPTIMAL,
                    recordLimits=SPEAKEASY_RECORD_LIMITS,
                    emulationOutputFolder=None
                ):
        
        if isinstance(speakeasyConfig, str):
            if not os.path.exists(speakeasyConfig):
                raise Exception(f"Speakeasy config file not found: {speakeasyConfig}")
            with open(speakeasyConfig, "r") as f:
                self.speakeasyConfig = json.load(f)
        elif isinstance(speakeasyConfig, dict):
            self.speakeasyConfig = speakeasyConfig
        else:
            self.speakeasyConfig = None
        
        self.speakeasyRecords = speakeasyRecords
        self.recordSubFilter = recordSubFilter
        self.recordLimits = recordLimits

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
                
        if path is None and data is None:
            raise ValueError("Either 'file' or 'data' must be specified.")
        if path:
            if not os.path.exists(path):
                raise ValueError(f"File {path} does not exist.")
            self.sampleName = os.path.basename(path).replace(".exe", "")
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
        if api_seq_len == 0:
            return None
        else:
            return self.parseReportEntryPoints(report["entry_points"])
    
    def parseReportEntryPoints(self, entryPoints):
        # clean up report
        recordDict = self.getRecordsFromReport(entryPoints)
        
        # filter out events with uninformative API sequences
        # i.e. emulation failed extract valuable info
        if 'apis' in self.speakeasyRecords and \
            recordDict['apis'].shape[0] == 1 and \
            recordDict['apis'].iloc[0].api_name == 'MSVBVM60.ordinal_100':
                return None

        # normalize
        if 'network_events.traffic' in self.speakeasyRecords:
            recordDict['network_events.traffic'] = normalizeTableIP(recordDict['network_events.traffic'], col='server')
        if 'file_access' in self.speakeasyRecords:
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
