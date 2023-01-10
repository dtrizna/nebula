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
from collections import Counter, defaultdict
from functools import reduce
import operator
from typing import Iterable
from nltk import WhitespaceTokenizer
from pandas import json_normalize, concat, DataFrame, merge

import nebula
from nebula.constants import *
from nebula.misc import getAlphaNumChars
from nebula.plots import plotCounterCountsLineplot, plotListElementLengths
from nebula.normalization import normalizeTableIP, normalizeTablePath
from nebula.ember import PEFeatureExtractor

import speakeasy

class JSONParser:
    def __init__(self,
                fields,
                normalized=False):
        super().__init__()
        self.fields = fields
        assert isinstance(normalized, bool), "normalized must be boolean!"
        self.normalized = normalized

    @staticmethod
    def filterNonNormalizedField(jsonEvent, field):
        keysLeft = field.split(".")
        keysIterated = []

        currentVal = jsonEvent
        for i in range(len(keysLeft)):
            keysIterated.append(keysLeft.pop(0))
            
            # get value from dictionary based on keysIterated
            if keysIterated[-1] not in currentVal:
                return None, keysIterated
            
            currentVal = reduce(operator.getitem, keysIterated, jsonEvent)
            if currentVal == []:
                return None, keysIterated
            
            elif isinstance(currentVal, dict):
                continue # iterate deeper
            elif isinstance(currentVal, list):
                table = json_normalize(jsonEvent, record_path=keysIterated)[keysLeft]
                return table, keysIterated
            else: # currentVal is not a collection, just value
                return currentVal, keysIterated

    def filterNonNormalizedEvent(self, jsonEvent):
        values = defaultdict(list)
        for field in self.fields:
            filteredValue, key = self.filterNonNormalizedField(jsonEvent, field)
            if filteredValue is not None:
                values['.'.join(key)].append(filteredValue)
        
        # merge tables into a single dataframe
        for key, valueList in values.items():
            if all([isinstance(x, DataFrame) for x in valueList]):
                values[key] = reduce(
                    lambda x,y: merge(x,y, left_index=True, right_index=True), 
                    valueList
                )
        return values

    def filterNormalizedEvent(self, jsonEvent):        
        table = json_normalize(jsonEvent)
        # preserve fields that are only table columns 
        cols = table.columns[table.columns.isin(self.fields)]
        table = table[cols]
        return table

    def filter(self, jsonEvents):
        if isinstance(jsonEvents, str):
            jsonEvents = json.loads(jsonEvents)
        assert isinstance(jsonEvents, (list, dict)), "jsonEvent must be list or dict!"
        if isinstance(jsonEvents, dict):
            jsonEvents = [jsonEvents]
        assert [isinstance(x, dict) for x in jsonEvents], "jsonEvent must be list of dicts!"

        if self.normalized:
            filteredEvents = [self.filterNormalizedEvent(x) for x in jsonEvents]
            return filteredEvents # list of Dataframes
        else:
            filteredEvents = [self.filterNonNormalizedEvent(x) for x in jsonEvents]
            return filteredEvents # list of dicts

    def filter_and_concat(self, jsonEvents):
        filteredEvents = self.filter(jsonEvents)
        recordDict = defaultdict(DataFrame)
        for tableDict in filteredEvents:
            for key in tableDict:
                recordDict[key] = concat([recordDict[key], tableDict[key]], axis=0, ignore_index=True)
        return recordDict


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
                    speakeasyRecordFields=SPEAKEASY_RECORD_FIELDS,
                    recordLimits=SPEAKEASY_RECORD_LIMITS,
                    emulationOutputFolder=None
                ):
        
        # setup speakseasy config
        if speakeasyConfig is None:
            speakeasyConfig = os.path.join(os.path.dirname(nebula.__file__), "configs", "speakeasyConfig.json")
        if isinstance(speakeasyConfig, dict):
            self.speakeasyConfig = speakeasyConfig
        else:
            assert os.path.exists(speakeasyConfig), f"Speakeasy config file not found: {speakeasyConfig}"
            with open(speakeasyConfig, "r") as f:
                self.speakeasyConfig = json.load(f)
        
        self.recordLimits = recordLimits
        self.parser = JSONParser(fields=speakeasyRecordFields)

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
            recordDict['apis']['args'] = recordDict['apis']['args'].apply(lambda x: [getAlphaNumChars(y) for y in x])
    
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
