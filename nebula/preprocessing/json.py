import os
# TODO: think about more secure alternative to pickle
import pickle
import logging
import json
import operator
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from functools import reduce
from typing import Iterable
from collections import Counter, defaultdict
from pandas import json_normalize, concat, DataFrame, merge

from nltk import WhitespaceTokenizer
import sentencepiece as spm

from nebula.constants import *
from nebula.misc import get_alphanum_chars


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


class JSONTokenizer:
    def __init__(self, 
                sequenceLength = 2048,
                patternCleanup = JSON_CLEANUP_SYMBOLS,
                stopwords = SPEAKEASY_TOKEN_STOPWORDS,
                specialTokens = ["<unk>", "<pad>", "<mask>"]):
        self.sequenceLength = sequenceLength
        self.patternCleanup = patternCleanup
        self.stopwords = stopwords
        self.tokenizer = WhitespaceTokenizer()
        
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

    def dumpTokenizerFiles(self, outFolder):
        file = f"{outFolder}\\speakeasy_VocabSize_{self.vocabSize}.pkl"
        self.dumpVocab(file)
        logging.warning("Dumped vocab to {}".format(file))
        
        file = f"{outFolder}\\speakeasy_counter.pkl"
        logging.warning("Dumped vocab counter to {}".format(file))
        with open(file, "wb") as f:
            pickle.dump(self.counter, f)

        file = f"{outFolder}\\speakeasy_counter_plot.png"
        logging.warning("Dumped vocab counter plot to {}".format(file))


class JSONTokenizerBPE:
    def __init__(self,
                model_path=None,
                patternCleanup=JSON_CLEANUP_SYMBOLS,
                stopwords=SPEAKEASY_TOKEN_STOPWORDS,
                specialTokens = ["<unk>", "<pad>", "<mask>"]):
        self.patternCleanup = patternCleanup
        self.stopwords = stopwords

        self.specialTokens = dict(zip(specialTokens, range(len(specialTokens))))
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.pad_token_id = self.specialTokens[self.pad_token]
        self.unk_token_id = self.specialTokens[self.unk_token]
        self.mask_token_id = self.specialTokens[self.mask_token]

        self.vocab = None
        self.reverse_vocab = None
        if model_path:
            self.tokenizer = spm.SentencePieceProcessor(model_file=model_path.rstrip(".model")+".model")
            logging.warning(" [!] Successfully loaded pre-trained tokenizer model!")
            self.model_path = model_path
            self.load_vocab()
        else:
            self.tokenizer = spm.SentencePieceTrainer
            logging.warning(" [!] You need to train tokenizer with .train() or specify 'model_path=' during initialization!")
    
    def split_string_to_chunks(self, s, chunkSize=4192):
        """This function should split a long string into smaller chunks of size chunkSize, 
        but it shouldn't split the string in the middle of a word.

        Args:
            s (str): Longstring
            chunkSize (int, optional): _description_. Defaults to 512.

        Returns:
            list: List of smaller strings
        """
        chunks = []
        words = s.split(" ")
        currentChunk = ""
        for word in words:
            if len(currentChunk) + len(word) < chunkSize:
                currentChunk += word + " "
            else:
                chunks.append(currentChunk)
                currentChunk = word + " "
        chunks.append(currentChunk)
        return chunks

    def clear_json_event(self, jsonData):
        """
        Removes all special characters from the json event.
        """
        assert isinstance(jsonData, (str, bytes, list, dict))
        jsonData = str(jsonData).lower()
        for pattern in self.patternCleanup:
            jsonData = jsonData.replace(pattern, " ")
        jsonData = [get_alphanum_chars(x) for x in jsonData.split(" ") if x not in self.stopwords]
        return ' '.join(jsonData)

    def load_vocab(self, vocabPath=None):
        if not vocabPath:
            vocabPath = self.model_path.rstrip(".model")+"_vocab.json"
            if not os.path.exists(vocabPath): # default sentencepiece -- after training
                vocabPath = self.model_path.rstrip(".model")+".vocab"
        with open(vocabPath, encoding="utf-8") as f:
            if vocabPath.endswith(".json"):
                self.vocab = json.load(f)
            else:
                data = f.read()
                vocab = [x.split("\t")[0] for x in data.split("\n")]
                self.vocab = {k:i for i,k in enumerate(vocab)}
        self.vocab.update(self.specialTokens)
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        logging.warning(f" [!] Loaded vocab with size {len(self.vocab)} from {vocabPath}")
        
    def dump_vocab(self):
        vocabFileName = self.model_path.rstrip(".model") + "_vocab.json"
        with open(vocabFileName, "w") as f:
            json.dump(self.vocab, f, indent=4)

    def train(self, jsonData, model_prefix="tokenizer", vocab_size=1024, model_type="bpe", split_by_number=False, spLength=4192, removeTrainFiles=True):
        """
        Trains the tokenizer on the given json data.
        """
        jsonDataClean = self.clear_json_event(jsonData)
        # splitting a string into chunks of 4192 characters since this sentencepiece limitation
        jsonDataChunks = self.split_string_to_chunks(jsonDataClean, chunkSize=spLength)
        # dump jsonDataClean to file
        trainFile = f"{model_prefix}_trainset_{int(time())}.txt"
        with open(trainFile, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonDataChunks))

        trainCmd = " ".join([
            f"--input={trainFile}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            f"--split_by_number={split_by_number}",
            f"--max_sentence_length={spLength}",
            f"--max_sentencepiece_length=64"
        ])
        print(f"Training tokenizer with command: {trainCmd}")
        self.tokenizer.Train(trainCmd)
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        
        self.model_path = model_prefix
        self.load_vocab()

        if removeTrainFiles:
            os.remove(trainFile)
            os.remove(f"{model_prefix}.vocab")
    
    def tokenize(self, jsonData):
        """
        Tokenizes the given json data.
        """
        if isinstance(jsonData, (str, bytes, dict)):
            jsonData = [jsonData]
        jsonDataClean = [self.clear_json_event(x) for x in jsonData]
        return [self.tokenizer.encode_as_pieces(x) for x in jsonDataClean]

    def encode(self, jsonData):
        """
        Encodes the given json data.
        """
        if isinstance(jsonData, (str, bytes, dict)):
            jsonData = [jsonData]
        jsonDataClean = [self.clear_json_event(x) for x in jsonData]
        return [self.tokenizer.encode_as_ids(x) for x in jsonDataClean]

    def pad_sequence(self, encodedSequence):
        if len(encodedSequence) > self.sequenceLength:
            return encodedSequence[:self.sequenceLength]
        else:
            return encodedSequence + [self.pad_token_id] * (self.sequenceLength - len(encodedSequence))
    
    def pad_sequence_list(self, encodedSequenceList, sequenceLength=512):
        self.sequenceLength = sequenceLength
        return np.array([self.pad_sequence(x) for x in encodedSequenceList], dtype=np.int32)

    def pad_sequences(self, encodedSequences, sequenceLength=512):
        return self.pad_sequence_list(encodedSequences, sequenceLength=sequenceLength)

