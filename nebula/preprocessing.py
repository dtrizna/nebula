# TODO: think about more secure alternative to pickle
import pickle
import logging
import numpy as np
from copy import deepcopy
from collections import Counter
from collections.abc import Iterable
from nltk import WhitespaceTokenizer
from tqdm import tqdm
from nebula.constants import *


class JSONTokenizer(object):
    def __init__(self, 
                patternCleanup=JSON_CLEANUP_SYMBOLS,
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
    
    def padSequence(self, encodedSequence, maxLen=None):
        self.maxLen = maxLen if maxLen else self.maxLen
        if len(encodedSequence) > self.maxLen:
            return encodedSequence[:self.maxLen]
        else:
            return encodedSequence + [self.pad_token_id] * (self.maxLen - len(encodedSequence))
    
    def padSequenceList(self, encodedSequenceList, maxLen=None):
        self.maxLen = maxLen if maxLen else self.maxLen
        return np.array([self.padSequence(x) for x in encodedSequenceList], dtype=np.int32)

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
