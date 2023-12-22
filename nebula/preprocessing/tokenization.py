import os
import logging
import json
import operator
import numpy as np
from tqdm import tqdm
from time import time
from functools import reduce
from typing import Iterable
from collections import Counter, defaultdict
from pandas import json_normalize, concat, DataFrame, merge

from nltk import WhitespaceTokenizer, WordPunctTokenizer
import sentencepiece as spm

from nebula.constants import *
import string


class JSONFilter:
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
                 seq_len,
                 vocab_size,
                 cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                 stopwords=SPEAKEASY_TOKEN_STOPWORDS,
                 special_tokens = ["<pad>", "<unk>", "<mask>"]
        ):
        assert isinstance(seq_len, int), "seq_len must be integer!"
        self.seq_len = seq_len
        assert isinstance(vocab_size, int), "vocab_size must be integer!"
        self.vocab_size = vocab_size
        assert isinstance(cleanup_symbols, (list, tuple, None)), "cleanup_symbols must be list or tuple!"
        self.cleanup_symbols = cleanup_symbols
        assert isinstance(stopwords, (list, tuple, None)), "stopwords must be list or tuple!"
        self.stopwords = stopwords
        
        self.special_tokens = dict(zip(special_tokens, range(len(special_tokens))))
        assert len(self.special_tokens) >= 3, "special_tokens must contain at least 3 tokens for pad, unk, and mask!"
        self.pad_token = special_tokens[0]
        self.unk_token = special_tokens[1]
        self.mask_token = special_tokens[2]
        self.pad_token_id = self.special_tokens[self.pad_token]
        self.unk_token_id = self.special_tokens[self.unk_token]
        self.mask_token_id = self.special_tokens[self.mask_token]

        self.vocab = None
        self.reverse_vocab = None
        
    def clear_json_event(self, text):
        """
        Removes all special characters from the json event.
        """
        assert isinstance(text, (str, bytes, list, dict))
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        text = ' '.join(text) if isinstance(text, list) and isinstance(text[0], str) else text
        text = str(text).lower()
        if self.cleanup_symbols:
            for pattern in self.cleanup_symbols:
                text = text.replace(pattern, ' ')
        if self.stopwords:
            for pattern in self.stopwords:
                text = text.replace(pattern, '')
        # all other punctuation -- replace with space
        text = text.translate(
                str.maketrans(
                    string.punctuation,
                    " " * len(string.punctuation)
                )
            )
        return text

    def pad_sequence(self, encoded_sequence, seq_len=None):
        if seq_len:
            assert isinstance(seq_len, int), "seq_len must be integer!"
            self.seq_len = seq_len
        if len(encoded_sequence) >= self.seq_len:
            return encoded_sequence[:self.seq_len]
        else:
            padded = np.pad(
                encoded_sequence, 
                (0, self.seq_len - len(encoded_sequence)), 
                mode='constant', 
                constant_values=self.pad_token_id
            )
            return padded
    
    def pad_sequence_list(self, encoded_sequence_list, seq_len=None):
        return np.array([self.pad_sequence(x, seq_len) for x in encoded_sequence_list], dtype=np.int32)

    def pad_sequences(self, encoded_sequences, seq_len=None):
        return self.pad_sequence_list(encoded_sequences, seq_len=seq_len)


class JSONTokenizerNaive(JSONTokenizer):
    def __init__(self, 
                vocab_size,
                seq_len,
                vocab=None,
                cleanup_symbols = JSON_CLEANUP_SYMBOLS,
                stopwords = SPEAKEASY_TOKEN_STOPWORDS,
                type = "whitespace",
                counter_dump=False
                ):
        super().__init__(
            seq_len,
            vocab_size,
            cleanup_symbols,
            stopwords
        )
        assert type in ["whitespace", "wordpunct"], "type must be either 'whitespace' or 'wordpiece'!"
        if type == "whitespace":
            self.tokenizer = WhitespaceTokenizer()
        elif type == "wordpunct":
            self.tokenizer = WordPunctTokenizer()
        self.counter = None
        self.counter_dump = counter_dump
        self.vocab_error = "Vocabulary not initialized! Use build_vocab() first or load it using load_vocab()!"
        if vocab is not None:
            self.load_vocab(vocab)
        
    def tokenize_event(self, jsonEvent):
        jsonEventClean = self.clear_json_event(jsonEvent)
        tokenizedJsonEvent = self.tokenizer.tokenize(jsonEventClean)
        # if self.stopwords:
        #     tokenizedJsonEvent = [x for x in tokenizedJsonEvent if x not in self.stopwords]
        return tokenizedJsonEvent

    def tokenize(self, sample):
        if isinstance(sample, dict):
            return self.tokenize_event(str(sample))
        elif isinstance(sample, (str, bytes)):
            return self.tokenize_event(sample)
        elif isinstance(sample, Iterable):
            return [self.tokenize_event(str(x)) for x in sample]
        else:
            raise TypeError("tokenize(): Input must be a string, bytes, or Iterable!")
    
    def build_vocab(self, corpus, vocab_size=None, model_prefix="whitespace", counter_dump=False):
        """Builds the vocabulary from the corpus and preserve the
         top vocabSize tokens based on appearance counts."""
        if vocab_size:
            self.vocab_size = vocab_size

        self.counter = Counter()
        for text in tqdm(corpus):
            text = self.clear_json_event(text)
            tokens = self.tokenizer.tokenize(text)
            self.counter.update(tokens)
        
        self.vocab = self.counter.most_common(self.vocab_size-len(self.special_tokens))
        self.vocab = [token for token, _ in self.vocab]
        self.vocab = list(self.special_tokens.keys()) + self.vocab
        self.vocab = {token: index for index, token in enumerate(self.vocab)}
        self.reverse_vocab = {index: token for token, index in self.vocab.items()}
        self.dump_vocab(model_prefix)
        if counter_dump or self.counter_dump:
            self.dump_counter(model_prefix)
    
    def train(self, tokenizedListSequence, vocab_size=None, model_prefix="", counter_dump=False):
        self.build_vocab(tokenizedListSequence, vocab_size, model_prefix, counter_dump)

    def dump_vocab(self, vocab_prefix="whitespace"):
        with open(vocab_prefix+f"_vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=4)
        logging.warning("Dumped vocab to {}".format(vocab_prefix+f"_vocab.json"))
    
    def dump_counter(self, prefix):
        file = f"{prefix}_counter.json"
        with open(file, "w") as f:
            json.dump(self.counter, f, indent=4)
        logging.warning(f"Dumped vocab counter to {file}")

    def load_vocab(self, vocab):
        if isinstance(vocab, dict):
            self.vocab = vocab
        else:
            with open(vocab) as f:
                self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}

    def load_from_pretrained(self, vocab):
        self.load_vocab(vocab)

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
            raise Exception("convertTokensToIds(): " + self.vocab_error)
    
    def encode(self, inputs, pad=True, tokenize=True):
        if isinstance(inputs, (str, bytes, dict)) or \
            (isinstance(inputs, list) and isinstance(inputs[0], (str, bytes))):
            inputs = [inputs]
        if tokenize:
            inputs = self.tokenize(inputs)
        if isinstance(inputs[0], str):
            # means we got a single example for encoding
            encoded = [self.convertTokensToIds(inputs)]
        else:
            encoded = self.convertTokenListToIds(inputs)
        # apply padding to each element in list
        if pad:
            return self.pad_sequence_list(encoded)
        else:
            return encoded
    
    def decode(self, encodedSequence):
        if self.vocab and self.reverse_vocab:
            decodedSequence = []
            for x in encodedSequence:
                if x == self.pad_token_id:
                    break
                elif x in self.reverse_vocab:
                    decodedSequence.append(self.reverse_vocab[x])
                else:
                    decodedSequence.append(self.unk_token)
            return decodedSequence
        else:
            raise Exception("detokenize(): " + self.vocab_error)


class JSONTokenizerBPE(JSONTokenizer):
    def __init__(self,
                vocab_size,
                seq_len,
                model_path=None,
                vocab=None,
                cleanup_symbols=JSON_CLEANUP_SYMBOLS,
                stopwords=SPEAKEASY_TOKEN_STOPWORDS,
        ):
        super().__init__(
            seq_len,
            vocab_size,
            cleanup_symbols,
            stopwords
        )

        if model_path is not None:
            self.tokenizer = spm.SentencePieceProcessor(model_file=model_path.replace(".model","")+".model")
            logging.info(" [!] Successfully loaded pre-trained tokenizer model!")
            self.model_path = model_path
            self.load_vocab(vocab=vocab)
        else:
            self.tokenizer = spm.SentencePieceTrainer
            msg = " [!] Initialized tokenizer without pre-trained model.\n\t"
            msg += "You need to train tokenizer with .train() or specify 'model_path=' during initialization!"
            logging.warning(msg)
    
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

    def load_vocab(self, vocab=None):
        if isinstance(vocab, dict):
            self.vocab = vocab
            self.reverse_vocab = {v:k for k,v in self.vocab.items()}
            return

        # parsing default sentencepiece vocab file
        if vocab is None:
            vocab = self.model_path.replace(".model","")+"_vocab.json"
        if not os.path.exists(vocab): # default sentencepiece -- after training
            vocab = self.model_path.replace(".model", "")+".vocab"
        if not os.path.exists(vocab):
            logging.error(f" [!] Vocab file {vocab} does not exist! .load_vocab() failed!")
            return

        with open(vocab, encoding="utf-8") as f:
            if vocab.endswith(".json"):
                self.vocab = json.load(f)
            else:
                data = f.read()
                vocab = [x.split("\t")[0] for x in data.split("\n")]
                self.vocab = {k:i for i,k in enumerate(vocab)}
        # update vocab with special tokens, but ensure that they are unique & at correct locations
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        for k,v in self.special_tokens.items():
            keys[v] = k
        
        self.vocab = dict(zip(keys, values))
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        logging.info(f" [!] Loaded vocab from {vocab}")
        
    def dump_vocab(self):
        vocabFileName = self.model_path.replace(".model","") + "_vocab.json"
        with open(vocabFileName, "w") as f:
            json.dump(self.vocab, f, indent=4)

    def train(
        self,
        jsonData,
        vocab_size=None,
        model_prefix="bpe",
        model_type="bpe",
        split_by_number=False,
        spLength=4192,
        removeTrainFiles=True
    ):
        """
        Trains the tokenizer on the given json data.
        """
        logging.warning(" [*] Data preparation for SentencePiece tokenizer...")
        jsonDataClean = self.clear_json_event(jsonData)
        # splitting a string into chunks of 4192 characters since this sentencepiece limitation
        jsonDataChunks = self.split_string_to_chunks(jsonDataClean.replace("\\\\", "\\"), chunkSize=spLength)
        # dump jsonDataClean to file
        logging.warning(" [*] Saving to disk...")
        trainFile = f"{model_prefix}_trainset_{int(time())}.txt"
        with open(trainFile, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonDataChunks))

        if vocab_size:
            self.vocab_size = vocab_size
        
        trainCmd = " ".join([
            f"--input={trainFile}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={self.vocab_size}",
            f"--model_type={model_type}",
            f"--split_by_number={split_by_number}",
            f"--max_sentence_length={spLength}",
            f"--max_sentencepiece_length=64"
        ])
        logging.warning(f" [!] Training tokenizer with command: {trainCmd}")
        self.tokenizer.Train(trainCmd)
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        
        self.model_path = model_prefix
        self.load_vocab()
        self.dump_vocab()

        if removeTrainFiles:
            os.remove(trainFile)
            os.remove(f"{model_prefix}.vocab")
    
    def tokenize(self, inputs):
        """
        Tokenizes the given json data.
        """
        if isinstance(inputs, (str, bytes, dict)) or \
            (isinstance(inputs, list) and isinstance(inputs[0], (str, bytes))):
            inputs = [inputs]
        data_clean = [self.clear_json_event(x) for x in inputs]
        return [self.tokenizer.encode_as_pieces(x) for x in data_clean]

    def encode(self, inputs, pad=True, tokenize=True):
        if not tokenize:
            raise NotImplementedError("SentencePiece tokenizer does not support encode without tokenize!")

        # if single sample, wrap in list
        if isinstance(inputs, (str, bytes, dict)) or \
            (isinstance(inputs, list) and isinstance(inputs[0], (str, bytes))):
            inputs = [inputs]

        data_clean = [self.clear_json_event(x) for x in inputs]
        encoded = [self.tokenizer.encode_as_ids(x) for x in data_clean]
        if pad:
            return self.pad_sequence_list(encoded)
        else:
            return encoded
