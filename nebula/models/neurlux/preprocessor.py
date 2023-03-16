from nltk import WhitespaceTokenizer
from collections import Counter
import string
import json
from os.path import join, exists
import numpy as np
from tqdm import tqdm

class NeurLuxPreprocessor:
    """Preprocesses text for NeurLux.
    Based on https://github.com/ucsb-seclab/Neurlux/blob/main/attention_train_all.py#L113
    They use `tf.keras.preprocessing.text.Tokenizer`. This class replicates:
    https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization"""

    def __init__(self,
                 vocab_size=10000,
                 max_length=512):
        """Initializes a NeurLuxPreprocessor object."""
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = WhitespaceTokenizer()
        self.vocab = None
        self.reverse_vocab = None
    
    def lowercase_and_clear_punctuation(self, text):
        """Lowercase and clear punctuation from text."""
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        elif not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def train(self, corpus):
        self.build_vocab(corpus)
    
    def build_vocab(self, corpus):
        """Builds the vocabulary from the corpus and preserve the
         top vocabSize tokens based on appearance counts."""
        self.vocab = Counter()
        for text in tqdm(corpus):
            text = self.lowercase_and_clear_punctuation(text)
            tokens = self.tokenizer.tokenize(text)
            self.vocab.update(tokens)
        # -2 to account for <unk> and <pad> tokens
        self.vocab = self.vocab.most_common(self.vocab_size-2)
        self.vocab = [token for token, _ in self.vocab]
        self.vocab = ['<unk>', '<pad>'] + self.vocab
        self.vocab = {token: index for index, token in enumerate(self.vocab)}
        self.reverse_vocab = {index: token for token, index in self.vocab.items()}
    
    def dump_vocab(self, outfolder):
        vocab_file = join(outfolder, f"vocab_{self.vocab_size}.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f, indent=4)
        return vocab_file
    
    def load_vocab(self, vocab):
        if isinstance(vocab, dict):
            self.vocab = vocab
        elif exists(vocab):
            with open(vocab) as f:
                self.vocab = json.load(f)
        else:
            raise ValueError("Vocabulary must be a dictionary or a path to a JSON file.")

    def tokenize(self, text):
        """Preprocesses text for NeurLux."""
        assert self.vocab is not None, "Vocabulary not built yet. Use `train` on corpus of training data."
        text = self.lowercase_and_clear_punctuation(text)
        tokens = self.tokenizer.tokenize(text)
        tokens = [token if token in self.vocab else '<unk>' for token in tokens]
        return tokens

    def encode(self, tokenized):
        """Encodes tokenized text."""
        return [self.vocab[token] for token in tokenized]

    def preprocess(self, text):
        """Preprocesses text for NeurLux."""
        tokenized = self.tokenize(text)
        encoded = self.encode(tokenized)
        return encoded

    def preprocess_sequence(self, sequence):
        """Preprocesses a sequence of text for NeurLux."""
        return [self.preprocess(text) for text in sequence]
    
    def decode(self, encoded):
        """Decodes encoded text."""
        decoded = [self.reverse_vocab[index] for index in encoded]
        return " ".join(decoded)
    
    def pad(self, encoded):
        """Pads encoded text to length."""
        return encoded[:self.max_length] + [self.vocab['<pad>']] * (self.max_length - len(encoded))

    def pad_sequence(self, sequence):
        """Pads a sequence of encoded text to length."""
        padded = [self.pad(encoded) for encoded in sequence]
        return np.vstack(padded).astype(np.int32)
