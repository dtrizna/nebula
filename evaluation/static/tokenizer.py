import os
import json
import logging
import numpy as np
import sentencepiece as spm
from time import time
from tqdm import tqdm
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class StaticBytesTokenizer:
    def __init__(self,
                vocab_size: int,
                model_path: Optional[str] = None,
                vocab: Optional[Dict[str, int]] = None,
                cleanup_bytes: Optional[List[bytes]] = [b"\x00"],
                character_coverage: float = 0.9995,
                outfolder: Optional[str] = ""
        ):
        self.vocab_size = vocab_size
        self.cleanup_bytes = cleanup_bytes
        self.character_coverage = character_coverage

        self.vocab = None
        self.reverse_vocab = None
        
        self.pad_token_id = 0
        self.pad_token = '<pad>'

        self.unk_token_id = 1
        self.unk_token = '<unk>'

        self.outfolder = outfolder
        os.makedirs(self.outfolder, exist_ok=True)

        if model_path is not None:
            self.tokenizer = spm.SentencePieceProcessor(model_file=model_path.replace(".model","")+".model")
            logging.warning("[!] Successfully loaded pre-trained tokenizer model!")
            self.model_path = model_path
            self.load_vocab(vocab=vocab)
        else:
            self.tokenizer = spm.SentencePieceTrainer
            msg = "[!] Initialized tokenizer without pre-trained model.\n\t"
            msg += "You need to train tokenizer with .train() or specify 'model_path=' during initialization!"
            logging.warning(msg)


    def load_vocab(self, vocab: Optional[Dict[str, int]] = None):
        if isinstance(vocab, dict):
            self.vocab = vocab
            self.reverse_vocab = {v:k for k,v in self.vocab.items()}
            return

        # If None -- trying to load default sentencepiece vocab file
        if vocab is None:
            vocab = self.model_path.replace(".model","")+"_vocab.json"
        if not os.path.exists(vocab): # default sentencepiece -- after training
            vocab = self.model_path.replace(".model", "")+".vocab"
        if not os.path.exists(vocab):
            logging.error(f"[!] Vocab file {vocab} does not exist! .load_vocab() failed!")
            return
        
        with open(vocab, encoding="utf-8") as f:
            if vocab.endswith(".json"):
                self.vocab = json.load(f)
            else: # sentencepiece vocab parsing
                data = f.read()
                vocab = [x.split("\t")[0] for x in data.split("\n")]
                self.vocab = {k:i for i,k in enumerate(vocab)}
        
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        logging.info(f"[!] Loaded vocab with {len(self.vocab)} tokens.")


    def dump_vocab(self):
        vocab_filename = self.model_path.replace(".model","") + "_vocab.json"
        with open(vocab_filename, "w") as f:
            json.dump(self.vocab, f, indent=4)
        logging.info(f"[!] Vocab dumped to {vocab_filename}.")


    def pad_sequence(self, encoded_sequence, seq_len=None):
        if len(encoded_sequence) >= seq_len:
            return encoded_sequence[:seq_len]
        else:
            padded = np.pad(
                encoded_sequence, 
                (0, seq_len - len(encoded_sequence)), 
                mode='constant', 
                constant_values=self.pad_token_id
            )
            return padded


    def pad_sequences(self, encoded_sequence_list, seq_len=None):
        return [self.pad_sequence(seq, seq_len) for seq in tqdm(encoded_sequence_list, desc="Padding")]


    def read_bytez(self, file_path, seqlen=None):
        with open(file_path, 'rb') as file:
            bytez = file.read()

            # remove unnecessary bytes like null byte
            for cleanup_byte in self.cleanup_bytes:
                bytez = bytez.replace(cleanup_byte, b"")

            if seqlen is not None and len(bytez) > seqlen:
                bytez = bytez[:seqlen]
        
        return bytez


    def read_corpus_bytez(self, corpus_filepaths, seqlen=None):
        # NOTE: ugly but fastest, ~1.5k pe files in 1 min
        # tried multi-threaded with ThreadPoolExecutor, but it wasn't faster
        return [self.read_bytez(file_path, seqlen=seqlen) for file_path in tqdm(corpus_filepaths, desc="Reading corpus bytez")]


    def train(
        self,
        corpus_bytez: List[bytes] = None,
        corpus_filepaths: List[str] = None,
        remove_train_files: bool = True,
        # sentencepiece configuration parameters
        spLength: int = 4192,
        split_by_number: bool = True,
    ):
        if corpus_bytez is None and corpus_filepaths is None:
            raise ValueError("[-] Either 'corpus_bytez' or 'corpus_filepaths' must be provided")
        elif corpus_filepaths is not None and corpus_bytez is not None:
            raise ValueError("[-] Only one of 'corpus_bytez' or 'corpus_filepaths' must be provided")
        elif corpus_bytez is None and corpus_filepaths is not None:
            # If corpus_filepaths is provided, use read_bytez to construct corpus_bytez
            logging.warning("[*] Reading corpus bytez from filepaths...")
            corpus_bytez = self.read_corpus_bytez(corpus_filepaths)

        logging.warning("[*] Sentencepiece requires saving data to disk...")
        trainfile = os.path.join(self.outfolder, f"sp_bpe_trainset_{int(time())}.txt")
        with open(trainfile, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus_bytez))
        
        model_prefix = os.path.join(self.outfolder, f"sp_bpe")
        trainOptions = dict(
            input=trainfile,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            max_sentence_length=spLength,
            normalization_rule_name="identity", # no normalization
            # NOTE: before
            # max_sentencepiece_length=64,
            # rainer_interface.cc(124) LOG(WARNING) Too many sentences are loaded! (15893851), which may slow down training.
            # trainer_interface.cc(126) LOG(WARNING) Consider using --input_sentence_size=<size> and --shuffle_input_sentence=true.
            # trainer_interface.cc(129) LOG(WARNING) They allow to randomly sample <size> sentences from the entire corpus.
            # trainer_interface.cc(409) LOG(INFO) Loaded all 15893851 sentences
            # trainer_interface.cc(416) LOG(INFO) Skipped 30083 too long sentences.
            # NOTE: after
            max_sentencepiece_length=16,
            input_sentence_size=5e6, # maximum size of sentences the trainer loads
            shuffle_input_sentence=True,
            # NOTE: other options
            byte_fallback=True, # NOTE: must be True for bytes input
            character_coverage=self.character_coverage,
            remove_extra_whitespaces=False,
            split_digits=True,
            split_by_number=split_by_number,
            bos_id=-1,
            eos_id=-1,
            unk_id=self.unk_token_id,
            pad_id=self.pad_token_id,
            num_threads=os.cpu_count()
        )
        logging.warning(f"[!] Training tokenizer with the following options: {trainOptions}")
        self.tokenizer.Train(**trainOptions)
        logging.warning(f"[!] Tokenizer training completed. Model saved to {model_prefix}.model.")
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")            
        self.model_path = model_prefix
        self.load_vocab() # NOTE: loads default tabulated spm vocab
        self.dump_vocab() # NOTE: to dump proper JSON version

        if remove_train_files:
            os.remove(trainfile) # NOTE: temp file needed for training bcause spm doesn't accept memory object
            os.remove(f"{model_prefix}.vocab") # NOTE: removed default sentencepiece vocab

        return corpus_bytez

    def tokenize(self, inputs: List):
        # if single sample, wrap in list
        if isinstance(inputs, (str, bytes)):
            inputs = [inputs]
        
        return [self.tokenizer.encode_as_pieces(x) for x in inputs]

    def encode(
            self,
            inputs: List,
            pad_len: Optional[int] = None,
            tokenize: bool = True
    ):
        if not tokenize: # NOTE: for readability and informative purposes
            raise NotImplementedError("SentencePiece tokenizer does not support encode without tokenize!")

        # if single sample, wrap in list
        if isinstance(inputs, (str, bytes)):
            inputs = [inputs]

        encoded = [self.tokenizer.encode_as_ids(x) for x in tqdm(inputs, desc="Encoding")]

        if pad_len is not None:
            encoded = self.pad_sequences(encoded, seq_len=pad_len)

        return np.array(encoded, dtype=np.int32)
