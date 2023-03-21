# https://dl.acm.org/doi/10.1145/3560830.3563726
import numpy as np
import logging
import json
from tqdm import tqdm
from os.path import join, exists
from collections import Counter
from .preprocessor import report_to_apiseq, flatten

import torch
from torch import nn
from torch.nn import functional as F

class Core1DConvNet(nn.Module):
    def __init__(self, 
                # embedding params
                vocab_size = 152,
                embedding_dim = 96,
                # conv params
                filter_sizes = [2, 3, 4, 5],
                num_filters = [128, 128, 128, 128],
                batch_norm_conv = False,
                # ffnn params
                hidden_neurons = [1024, 512, 256, 128],
                batch_norm_ffnn = True,
                dropout = 0.5,
                num_classes = 1):
        super().__init__()
        
        # embdding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        # convolutions
        self.conv1d_module = nn.ModuleList()
        for i in range(len(filter_sizes)):
                if batch_norm_conv:
                    module = nn.Sequential(
                                nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=num_filters[i],
                                    kernel_size=filter_sizes[i]),
                                nn.BatchNorm1d(num_filters[i])
                            )
                else:
                    module = nn.Conv1d(in_channels=embedding_dim,
                                    out_channels=num_filters[i],
                                    kernel_size=filter_sizes[i])
                self.conv1d_module.append(module)

        # Fully-connected layers
        conv_out = np.sum(num_filters)
        self.ffnn = []

        for i,h in enumerate(hidden_neurons):
            self.ffnn_block = []
            if i == 0:
                self.ffnn_block.append(nn.Linear(conv_out, h))
            else:
                self.ffnn_block.append(nn.Linear(hidden_neurons[i-1], h))
            # add BatchNorm to every layer except last
            if batch_norm_ffnn:# and not i+1 == len(hidden_neurons):
                self.ffnn_block.append(nn.BatchNorm1d(h))
            self.ffnn_block.append(nn.ReLU())
            if dropout:
                self.ffnn_block.append(nn.Dropout(dropout))
            self.ffnn.append(nn.Sequential(*self.ffnn_block))
        
        self.ffnn = nn.Sequential(*self.ffnn)
        self.fc_output = nn.Linear(hidden_neurons[-1], num_classes)

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])
    
    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x_conv = [self.conv_and_max_pool(embedded, conv1d) for conv1d in self.conv1d_module]
        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        out = self.fc_output(x_fc)
        return out


class QuoVadisModel(Core1DConvNet):
    def __init__(self, vocab=None, seq_len=150, num_classes=1):
        self.seq_len = seq_len
        if vocab is None:
            logging.warning(" [!] Class initialized without vocabulary as preprocessor - use .build_vocab()!")
            super().__init__(num_classes=num_classes)
        else:
            if isinstance(vocab, dict):
                self.vocab = vocab
            elif exists(vocab):
                with open(vocab) as f:
                    self.vocab = json.load(f)
            else:
                raise ValueError(f"vocab: should be either dict or filepath to JSON object, got: {vocab}")
            super().__init__(vocab_size=len(self.vocab), num_classes=num_classes)

    def build_vocab(self, reports, top_api=600, sequences=False):
        if sequences:
            api_sequences = reports
        else:
            api_sequences = []
            for report in tqdm(reports):
                api_sequences.append(report_to_apiseq(report))
        api_counter = Counter(flatten(api_sequences))
        api_calls_preserved = [x[0] for x in api_counter.most_common(top_api-2)] # -2 to account for pad & other
        self.vocab = dict(zip(['<pad>', '<other>'] + api_calls_preserved, range(len(api_calls_preserved)+2)))
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        return api_sequences
    
    def dump_vocab(self, outfolder):
        vocab_file = join(outfolder, f"vocab_{len(self.vocab)}.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f, indent=4)
        logging.warning(f" [!] Vocabulary saved to {vocab_file}.")
        return vocab_file
        
    def predict_apiseq(self, apiseq):
        x = self.apiseq_to_arr(apiseq, self.vocab, self.seq_len)
        x = torch.LongTensor(x.reshape(1,-1)).to(self.device)
        self.eval()
        logits = self(x)
        prediction = torch.argmax(logits, dim=1).flatten()
        return logits, prediction

    def predict_report(self, report):
        apiseq = report_to_apiseq(report)
        logits, preds = self.predict_apiseq(apiseq)
        return logits, preds
    
    def apisequences_to_arr(self, api_sequences):
        return np.array([self.apiseq_to_arr(seq) for seq in api_sequences], dtype=np.int32)

    def apiseq_to_arr(self, rawseq):
        """Function that transforms raw API sequence list to encoded array of fixed length."""
        filtered_seq = self.api_filter(rawseq)
        padded_seq = self.pad_array(filtered_seq, drop="middle")
        return padded_seq

    def pad_array(self, arr, drop="middle"):
        """Function that takes arbitrary length array and returns either padded or truncated array.
        Args:
            arr (np.ndarray): Arbitrary length numpy array.
            length (int, optional): Length of returned array. Defaults to 150.
            how (str): choice ["last", "middle"] - specifies how to perform truncation.
        Returns:
            [np.ndarray]: Fixed length array.
        """
        if arr.shape[0] < self.seq_len:
            # pad with 0 at the end
            return np.pad(arr, [0, self.seq_len-arr.shape[0]], mode="constant", constant_values=self.vocab['<pad>'])
        else:
            if drop == "middle":
                # take only "length" characters, but drop middle ones
                return np.hstack([arr[:int(self.seq_len/2)],arr[-round(self.seq_len/2):]])
            elif drop == "first":
                # take only last "length" characters
                return arr[-self.seq_len:]
            elif drop == "last":
                return arr[:self.seq_len]
            else:
                raise NotImplementedError

    def api_filter(self, rawseq):
        """Function that takes array and preserves only elements in vocabulary."""
        seq = [self.vocab[x] if x in self.vocab.keys() else self.vocab['<other>'] for x in rawseq]
        return np.array(seq, dtype=int)
