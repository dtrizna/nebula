# https://github.com/ucsb-seclab/Neurlux/blob/main/attention_train_all.py#L129

import torch.nn as nn
import torch.nn.functional as F
from .preprocessor import NeurLuxPreprocessor

EMBEDDING_DIM = 256
VOCAB_SIZE = 10000
MAX_LEN = 87000

class Attention(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, num_heads=1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        # x should be (B, L, D)
        x, _ = self.attention(x, x, x)
        x = x.reshape(x.shape[0], -1)
        return x
    

class NeurLuxModel(nn.Module):
    def __init__(self,
                 embedding_dim=EMBEDDING_DIM,
                 vocab_size=VOCAB_SIZE,
                 seq_len=MAX_LEN):
        super(NeurLuxModel, self).__init__()
        self.__name__ = "NeurLux"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=100,
                              kernel_size=4, padding=1)
        self.lstm = nn.LSTM(100, 32, bidirectional=True,
                            batch_first=True)
        self.attention = Attention(embed_dim=64) # LSTM_out * 2
        attention_out = 64*int((seq_len-1)/4)
        self.fc1 = nn.Linear(attention_out, 10)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = F.relu(x)
        # (B, L, MAX_LEN-1)
        x = F.max_pool1d(x, 4)
        # (B, L, (MAX_LEN-1)/4)
        # when MAX_LEN = 2048, (MAX_LEN-1)/4 = 511
        # when MAX_LEN = 87000, (MAX_LEN-1)/4 = 21749
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # (B, L, D)
        # where D == H_out (32) * 2 (bidirectional)
        x = self.attention(x)
        # (B, L*D)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x