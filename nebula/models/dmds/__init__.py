# from: https://arxiv.org/pdf/1907.07352.pdf
# https://github.com/joddiy/DynamicMalwareAnalysis
# code asked privately, and converted from Keras to PyTorch by us
import torch
import torch.nn as nn

from functools import reduce
from operator import __add__

class Conv2dSamePadding(nn.Conv1d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
                  [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class DMDSGatedCNN(nn.Module):
    def __init__(
            self,
            ndim=98,
            seq_len=500,
            conv_out_dim=128,
            lstm_hidden=100,
            dense_hidden=64,
            dropout=0.5,
            num_classes=1
    ):
        super().__init__()
        self.conv1 = Conv2dSamePadding(ndim, conv_out_dim, kernel_size=2)#, stride=1, padding=1)
        self.sig1 = nn.Sigmoid()
        self.conv2 = Conv2dSamePadding(ndim, conv_out_dim, kernel_size=2)#, stride=1, padding=1)
        self.sig2 = nn.Sigmoid()
        
        self.conv3 = Conv2dSamePadding(ndim, conv_out_dim, kernel_size=3)#, stride=1, padding=1)
        self.sig3 = nn.Sigmoid()
        self.conv4 = Conv2dSamePadding(ndim, conv_out_dim, kernel_size=3)#, stride=1, padding=1)
        self.sig4 = nn.Sigmoid()
        
        self.lstm = nn.LSTM(conv_out_dim*2, lstm_hidden, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(lstm_hidden*2, dense_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.dense2 = nn.Linear(dense_hidden, num_classes)
        
        self.batch_norm1 = nn.BatchNorm1d(seq_len)
        self.batch_norm2 = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        """
        Input: (B, L, C) where B - batch size, L - length of sequence, C - feature dim of each sequence element 
        """
        x = self.batch_norm1(x)
        x = torch.permute(x, (0, 2, 1))

        gated_0 = self.conv1(x)
        gated_0 = self.sig1(gated_0)
        gated_0 = gated_0 * self.conv2(x)
        
        gated_1 = self.conv3(x)
        gated_1 = self.sig3(gated_1)
        gated_1 = gated_1 * self.conv4(x)
        
        x = torch.cat([gated_0, gated_1], dim=1)
        x = torch.permute(x, (0, 2, 1))
        x = self.batch_norm2(x)
        
        x, _ = self.lstm(x)
        
        x = torch.max(x, dim=1)[0]
        
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
