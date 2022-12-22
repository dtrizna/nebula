import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Cnn1DLinear(nn.Module):
    def __init__(self, 
                # embedding params
                vocabSize = None,
                embeddingDim = 64,
                paddingIdx = 0,
                # conv params
                filterSizes = [2, 3, 4, 5],
                numFilters = [128, 128, 128, 128],
                batchNormConv = False,
                convPadding = 0, # default
                # ffnn params
                hiddenNeurons = [256, 128],
                batchNormFFNN = False,
                dropout = 0.5,
                numClasses = 1): # binary classification
        super().__init__()
        self.__name__ = "Cnn1DLinear"
        # embdding
        self.embedding = nn.Embedding(vocabSize, 
                                  embeddingDim, 
                                  padding_idx=paddingIdx)

        # convolutions
        self.conv1dModule = nn.ModuleList()
        for i in range(len(filterSizes)):
                if batchNormConv:
                    module = nn.Sequential(
                                nn.Conv1d(in_channels=embeddingDim,
                                    out_channels=numFilters[i],
                                    kernel_size=filterSizes[i]),
                                nn.BatchNorm1d(numFilters[i])
                            )
                else:
                    module = nn.Conv1d(in_channels=embeddingDim,
                                    out_channels=numFilters[i],
                                    kernel_size=filterSizes[i],
                                    #padding=filterSizes[i]//2
                                    padding=convPadding
                                )
                self.conv1dModule.append(module)
        convOut = np.sum(numFilters)
        
        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(convOut, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i-1], h))

            # add BatchNorm to every layer except last
            if batchNormFFNN and i < len(hiddenNeurons)-1:
                self.ffnnBlock.append(nn.BatchNorm1d(h))

            self.ffnnBlock.append(nn.ReLU())

            if dropout:
                self.ffnnBlock.append(nn.Dropout(dropout))
            
            self.ffnn.append(nn.Sequential(*self.ffnnBlock))
        self.ffnn = nn.Sequential(*self.ffnn)
        
        self.fcOutput = nn.Linear(hiddenNeurons[-1], numClasses)
        self.relu = nn.ReLU()

    @staticmethod
    def convAndMaxPool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x_conv = [self.convAndMaxPool(embedded, conv1d) for conv1d in self.conv1dModule]
        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        out = self.fcOutput(x_fc)
        return out


class Cnn1DLSTM(nn.Module):
    def __init__(self, 
                    vocabSize = None, 
                    embeddingDim = 64,
                    hiddenDim = 128,
                    cnnFilterSizes = [2,3,4,5],
                    lstmLayers = 1,
                    convPadding = 0, # default
                    bidirectionalLSTM = True,
                    outputDim = 1): # binary classification
        super(Cnn1DLSTM, self).__init__()

        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        # convPadding = k//2 # if k%2==1 else k//2-1 -- evaluate different padding!
        self.convs = nn.ModuleList([nn.Conv1d(embeddingDim, hiddenDim, kernel_size=k, padding=convPadding) for k in cnnFilterSizes])
        self.lstm = nn.LSTM(len(cnnFilterSizes) * hiddenDim, hiddenDim, lstmLayers, bidirectional=bidirectionalLSTM)
        self.linear = nn.Linear(hiddenDim * 2 if bidirectionalLSTM else hiddenDim, outputDim)
        
    def forward(self, input, hidden=None):
        # input is of shape (batch_size, sequence_length)
        embedded = self.embedding(input)
        # embedded is of shape (batch_size, sequence_length, embedding_size)
        convolved = [conv(embedded.permute(0, 2, 1)).permute(0, 2, 1) for conv in self.convs]
        # each element in convolved is of shape (batch_size, sequence_length, hidden_size)
        concatenated = torch.cat(convolved, dim=2)
        # concatenated is of shape (batch_size, sequence_length, hidden_size * len(kernel_sizes))
        output, hidden = self.lstm(concatenated, hidden)
        # output is of shape (batch_size, sequence_length, hidden_size * 2 if bidirectional else hidden_size)
        logits = self.linear(output)
        # logits is of shape (batch_size, sequence_length, output_size)
        logits = logits.mean(dim=1)
        # logits is now of shape (batch_size, output_size)
        return logits


class LSTM(nn.Module):
    def __init__(self, 
                    vocabSize = None, 
                    embeddingDim = 64, 
                    hiddenDim = 128, 
                    lstmLayers = 1,
                    bidirectionalLSTM = True,
                    outputDim = 1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, lstmLayers, bidirectional=bidirectionalLSTM)
        self.linear = nn.Linear(hiddenDim * 2 if bidirectionalLSTM else hiddenDim, outputDim)
        
    def forward(self, input, hidden=None):
        # input is of shape (batch_size, sequence_length)
        embedded = self.embedding(input)
        # embedded is of shape (batch_size, sequence_length, embedding_size)
        output, hidden = self.lstm(embedded, hidden)
        # output is of shape (batch_size, sequence_length, hidden_size * 2 if bidirectional else hidden_size)
        logits = self.linear(output).mean(dim=1)
        return logits

