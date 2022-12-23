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
        # conv(x).permute(0, 2, 1) is of shape (batch_size, sequence_length, num_filters)
        # max(1)[0] is of shape (batch_size, num_filters)
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x_conv = [self.convAndMaxPool(embedded, conv1d) for conv1d in self.conv1dModule]
        # x_conv: list of tensors of shape (batch_size, num_filters)
        # torch.cat(x_conv, dim=1) is of shape (batch_size, num_filters * len(kernel_sizes))
        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        out = self.fcOutput(x_fc)
        return out


class Cnn1DLSTM(nn.Module):
    def __init__(self, 
                # embedding params
                vocabSize = None,
                embeddingDim = 64,
                paddingIdx = 0,
                # conv params
                filterSizes = [2, 3, 4, 5],
                numFilters = [128, 128, 128, 128],
                batchNormConv = False,
                convPadding = 0,
                # lstm params
                lstmHidden = 128,
                lstmLayers = 2,
                lstmDropout = 0,
                lstmBidirectional = False,
                # ffnn params
                hiddenNeurons = [256, 128],
                batchNormFFNN = False,
                dropout = 0.5,
                numClasses = 1): # binary classification
        super().__init__()
        self.__name__ = "Cnn1DToLSTM"
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
        
        self.lstm = nn.LSTM(input_size=convOut,
                            hidden_size=lstmHidden,
                            num_layers=lstmLayers,
                            dropout=lstmDropout,
                            batch_first=True,
                            bidirectional=lstmBidirectional)
        
        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = nn.Sequential()
            if i == 0:
                self.ffnnBlock.append(nn.Linear(lstmHidden * 2 if lstmBidirectional else lstmHidden, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i-1], h))

            # add BatchNorm to every layer except last
            if batchNormFFNN and i < len(hiddenNeurons)-1:
                self.ffnnBlock.append(nn.BatchNorm1d(h))
            # add dropout to every layer except last

            self.ffnnBlock.append(nn.ReLU())

            if dropout > 0 and i < len(hiddenNeurons)-1:
                self.ffnnBlock.append(nn.Dropout(dropout))
            
            self.ffnn.append(self.ffnnBlock)
        self.ffnn = nn.Sequential(*self.ffnn)
        self.output = nn.Linear(hiddenNeurons[-1], numClasses)

    def forward(self, x):
        # x.shape = (batchSize, seqLen)
        x = self.embedding(x)
        # x.shape = (batchSize, embeddingDim, seqLen)
        x = x.permute(0, 2, 1)
        # x.shape = (batchSize, seqLen, embeddingDim)
        convOut = []
        for i in range(len(self.conv1dModule)):
            conv = self.conv1dModule[i](x)
            # conv.shape = (batchSize, numFilters[i], seqLen)
            conv = F.relu(conv)
            # conv.shape = (batchSize, numFilters[i], seqLen)
            conv = F.max_pool1d(conv, conv.shape[2])
            # conv.shape = (batchSize, numFilters[i], 1)
            conv = conv.squeeze(2)
            # conv.shape = (batchSize, numFilters[i])
            convOut.append(conv)
        convOut = torch.cat(convOut, dim=1)
        # convOut.shape = (batchSize, convOut)
        lstmOut, _ = self.lstm(convOut.unsqueeze(1))
        # lstmOut.shape = (batchSize, 1, lstmHidden)
        lstmOut = lstmOut.squeeze(1)
        # lstmOut.shape = (batchSize, lstmHidden)
        ffnnOut = self.ffnn(lstmOut)
        # ffnnOut.shape = (batchSize, hiddenNeurons[-1])
        out = self.output(ffnnOut)
        # out.shape = (batchSize, numClasses)
        return out


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

