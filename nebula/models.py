import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nebula.constants import *

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


class Cnn1DLinearLM(nn.Module):
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
                hiddenNeurons = [512, 256],
                batchNormFFNN = False,
                dropout = 0.5,
                # pretrain layers
                pretrainLayers = [512, 1024],
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
        
        # core ffnn
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
        
        # classification output
        self.fcOutput = nn.Linear(hiddenNeurons[-1], numClasses)

        # pretrain layers
        self.preTrainLayers = []
        for i, h in enumerate(pretrainLayers):
            self.preTrainBlock = []
            if i == 0:
                self.preTrainBlock.append(nn.Linear(hiddenNeurons[-1], h))                
            else:
                self.preTrainBlock.append(nn.Linear(pretrainLayers[i-1], h))
            self.preTrainBlock.append(nn.ReLU())
            if dropout:
                self.preTrainBlock.append(nn.Dropout(dropout))
            self.preTrainLayers.append(nn.Sequential(*self.preTrainBlock))
        self.preTrainLayers.append(nn.Linear(pretrainLayers[-1], vocabSize))
        self.preTrainLayers = nn.Sequential(*self.preTrainLayers)

    @staticmethod
    def convAndMaxPool(x, conv):
        """Convolution and global max pooling layer"""
        # conv(x).permute(0, 2, 1) is of shape (batch_size, sequence_length, num_filters)
        # max(1)[0] is of shape (batch_size, num_filters)
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def core(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x_conv = [self.convAndMaxPool(embedded, conv1d) for conv1d in self.conv1dModule]
        # x_conv: list of tensors of shape (batch_size, num_filters)
        # torch.cat(x_conv, dim=1) is of shape (batch_size, num_filters * len(kernel_sizes))
        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        return x_fc

    def get_representations(self, inputs):
        return self.core(inputs).reshape(1, -1)

    def classify_representations(self, representations):
        return self.fcOutput(representations)

    def pretrain(self, inputs):
        x_core = self.core(inputs)
        return self.preTrainLayers(x_core)

    def forward(self, inputs):
        x_core = self.core(inputs)
        return self.fcOutput(x_core)


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
                    lstmHidden = 256,
                    lstmLayers = 1,
                    lstmDropout=0.1,
                    lstmBidirectional = True,
                    hiddenNeurons = [256, 64],
                    batchNormFFNN = False,
                    dropout=0.5,
                    numClasses = 1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.lstm = nn.LSTM(embeddingDim, lstmHidden, lstmLayers, bidirectional=lstmBidirectional, dropout=lstmDropout)
        #self.linear = nn.Linear(hiddenDim * 2 if bidirectionalLSTM else hiddenDim, outputDim)
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
        
    def forward(self, input, hidden=None):
        # input is of shape (batch_size, sequence_length)
        embedded = self.embedding(input)
        # embedded is of shape (batch_size, sequence_length, embedding_size)
        x, hidden = self.lstm(embedded, hidden)
        # x is of shape (batch_size, sequence_length, hidden_size * 2 if bidirectional else hidden_size)
        x = self.ffnn(x.mean(axis=1))
        logits = self.output(x)
        return logits


class MLP(nn.Module):
    def __init__(self, 
                feature_dimension=STATIC_FEATURE_VECTOR_LENGTH, 
                layer_sizes = [1024, 512, 256],
                dropout=0.5):
        super(MLP,self).__init__()
        layers = []
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(nn.Linear(feature_dimension, ls))
            else:
                layers.append(nn.Linear(layer_sizes[i-1], ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.model_base = nn.Sequential(*tuple(layers))
        self.malware_head = nn.Sequential(
                                nn.Linear(layer_sizes[-1], 64), 
                                nn.Linear(64, 1)
                            )
        
    def forward(self, data):
        base_result = self.model_base(data)
        return self.malware_head(base_result)
    
    def get_representations(self, x):
        return self.model_base(torch.Tensor(x)).reshape(1,-1)
    
    def classify_representations(self, representations):
        return self.malware_head(representations)
