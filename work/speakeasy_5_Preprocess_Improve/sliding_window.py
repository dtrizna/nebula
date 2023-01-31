import os
import json
import time
import logging
import numpy as np
from sklearn.utils import shuffle

from torch import cuda
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import sys
sys.path.append("../..")
sys.path.append(".")
from nebula import ModelTrainer
from nebula.attention import TransformerEncoderLM, PositionalEncoding, TransformerEncoderWithChunking
from nebula.misc import get_path, set_random_seed, clear_cuda_cache
SCRIPT_PATH = get_path(type="script")
REPO_ROOT = os.path.join(SCRIPT_PATH, "..", "..")

# ====== LOGGING
logFile = f"CrossValidationRun_{int(time.time())}.log"
logging.basicConfig(
    filename=None, #os.path.join(outputFolder, logFile),
    level=logging.WARNING,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


random_state = 42
set_random_seed(random_state)

run_config = {
    "train_limit": 1000,
    "optim_step_size": 1000,
    "batch_size": 64,
    "chunk_size": 96
}

# ===== LOADING DATA ==============
vocabSize = 50000
maxLen = 512
xTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
xTrain = np.load(xTrainFile)
yTrainFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_trainset_BPE", "speakeasy_y.npy")
yTrain = np.load(yTrainFile)
xTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE", f"speakeasy_VocabSize_50000_maxLen_{maxLen}_x.npy")
xTest = np.load(xTestFile)
yTestFile = os.path.join(REPO_ROOT, "data", "data_filtered", "speakeasy_testset_BPE", "speakeasy_y.npy")
yTest = np.load(yTestFile)

if run_config['train_limit']:
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=random_state)
    xTrain = xTrain[:run_config['train_limit']]
    yTrain = yTrain[:run_config['train_limit']]
    xTest, yTest = shuffle(xTest, yTest, random_state=random_state)
    xTest = xTest[:run_config['train_limit']]
    yTest = yTest[:run_config['train_limit']]

vocabFile = os.path.join(REPO_ROOT, "nebula", "objects", "speakeasy_BPE_50000_vocab.json")
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
vocabSize = len(vocab) # adjust it to exact number of tokens in the vocabulary


# class TransformerEncoderWithChunking(nn.Module):
#     def __init__(self,
#                     vocabSize: int, # size of vocabulary
#                     maxLen: int, # maximum length of input sequence
#                     dModel: int = 32, # embedding & transformer dimension
#                     nHeads: int = 8, # number of heads in nn.MultiheadAttention
#                     dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
#                     nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#                     numClasses: int = 1, # 1 ==> binary classification 
#                     hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
#                     layerNorm: bool = False, # whether to normalize decoder's FFNN layers
#                     pretrainLayers: list = [1024], # pretrain layers
#                     dropout: float = 0.3):
#         super().__init__()
#         assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
#         self.__name__ = 'Transformer'
#         self.encoder = nn.Embedding(vocabSize, dModel)
#         self.pos_encoder = PositionalEncoding(dModel, dropout)
#         encoder_layers = TransformerEncoderLayer(dModel, nHeads, dHidden, dropout, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
#         self.d_model = dModel

#         nr_of_chunks = int(maxLen/run_config['chunk_size']) + 1
#         input_neurons = run_config['chunk_size'] * nr_of_chunks * dModel
#         self.ffnn = []
#         for i,h in enumerate(hiddenNeurons):
#             self.ffnnBlock = []
#             if i == 0:
#                 self.ffnnBlock.append(nn.Linear(input_neurons, h))
#             else:
#                 self.ffnnBlock.append(nn.Linear(hiddenNeurons[i-1], h))

#             # add BatchNorm to every layer except last
#             if layerNorm and i < len(hiddenNeurons)-1:
#                 self.ffnnBlock.append(nn.LayerNorm(h))

#             self.ffnnBlock.append(nn.ReLU())

#             if dropout:
#                 self.ffnnBlock.append(nn.Dropout(dropout))
            
#             self.ffnn.append(nn.Sequential(*self.ffnnBlock))
#         self.ffnn = nn.Sequential(*self.ffnn)

#         # pretrain layers
#         self.preTrainLayers = []
#         for i, h in enumerate(pretrainLayers):
#             self.preTrainBlock = []
#             if i == 0:
#                 self.preTrainBlock.append(nn.Linear(hiddenNeurons[-1], h))                
#             else:
#                 self.preTrainBlock.append(nn.Linear(pretrainLayers[i-1], h))
#             self.preTrainBlock.append(nn.ReLU())
#             if dropout:
#                 self.preTrainBlock.append(nn.Dropout(dropout))
#             self.preTrainLayers.append(nn.Sequential(*self.preTrainBlock))
#         self.preTrainLayers.append(nn.Linear(pretrainLayers[-1], vocabSize))
#         self.preTrainLayers = nn.Sequential(*self.preTrainLayers)

#         self.fcOutput = nn.Linear(hiddenNeurons[-1], numClasses)

#         self.init_weights()

#     def init_weights(self) -> None:
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         for block in self.ffnn:
#             for layer in block:
#                 if isinstance(layer, nn.Linear):
#                     layer.bias.data.zero_()
#                     layer.weight.data.uniform_(-initrange, initrange)

#     def core(self, src: Tensor) -> Tensor:
#         chunks = []
#         for chunk in torch.split(src, split_size_or_sections=run_config['chunk_size'], dim=1):
#         # how to pad mask works -- (left, 0, right -- up to chunk_size from existing)
#             if chunk.shape[1] < run_config['chunk_size']:
#                 pad_mask = (0,run_config['chunk_size']-chunk.shape[1])
#                 chunk = torch.nn.functional.pad(chunk, pad=pad_mask)
    
#             chunk = self.encoder(chunk) * math.sqrt(self.d_model)
#             chunk = self.pos_encoder(chunk)        
#             chunk = self.transformer_encoder(chunk)
#             # at this stage each chunk is: (batch_size, chunk_size, d_model)
#             chunks.append(chunk)
#         # after cat it'd be: (batch_size, chunk_size * nr_of_chunks * d_model, d_model)
#         # where nr_of_chunks = int(maxLen/run_config['chunk_size']) + 1
#         x = torch.cat(chunks, dim=1)
#         x = x.view(x.size(0), -1)
#         x = self.ffnn(x)
#         return x
    
#     def pretrain(self, x: Tensor) -> Tensor:
#         x = self.core(x)
#         x = self.preTrainLayers(x)
#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.core(x)
#         out = self.fcOutput(x)
#         return out


modelArch = {
    "vocabSize": vocabSize,  # size of vocabulary
    "maxLen": maxLen,  # maximum length of the input sequence
    "chunk_size": run_config["chunk_size"],
    "dModel": 96,  # embedding & transformer dimension
    "nHeads": 8,  # number of heads in nn.MultiheadAttention
    "dHidden": 256,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [128],
    "layerNorm": False,
    "dropout": 0.3
}

model = TransformerEncoderWithChunking(**modelArch)

device = "cuda" if cuda.is_available() else "cpu"
modelInterfaceConfig = {
    "device": device,
    "model": model, 
    "lossFunction": BCEWithLogitsLoss(),
    "optimizerClass": Adam,
    "optimizerConfig": {"lr": 2.5e-4},
    "optimSchedulerClass": "step",
    "optim_step_size": run_config["optim_step_size"],
    "outputFolder": None, # will be set later
    "batchSize": run_config["batch_size"],
    "verbosityBatches": 100,
}

# ===== TRAINING ==============
import torch

trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(xTrain).long(), torch.from_numpy(yTrain).float()),
            batch_size=run_config["batch_size"],
            shuffle=True
        )

loss = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=2.5e-4)

for i, (data, target) in enumerate(trainLoader):
    print(f"Batch: {i}")
    logits = model(data)
    loss_value = loss(logits, target.reshape(-1,1))
    print(f"Loss: {loss_value.item():.6f}")
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()