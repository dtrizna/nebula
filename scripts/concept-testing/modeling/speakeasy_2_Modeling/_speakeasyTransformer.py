import numpy as np
import torch
import logging
import os
import pickle
import sys
sys.path.extend(['..', '.'])
from nebula.attention import TransformerEncoderModel
from nebula import ModelTrainer
from sklearn.utils import shuffle

outputFolder = r"C:\Users\dtrizna\Code\nebula\tests\speakeasy_2_Modeling\output"
# os.makedirs(outputFolder, exist_ok=True)
logFile = None # "transformer.log"
if logFile:
    logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# =============== LOAD DATA
train_limit = 500 # None
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)

if train_limit:
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:train_limit]
    y_train = y_train[:train_limit]

logging.warning(f" [!] Dataset size: {len(x_train)}")

vocabPath = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500.pkl"
with open(vocabPath, "rb") as f:
    vocab = pickle.load(f)


# =============== DEFINE MODEL & ITS CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformerEncoderModelConfig = {
    "nTokens": len(vocab),  # size of vocabulary
    "dModel": 200,  # embedding & transformer dimension
    "nHeads": 2,  # number of heads in nn.MultiheadAttention
    "dHidden": 200,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "classifier_head": [64],
    "layerNorm": False,
    "dropout": 0.5
}
model = TransformerEncoderModel(**transformerEncoderModelConfig).to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelConfig = {
    "device": device,
    "model": model,
    "loss_function": torch.nn.BCEWithLogitsLoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
    "outputFolder": None,
    "verbosity_n_batches": 10
}

ModelTrainer(**modelConfig).fit(x_train, y_train, epochs=3)

trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(x_train).long(), 
                torch.from_numpy(y_train).float()),
            batch_size=64,
            shuffle=True)
data = next(iter(trainLoader))

out = model(data[0].to(device))#, src_mask)
outProbs = torch.sigmoid(out)
import pdb; pdb.set_trace()