import numpy as np
import torch
import pickle
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelAPI
from nebula import JSONTokenizer

x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_10000_maxLen_2048.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_10000_maxLen_2048_y.npy"
y_train = np.load(y_train)
print(x_train.shape)
print(y_train.shape)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_train).long(), torch.from_numpy(y_train).long()),
    batch_size=32,
    shuffle=True
)

# x_test = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset\speakeasy_VocabSize_10000_maxLen_2048.npy"
# x_test = np.load(x_test)
# y_test = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset\speakeasy_VocabSize_10000_maxLen_2048_y.npy"
# y_test = np.load(y_test)
# print(x_test.shape)
# print(y_test.shape)
# test_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(torch.from_numpy(x_test).long(), torch.from_numpy(y_test).long()),
#     batch_size=32,
#     shuffle=True
# )

device = "cuda:0" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 1500
MAX_LEN = 2048

model = Cnn1DLinear(vocabSize=VOCAB_SIZE)
lossFunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# vocabFile = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_10000.pkl"
# vocab = pickle.load(open(vocabFile, "rb"))
# tokenizer = JSONTokenizer()
# tokenizer.loadVocab(vocabFile)

trainer = ModelAPI(device, model, lossFunction, optimizer, outputFolder="./modelTests")
trainer.fit(train_loader, epochs=3)
import pdb; pdb.set_trace()