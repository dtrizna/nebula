import numpy as np
import torch
import pickle
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelAPI

train_limit = 1000

x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)[:train_limit]

y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)[:train_limit]

print("Trainset size: ", len(x_train))

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_train).long(), torch.from_numpy(y_train).long()),
    batch_size=32,
    shuffle=True
)

test_limit = 5000

x_test = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_test = np.load(x_test)[:test_limit]

y_test = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_test = np.load(y_test)[:test_limit]

print("Testset size: ", len(x_test))

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_test).long(), torch.from_numpy(y_test).long()),
    batch_size=32,
    shuffle=True
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 1500

model = Cnn1DLinear(vocabSize=VOCAB_SIZE)
lossFunction = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

outFolder = r".\tests\modeling\Cnn1DLinear"
verbosity = 100
trainer = ModelAPI(device, model, lossFunction, optimizer, outputFolder=outFolder, verbosityBatches=verbosity)
trainer.fit(train_loader, epochs=3)

print("Evaluating on test set...")
testLoss, testMetrics = trainer.evaluate(test_loader)
print("Test F1-score:", testMetrics[0][0])

with open(outFolder + r"\testMetrics.pkl", "wb") as f:
    pickle.dump(testMetrics, f)
with open(outFolder + r"\testLoss.pkl", "wb") as f:
    pickle.dump(testLoss, f)

import pdb; pdb.set_trace()