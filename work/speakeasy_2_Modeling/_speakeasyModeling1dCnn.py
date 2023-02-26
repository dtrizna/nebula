import numpy as np
import torch
import pickle
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear
from nebula import ModelTrainer

from sklearn.utils import shuffle

# config
train_limit = 1000
test_limit = 5000
VOCAB_SIZE = 1500
device = "cuda:0" if torch.cuda.is_available() else "cpu"
outFolder = None # r".\tests\modeling\Cnn1DLinear"
verbosity = 100

# ===============
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)

if train_limit:
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:train_limit]
    y_train = y_train[:train_limit]
    
print("Trainset size: ", len(x_train))

# ===============
x_test = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_test = np.load(x_test)
y_test = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_testset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_test = np.load(y_test)

if test_limit:
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    x_test = x_test[:test_limit]
    y_test = y_test[:test_limit]

print("Testset size: ", len(x_test))

# ===============
model = Cnn1DLinear(vocabSize=VOCAB_SIZE)
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = ModelTrainer(device, model, loss_function, optimizer, outputFolder=outFolder, verbosity_n_batches=verbosity)

# ===============
print("Training...")
trainer.fit(x_train, y_train, epochs=3)

# ===============
print("Evaluating on test set...")
testLoss, testMetrics = trainer.evaluate(x_test, y_test)
print("Test F1-score:", testMetrics[0][0])

if outFolder: # dump test metrics too
    with open(outFolder + r"\testMetrics.pkl", "wb") as f:
        pickle.dump(testMetrics, f)
    with open(outFolder + r"\testLoss.pkl", "wb") as f:
        pickle.dump(testLoss, f)

import pdb; pdb.set_trace()