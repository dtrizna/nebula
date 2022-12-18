import numpy as np
import torch
import logging
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelAPI
from nebula.evaluation import getCrossValidationMetrics

from sklearn.utils import shuffle

train_limit = 500
nFolds = 3
epochs = 2
fprValues = [0.01, 0.1, 0.2]

VOCAB_SIZE = 1500
model = Cnn1DLinear(vocabSize=VOCAB_SIZE)
modelConfig = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "model": model,
    "lossFunction": torch.nn.BCEWithLogitsLoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
    "outputFolder": None, #r".\tests\speakeasy_2_Modeling\Cnn1DLinear",
    "verbosityBatches": 10
}


# LOAD DATA
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)

if train_limit:
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:train_limit]
    y_train = y_train[:train_limit]

logging.warning(f" [!] Dataset size: {len(x_train)}")

# DO Cross Validation
metrics = getCrossValidationMetrics(ModelAPI, modelConfig, x_train, y_train, 
                                epochs=epochs, folds=nFolds, fprs=fprValues, mean=True)

# metrics -- {fpr1: [mean_tpr, mean_f1], fpr2: [mean_tpr, mean_f1], ...]}
msg = f" [!] Mean values over {nFolds} folds:\n"
for fpr in metrics:
    msg += f"\tFPR: {fpr:>6} -- TPR: {metrics[fpr][0]:.4f} -- F1: {metrics[fpr][1]:.4f}\n"
logging.warning(msg)
