import numpy as np
import torch
import logging
import os
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelAPI
from nebula.evaluation import getCrossValidationMetrics
from nebula.misc import dictToString
from sklearn.utils import shuffle

outputFolder = rf"C:\Users\dtrizna\Code\nebula\tests\speakeasy_3_CV_ArchSelection"
logFile = "speakeasy_3_CV_ArchSelection_HiddenLayers.log"
if logFile:
    logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# =============== Cross-Valiation CONFIG
train_limit = None
nFolds = 3
epochs = 5
fprValues = [0.0001, 0.001, 0.01, 0.1]

# =============== LOAD DATA
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)

if train_limit:
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:train_limit]
    y_train = y_train[:train_limit]

logging.warning(f" [!] Dataset size: {len(x_train)}")

# =============== DEFINE MODEL & ITS CONFIG
hiddenLayers = [[128], [256, 128], [512, 256], [1024, 512, 256]]
#for embeddingDim in [32, 64, 96]:
for layers in hiddenLayers:
    VOCAB_SIZE = 1500
    modelArch = {
        "vocabSize": VOCAB_SIZE,
        "embeddingDim": 64,
        "hiddenNeurons": layers,
        "batchNormConv": False,
        "batchNormFFNN": False,
        "filterSizes": [2, 3, 4, 5]
    }
    model = Cnn1DLinear(**modelArch)

    configStr = dictToString(modelArch)
    metricFilename = f"metrics_{model.__name__}_ep_{epochs}_cv_{nFolds}_{configStr}.json"
    metricFileFullpath = os.path.join(outputFolder, metricFilename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelConfig = {
        "device": device,
        "model": model,
        "lossFunction": torch.nn.BCEWithLogitsLoss(),
        "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
        "outputFolder": None,
        "verbosityBatches": 100
    }

    # =============== DO Cross Validation
    logging.warning(f" [!] Epochs per fold: {epochs} | Model config: {modelArch}")
    
    metrics = getCrossValidationMetrics(ModelAPI, modelConfig, x_train, y_train, 
                                    epochs=epochs, folds=nFolds, fprs=fprValues, 
                                    mean=True, metricFile=metricFileFullpath)

    # metrics -- {fpr1: [mean_tpr, mean_f1], fpr2: [mean_tpr, mean_f1], ...]}
    msg = f" [!] Average epoch time: {metrics['avg_epoch_time']:.2f}s | Mean values over {nFolds} folds:\n"
    for fpr in fprValues:
        msg += f"\tFPR: {fpr:>6} -- TPR: {metrics[fpr][0]:.4f} -- F1: {metrics[fpr][1]:.4f}\n"

    logging.warning(msg)
