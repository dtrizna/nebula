import numpy as np
import torch
import logging
import os
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelAPI
from nebula.attention import TransformerEncoderModel
from nebula.evaluation import getCrossValidationMetrics
from nebula.misc import dictToString
from sklearn.utils import shuffle
import time

#RUN_NAME = "Cnn1DLinearHiddenLayers"
RUN_NAME = "TransformerEmbeddingDim"

VOCAB_SIZE = 1500
# modelArch = {
#     "vocabSize": VOCAB_SIZE,
#     "embeddingDim": 64,
#     "hiddenNeurons": None,
#     "batchNormConv": False,
#     "batchNormFFNN": False,
#     "filterSizes": [2, 3, 4, 5]
# }

modelArch = {
    "nTokens": VOCAB_SIZE,  # size of vocabulary
    "dModel": 192,  # embedding & transformer dimension
    "nHeads": 2,  # number of heads in nn.MultiheadAttention
    "dHidden": 200,  # dimension of the feedforward network model in nn.TransformerEncoder
    "nLayers": 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    "numClasses": 1, # binary classification
    "hiddenNeurons": [64],
    "layerNorm": False,
    "dropout": 0.5
}


# =============== Cross-Valiation CONFIG
train_limit = None # 500
nFolds = 3
epochs = 3
fprValues = [0.0001, 0.001, 0.01, 0.1]
rest = 60*5 # 5 minutes to rest between folds (to cool down)

# ================ setup

outputRootFolder = r"C:\Users\dtrizna\Code\nebula\evaluation\crossValidation"
outputFolder = os.path.join(outputRootFolder, RUN_NAME)
os.makedirs(outputFolder, exist_ok=True)

logFile = f"ArchSelection_{RUN_NAME}_{int(time.time())}.log"

if logFile:
    logging.basicConfig(filename=os.path.join(outputFolder, logFile), level=logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# =============== LOAD DATA
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_x.npy"
x_train = np.load(x_train)
y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)

if train_limit:
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_train = x_train[:train_limit]
    y_train = y_train[:train_limit]

logging.warning(f" [!] Dataset size: {len(x_train)}")

# =============== DEFINE MODEL & ITS CONFIG
#hiddenNeurons = [[128], [256, 128]]
for embeddingDim in [128, 192, 256]:
#for layers in hiddenNeurons:
    folder = f"trainingFiles"

    #modelArch["hiddenNeurons"] = layers
    modelArch["dModel"] = embeddingDim
    #model = Cnn1DLinear(**modelArch)
    model = TransformerEncoderModel(**modelArch)

    configStr = dictToString(modelArch)
    metricFilename = f"metrics_{model.__name__}_limit_{train_limit}_ep_{epochs}_cv_{nFolds}_{configStr}.json"
    metricFileFullpath = os.path.join(outputFolder, metricFilename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.warning(f" [!] Using device: {device}")
    modelConfig = {
        "device": device,
        "model": model,
        "lossFunction": torch.nn.BCEWithLogitsLoss(),
        "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
        "outputFolder": os.path.join(outputFolder, folder),
        "verbosityBatches": 100
    }

    # =============== DO Cross Validation
    logging.warning(f" [!] Epochs per fold: {epochs} | Model config: {modelArch}")
    
    metrics = getCrossValidationMetrics(ModelAPI, modelConfig, x_train, y_train, 
                                    epochs=epochs, folds=nFolds, fprs=fprValues, 
                                    mean=False, metricFile=metricFileFullpath)

    # if mean=True, can use this to print the results
    # metrics -- {fpr1: [mean_tpr, mean_f1], fpr2: [mean_tpr, mean_f1], ...]}
    # msg = f" [!] Average epoch time: {metrics['avg_epoch_time']:.2f}s | Mean values over {nFolds} folds:\n"
    # for fpr in fprValues:
    #     msg += f"\tFPR: {fpr:>6} -- TPR: {metrics[fpr][0]:.4f} -- F1: {metrics[fpr][1]:.4f}\n"
    # logging.warning(msg)

    if rest:
        time.sleep(rest)
