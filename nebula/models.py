import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import os
import pickle

from sklearn.metrics import f1_score, accuracy_score
import logging


class ModelAPI(object):
    def __init__(self, 
                    device, 
                    model,
                    lossFunction,
                    optimizer, 
                    verbosityBatches = 100,
                    outputFolder=None,
                    stateDict = None):
        self.model = model
        self.optimizer = optimizer
        self.lossFunction = lossFunction
        self.verbosityBatches = verbosityBatches

        self.device = device
        self.model.to(self.device)

        self.outputFolder = outputFolder
        if stateDict:
            self.model.load_state_dict(torch.load(stateDict))

    def loadState(self, stateDict):
        self.model.load_state_dict(torch.load(stateDict))

    def dumpResults(self):
        prefix = f"{self.outputFolder}\\{int(time.time())}"
        os.makedirs(self.outputFolder, exist_ok=True)

        modelFile = f"{prefix}-model.torch"
        torch.save(self.model.state_dict(), modelFile)

        with open(f"{prefix}-trainLosses.pickle", "wb") as f:
            pickle.dump(self.trainLosses, f)
        
        # in form [accuracy, f1-score]
        np.save(f"{prefix}-trainMetrics.npy", self.trainMetrics)
        
        with open(f"{prefix}-trainingTime.pickle", "wb") as f:
            pickle.dump(self.trainingTime, f)

        dumpString = f"""
        [!] {time.ctime()}: Dumped results:
                model: {modelFile}
                train loss list: {prefix}-train_losses.pickle
                train metrics : {prefix}-train_metrics.pickle
                duration: {prefix}-duration.pickle"""
        logging.warning(dumpString)

    def trainEpoch(self, trainLoader, epochId):
        trainMetrics = []
        trainLoss = []
        now = time.time()

        for batchIdx, (data, target) in enumerate(trainLoader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.lossFunction(logits, target.reshape(-1,1))

            trainLoss.append(loss.item())

            loss.backward() # derivatives
            self.optimizer.step() # parameter update  

            preds = (torch.sigmoid(logits).clone().detach().numpy() > 0.5).astype(np.int_)
            accuracy = accuracy_score(target, preds)
            f1 = f1_score(target, preds)
            trainMetrics.append([accuracy, f1])
            
            if batchIdx % self.verbosityBatches == 0:
                logging.warning(" [*] {}: Train Epoch: {} [{:^5}/{:^5} ({:^2.0f}%)]\tLoss: {:.4f}e-3 | F1-score: {:.2f} | Elapsed: {:.2f}s".format(
                    time.ctime(), epochId, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader), loss.item()*1e3, np.mean([x[1] for x in trainMetrics]), time.time()-now))
                now = time.time()
        
        trainMetrics = np.array(trainMetrics).mean(axis=0).reshape(-1,2)
        return trainLoss, trainMetrics


    def fit(self, X, y, epochs=10, batchSize=32):
        trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=batchSize,
            shuffle=True
        )
        
        self.model.train()
        
        self.trainLosses = []
        self.trainMetrics = []
        self.trainingTime = []
        try:
            for epochIdx in range(1, epochs + 1):
                epochStartTime = time.time()
                logging.warning(f" [*] Started epoch: {epochIdx}")

                epochTrainLoss, epochTrainMetric = self.trainEpoch(trainLoader, epochIdx)
                self.trainLosses.extend(epochTrainLoss)
                self.trainMetrics.append(epochTrainMetric)

                timeElapsed = time.time() - epochStartTime
                self.trainingTime.append(timeElapsed)
                logging.warning(f" [*] {time.ctime()}: {epochIdx + 1:^7} | Tr.loss: {np.mean(epochTrainLoss)*1e3:.4f}e-3 | Tr.F1.: {np.mean([x[1] for x in epochTrainMetric]):^9.2f} | {timeElapsed:^9.2f}s")
            if self.outputFolder:
                self.dumpResults()
        except KeyboardInterrupt:
            if self.outputFolder:
                self.dumpResults()

    def evaluate(self, X, y, batchSize=32):
        testLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=batchSize,
            shuffle=True
        )
        self.model.eval()

        self.testLoss = []
        self.testMetrics = []
        for data, target in testLoader:
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                logits = self.model(data)

            loss = self.lossFunction(logits, target.float().reshape(-1,1))
            self.testLoss.append(loss.item())

            preds = (torch.sigmoid(logits).clone().detach().numpy() > 0.5).astype(np.int_)
            accuracy = accuracy_score(target, preds)
            f1 = f1_score(target, preds)
            self.testMetrics.append([accuracy, f1])

        self.testMetrics = np.array(self.testMetrics).mean(axis=0).reshape(-1,2)
        return self.testLoss, self.testMetrics

    def predict_proba(self, arr):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.Tensor(arr).long().to(self.device))
        return torch.sigmoid(logits).clone().detach().numpy()
    
    def predict(self, arr, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict_proba(arr)
        return (logits > threshold).astype(np.int_)

class Cnn1DLinear(nn.Module):
    def __init__(self, 
                # embedding params
                vocabSize = None,
                embeddingDim = 32,
                paddingIdx = 0,
                # conv params
                filterSizes = [2, 3, 4, 5],
                numFilters = [128, 128, 128, 128],
                batchNormConv = False,
                # ffnn params
                hiddenNeurons = [256, 128],
                batchNormFFNN = False,
                dropout = 0.5,
                numClasses = 1): # binary classification
        super().__init__()

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
                                    kernel_size=filterSizes[i])
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
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x_conv = [self.convAndMaxPool(embedded, conv1d) for conv1d in self.conv1dModule]
        x_fc = self.ffnn(torch.cat(x_conv, dim=1))
        out = self.fcOutput(x_fc)
        return out
