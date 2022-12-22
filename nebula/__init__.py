from nebula.preprocessing import JSONTokenizer
from nebula.emulation import PEDynamicFeatureExtractor
from nebula.models import Cnn1DLinear

import os
import time
import pickle
import logging
import numpy as np

import torch
from sklearn.metrics import f1_score, accuracy_score


class ModelInterface(object):
    def __init__(self, 
                    device, 
                    model,
                    lossFunction,
                    optimizerClass,
                    optimizerConfig,
                    batchSize=64,
                    verbosityBatches = 100,
                    outputFolder=None,
                    stateDict = None):
        self.model = model
        self.optimizer = optimizerClass(self.model.parameters(), **optimizerConfig)
        self.lossFunction = lossFunction
        self.verbosityBatches = verbosityBatches
        self.batchSize = batchSize

        self.device = device
        self.model.to(self.device)

        self.outputFolder = outputFolder
        if stateDict:
            self.model.load_state_dict(torch.load(stateDict))

    def loadState(self, stateDict):
        self.model.load_state_dict(torch.load(stateDict))

    def dumpResults(self):
        prefix = os.path.join(self.outputFolder, f"trainingFiles_{int(time.time())}")
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

            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            preds = (predProbs > 0.5).astype(np.int_)

            # might be expensive to compute every batch?
            # tpr = get_tpr_at_fpr(predProbs, target.cpu().numpy(), 0.1)
            accuracy = accuracy_score(target.cpu(), preds)
            f1 = f1_score(target.cpu(), preds)
            trainMetrics.append([accuracy, f1])
            
            if batchIdx % self.verbosityBatches == 0:
                logging.warning(" [*] {}: Train Epoch: {} [{:^5}/{:^5} ({:^2.0f}%)]\tLoss: {:.6f} | F1-score: {:.2f} | Elapsed: {:.2f}s".format(
                    time.ctime(), epochId, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader), loss.item(), np.mean([x[1] for x in trainMetrics]), time.time()-now))
                now = time.time()
        
        trainMetrics = np.array(trainMetrics).mean(axis=0).reshape(-1,2)
        return trainLoss, trainMetrics


    def fit(self, X, y, epochs=10):
        trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batchSize,
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
                logging.warning(f" [*] {time.ctime()}: {epochIdx:^7} | Tr.loss: {np.mean(epochTrainLoss):.6f} | Tr.F1.: {np.mean([x[1] for x in epochTrainMetric]):^9.2f} | {timeElapsed:^9.2f}s")
            if self.outputFolder:
                self.dumpResults()
        except KeyboardInterrupt:
            if self.outputFolder:
                self.dumpResults()

    def train(self, X, y, epochs=10):
        self.fit(self, X, y, epochs=epochs)

    def evaluate(self, X, y):
        testLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batchSize,
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

            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            preds = (predProbs > 0.5).astype(np.int_)
            accuracy = accuracy_score(target.cpu(), preds)
            f1 = f1_score(target.cpu(), preds)
            self.testMetrics.append([accuracy, f1])

        self.testMetrics = np.array(self.testMetrics).mean(axis=0).reshape(-1,2)
        return self.testLoss, self.testMetrics

    def predict_proba(self, arr):
        out = torch.empty((0,1)).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(arr).long()),
            batch_size=self.batchSize,
            shuffle=False
        )

        self.model.eval()
        for data in loader:
            with torch.no_grad():
                logits = self.model(torch.Tensor(data[0]).long().to(self.device))
                out = torch.vstack([out, logits])
        
        return torch.sigmoid(out).clone().detach().cpu().numpy()
    
    def predict(self, arr, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict_proba(arr)
        return (logits > threshold).astype(np.int_)
