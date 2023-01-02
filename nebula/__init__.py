import nebula
from nebula.preprocessing import JSONTokenizer, PEDynamicFeatureExtractor, PEStaticFeatureExtractor
from nebula.models import Cnn1DLinearLM, MLP

import os
import time
import json
import logging
import numpy as np
from pandas import DataFrame

import torch
from torch import nn
from sklearn.metrics import f1_score
from nebula.evaluation import get_tpr_at_fpr


class ModelInterface(object):
    def __init__(self, 
                    model,
                    device, 
                    lossFunction,
                    optimizerClass,
                    optimizerConfig,
                    batchSize=64,
                    verbosityBatches = 100,
                    outputFolder=None,
                    falsePositiveRates=[0.0001, 0.001, 0.01, 0.1],
                    modelForwardPass=None,
                    stateDict = None):
        self.model = model    
        self.optimizer = optimizerClass(self.model.parameters(), **optimizerConfig)
        self.lossFunction = lossFunction
        self.verbosityBatches = verbosityBatches
        self.batchSize = batchSize

        self.falsePositiveRates = falsePositiveRates
        self.fprReportingIdx = 1 # what FPR stats to report every epoch

        self.device = device
        self.model.to(self.device)

        self.outputFolder = outputFolder
        if stateDict:
            self.model.load_state_dict(torch.load(stateDict))
        if modelForwardPass:
            self.model.forwardPass = modelForwardPass
        else:
            self.model.forwardPass = self.model.forward

    def loadState(self, stateDict):
        self.model.load_state_dict(torch.load(stateDict))

    def dumpResults(self):
        prefix = os.path.join(self.outputFolder, f"trainingFiles_{int(time.time())}")
        os.makedirs(self.outputFolder, exist_ok=True)

        modelFile = f"{prefix}-model.torch"
        torch.save(self.model.state_dict(), modelFile)
        np.save(f"{prefix}-trainTime.npy", np.array(self.trainingTime))
        np.save(f"{prefix}-trainLosses.npy", np.array(self.trainLosses))
        dumpString = f"""[!] {time.ctime()}: Dumped results:
                model     : {modelFile}
                losses    : {prefix}-train_losses.npy
                duration  : {prefix}-trainTime.npy"""

        if not np.isnan(self.trainF1s).all():
            np.save(f"{prefix}-trainF1s.npy", self.trainF1s)
            dumpString += f"\n\t\ttrain F1s : {prefix}-trainF1s.npy"
        if not np.isnan(self.trainTruePositiveRates).all():
            np.save(f"{prefix}-trainTPRs.npy", self.trainTruePositiveRates)
            dumpString += f"\n\t\ttrain TPRs: {prefix}-trainTPRs.npy"

        logging.warning(dumpString)
    
    def getMetrics(self, target, predProbs):
        metrics = []
        for fpr in self.falsePositiveRates:
            tpr, threshold = get_tpr_at_fpr(target, predProbs, fpr)
            f1_at_fpr = f1_score(target, predProbs >= threshold)
            metrics.append([tpr, f1_at_fpr])        
        return np.array(metrics)

    def trainEpoch(self, trainLoader, epochId):
        epochMetrics = []
        epochLosses = []
        now = time.time()

        for batchIdx, (data, target) in enumerate(trainLoader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model.forwardPass(data)
            # if dimension of target is 1, reshape to (-1,1)
            if target.dim() == 1:
                target = target.reshape(-1, 1)
            loss = self.lossFunction(logits, target)

            epochLosses.append(loss.item())

            loss.backward() # derivatives
            self.optimizer.step() # parameter update  

            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            if target.shape[1] == 1:
                batchMetrics = self.getMetrics(target.cpu().numpy(), predProbs)
                epochMetrics.append(batchMetrics)
            
            if batchIdx % self.verbosityBatches == 0:
                if target.shape[1] == 1:
                    metricsReport = f"FPR {self.falsePositiveRates[self.fprReportingIdx]} -- "
                    metricsReport += f"TPR {batchMetrics[self.fprReportingIdx, 0]:.4f} | F1 {batchMetrics[self.fprReportingIdx, 1]:.4f}"
                else:
                    metricsReport = "Not computing TPR and F1"
                logging.warning(" [*] {}: Train Epoch: {} [{:^5}/{:^5} ({:^2.0f}%)]\tLoss: {:.6f} | {} | Elapsed: {:.2f}s".format(
                    time.ctime(), epochId, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader), loss.item(), metricsReport, time.time() - now))
                now = time.time()
        
        if epochMetrics: # we do not collect metrics if target is not binary
            # taking mean across all batches
            epochMetrics = np.array(epochMetrics).mean(axis=0).reshape(-1,2)
            epochTPRs, epochF1s = epochMetrics[:,0], epochMetrics[:,1]
            # shape of each metrics array: (len(falsePositiveRates), )
            return epochLosses, epochTPRs, epochF1s
        else:
            return epochLosses, None, None

    def fit(self, X, y, epochs=10):
        trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batchSize,
            shuffle=True
        )
        self.model.train()
        self.trainTruePositiveRates = np.empty(shape=(epochs, len(self.falsePositiveRates)))
        self.trainF1s = np.empty(shape=(epochs, len(self.falsePositiveRates)))
        self.trainLosses = []
        self.trainingTime = []
        try:
            for epochIdx in range(1, epochs + 1):
                epochStartTime = time.time()
                logging.warning(f" [*] Started epoch: {epochIdx}")

                epochTrainLoss, epochTPRs, epochF1s = self.trainEpoch(trainLoader, epochIdx)
                self.trainTruePositiveRates[epochIdx-1] = epochTPRs
                self.trainF1s[epochIdx-1] = epochF1s
                self.trainLosses.extend(epochTrainLoss)
                timeElapsed = time.time() - epochStartTime
                self.trainingTime.append(timeElapsed)
                
                if epochTPRs is not None and epochF1s is not None:
                    metricsReport = f"FPR {self.falsePositiveRates[self.fprReportingIdx]} -- "
                    metricsReport += f"TPR: {epochTPRs[self.fprReportingIdx]:.2f} |  F1: {epochF1s[self.fprReportingIdx]:.2f}"
                else:
                    metricsReport = "Not computing TPR and F1"
                logging.warning(f" [*] {time.ctime()}: {epochIdx:^7} | Tr.loss: {np.mean(epochTrainLoss):.6f} | {metricsReport} | Elapsed: {timeElapsed:^9.2f}s")
            if self.outputFolder:
                self.dumpResults()
        except KeyboardInterrupt:
            if self.outputFolder:
                self.dumpResults()

    def train(self, X, y, epochs=10):
        self.fit(self, X, y, epochs=epochs)

    def evaluate(self, X, y, metrics="array"):
        testLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batchSize,
            shuffle=True
        )
        self.model.eval()

        self.testLoss = []
        self.testTruePositiveRates = []
        self.testF1s = []
        for data, target in testLoader:
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                logits = self.model.forwardPass(data)

            loss = self.lossFunction(logits, target.float().reshape(-1,1))
            self.testLoss.append(loss.item())

            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            batchMetrics = self.getMetrics(target.cpu().numpy(), predProbs)
            self.testTruePositiveRates.append(batchMetrics[:,0])
            self.testF1s.append(batchMetrics[:,1])
            
        # take mean across all batches for each metric
        # resulting shape of each metric: (len(falsePositiveRates), )
        self.testTruePositiveRates = np.array(self.testTruePositiveRates).mean(axis=0)
        self.testF1s = np.array(self.testF1s).mean(axis=0)
        if metrics == "array":
            return self.testLoss, self.testTruePositiveRates, self.testF1s
        elif metrics == "json":
            return self.metricsToJSON([self.testLoss, self.testTruePositiveRates, self.testF1s])
        else:
            raise ValueError("evaluate(): Inappropriate metrics value, must be in: ['array', 'json']")

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
    
    def metricsToJSON(self, metrics):
        tprs = dict(zip(["fpr_"+str(x) for x in self.falsePositiveRates], metrics[1]))
        f1s = dict(zip(["fpr_"+str(x) for x in self.falsePositiveRates], metrics[2]))
        metricDict = DataFrame([tprs, f1s], index=["tpr", "f1"]).to_dict()
        metricDict['loss'] = metrics[0]
        return metricDict


class PEHybridClassifier(nn.Module):
    def __init__(self,
                    speakeasyConfig=None,
                    vocab=None,
                    outputFolder=None,
                    malwareHeadLayers = [128, 64],
                    representationSize = 256,
                    dropout=0.5
        ):
        super(PEHybridClassifier, self).__init__()
        
        if speakeasyConfig is None:
            speakeasyConfigFile = os.path.join(os.path.dirname(nebula.__file__), "configs", "speakeasyConfig.json")
            with open(speakeasyConfigFile, "r") as f:
                speakeasyConfig = json.load(f)
        if vocab is None:
            vocabFile = os.path.join(os.path.dirname(nebula.__file__), "objects", "speakeasyReportVocab.json")
            with open(vocabFile, "r") as f:
                vocab = json.load(f)

        self.staticExtractor = PEStaticFeatureExtractor()
        # TODO: add configuration for dynamic extractor during init
        self.dynamicExtractor = PEDynamicFeatureExtractor(
            speakeasyConfig=speakeasyConfig,
            emulationOutputFolder=outputFolder,
        )
        self.tokenizer = JSONTokenizer()
        if vocabFile:
            self.tokenizer.load_from_pretrained(vocab)
        else:
            msg = "[!] No 'vocabFile' was provided, the tokenizer has to be pre-trained on your data."
            msg += " Please use the 'build_vocabulary()' method of this class to train the tokenizer."
            logging.warning(msg)
        
        # TODO: add ability to provide config for models
        self.staticModel = MLP(
            layer_sizes=[1024, 512, representationSize],
            dropout=dropout,
        )
        self.dynamicModel = Cnn1DLinearLM(
            vocabSize=len(self.tokenizer.vocab),
            hiddenNeurons=[512, representationSize],
            dropout=dropout,
        )

        # malware head
        layers = []
        for i,ls in enumerate(malwareHeadLayers):
            if i == 0:
                layers.append(nn.Linear(representationSize, ls))
            else:
                layers.append(nn.Linear(malwareHeadLayers[i-1], ls))
            layers.append(nn.LayerNorm(ls))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(malwareHeadLayers[-1], 1))
        self.malware_head = nn.Sequential(*layers)

        if outputFolder and not os.path.exists(outputFolder):
            os.makedirs(outputFolder, exist_ok=True)
        self.outputFolder = outputFolder

    def preprocess(self, data):
        if isinstance(data, str):
            with open(data, "rb") as f:
                bytez = f.read()
        elif isinstance(data, bytes):
            bytez = data
        else:
            raise ValueError("preprocess(): data must be a path to a PE file or a bytes object")
        staticFeatures = self.staticExtractor.feature_vector(bytez).reshape(1,-1)
        dynamicFeaturesJson = self.dynamicExtractor.emulate(data=bytez)
        dynamicFeatures = self.tokenizer.encode(dynamicFeaturesJson)
        return torch.Tensor(staticFeatures), torch.Tensor(dynamicFeatures)
    
    def forward(self, staticFeatures, dynamicFeatures):
        staticFeatures = self.staticModel.get_representations(staticFeatures.float())
        dynamicFeatures = self.dynamicModel.get_representations(dynamicFeatures.long())
        # sum static and dynamic features
        features = staticFeatures + dynamicFeatures
        return self.malware_head(features)

    def build_vocabulary(self, folderBenign, folderMalicious, vocabSize=10000):
        # TODO: training the tokenizer from a set of raw PE files
        raise NotImplementedError("build_vocabulary(): Not implemented yet")

    def train(
        self, 
        folderBenign, 
        folderMalicious, 
        epochs=10
    ):
        # TODO: train classifier using raw PE files in benign an malicious folders
        # Consider how to train static and dynamic models separately to get individual scores
        raise NotImplementedError("train(): Not implemented yet")
