import nebula
from nebula.preprocessing import JSONTokenizerBPE, PEDynamicFeatureExtractor, PEStaticFeatureExtractor
from nebula.optimization import OptimSchedulerGPT, OptimSchedulerStep
from nebula.models import Cnn1DLinearLM, MLP

import os
import time
import json
import logging
import numpy as np
from pandas import DataFrame

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from sklearn.metrics import f1_score, roc_auc_score
from nebula.misc import get_tpr_at_fpr
from tqdm import tqdm

class ModelTrainer(object):
    def __init__(self, 
                    model,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                    lossFunction=BCEWithLogitsLoss(),
                    optimizerClass=AdamW,
                    optimizerConfig={"lr": 2.5e-4},
                    optimSchedulerClass=None,
                    batchSize=64,
                    verbosityBatches=100,
                    outputFolder=None,
                    falsePositiveRates=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                    modelForwardPass=None,
                    stateDict = None,
                    timestamp = None,
                    optim_step_size=1000):
        self.model = model    
        self.optimizer = optimizerClass(self.model.parameters(), **optimizerConfig)
        self.lossFunction = lossFunction
        self.verbosityBatches = verbosityBatches
        self.batchSize = batchSize
        self.reporting_timestamp = timestamp

        # lr scheduling setup
        if optimSchedulerClass is None:
            self.optimScheduler = None
        elif optimSchedulerClass == "gpt":
            self.optimScheduler = OptimSchedulerGPT(self.optimizer)
        elif optimSchedulerClass == "step":
            if optim_step_size:
                self.optimScheduler = OptimSchedulerStep(self.optimizer, step_size=optim_step_size)
            else:
                self.optimScheduler = None
        else:
            self.optimScheduler = optimSchedulerClass(self.optimizer)

        self.falsePositiveRates = falsePositiveRates
        self.fprReportingIdx = 1 # what FPR stats to report every epoch

        self.device = device
        self.model.to(self.device)

        self.outputFolder = outputFolder
        if self.outputFolder is not None:
            self.trainingFileFolder = os.path.join(self.outputFolder, "training_files")
            os.makedirs(self.outputFolder, exist_ok=True)
            os.makedirs(self.trainingFileFolder, exist_ok=True)
            
        if stateDict:
            self.model.load_state_dict(torch.load(stateDict))
        if modelForwardPass:
            self.model.forwardPass = modelForwardPass
        else:
            self.model.forwardPass = self.model.forward
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # if .name in self.model attributes
        if not hasattr(self.model, "__name__"):
            self.model.__name__ = "model"
        logging.warning(f" [!] Iniatialized {self.model.__name__}. Total trainable parameters: {trainable_params/1e6:.4f}e6")

    def loadState(self, stateDict):
        self.model.load_state_dict(torch.load(stateDict))

    def dumpResults(self, prefix="", model_only=False):
        if self.reporting_timestamp:
            prefix = os.path.join(self.trainingFileFolder, f"{prefix}{self.reporting_timestamp}")
        else:
            prefix = os.path.join(self.trainingFileFolder, f"{prefix}{int(time.time())}")
        
        modelFile = f"{prefix}-model.torch"
        torch.save(self.model.state_dict(), modelFile)
        dumpString = f""" [!] {time.ctime()}: Dumped results:
                model       : {os.path.basename(modelFile)}"""
        if not model_only:
            np.save(f"{prefix}-trainTime.npy", np.array(self.training_time))
            dumpString += f"\n\t\ttrain time  : {os.path.basename(prefix)}-trainTime.npy"

            np.save(f"{prefix}-trainLosses.npy", np.array(self.train_losses))
            dumpString += f"\n\t\ttrain losses: {os.path.basename(prefix)}-trainLosses.npy"

            np.save(f"{prefix}-auc.npy", np.array(self.auc))
            dumpString += f"\n\t\ttrain AUC   : {os.path.basename(prefix)}-auc.npy"

            if not np.isnan(self.train_f1s).all():
                np.save(f"{prefix}-trainF1s.npy", self.train_f1s)
                dumpString += f"\n\t\ttrain F1s   : {os.path.basename(prefix)}-trainF1s.npy"
            if not np.isnan(self.train_tprs).all():
                np.save(f"{prefix}-trainTPRs.npy", self.train_tprs)
                dumpString += f"\n\t\ttrain TPRs  : {os.path.basename(prefix)}-trainTPRs.npy"
        logging.warning(dumpString)
    
    def getMetrics(self, target, predProbs):
        assert isinstance(target, np.ndarray) and isinstance(predProbs, np.ndarray)
        metrics = []
        for fpr in self.falsePositiveRates:
            tpr, threshold = get_tpr_at_fpr(target, predProbs, fpr)
            f1_at_fpr = f1_score(target, predProbs >= threshold)
            metrics.append([tpr, f1_at_fpr])
        return np.array(metrics)

    def trainOneEpoch(self, trainLoader, epochId):
        epoch_losses = []
        epochProbs = []
        targets = []
        now = time.time()

        for batchIdx, (data, target) in enumerate(trainLoader):
            targets.extend(target)
            data, target = data.to(self.device), target.to(self.device)

            logits = self.model.forwardPass(data)
            # if dimension of target is 1, reshape to (-1,1)
            if target.dim() == 1:
                target = target.reshape(-1, 1)
            loss = self.lossFunction(logits, target)

            epoch_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward() # derivatives
            self.optimizer.step() # parameter update
            
            # learning rate update
            self.globalBatchCounter += 1
            if self.optimScheduler is not None:
                self.optimScheduler.step(self.globalBatchCounter)

            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            epochProbs.extend(predProbs)
            
            if batchIdx % self.verbosityBatches == 0:
                metricsReportList = [f"Loss: {loss.item():.6f}", f"Elapsed: {time.time() - now:.2f}s"]
                if target.shape[1] == 1: # report TPR & F1 only for binary classification
                    batchSetMetrics = self.getMetrics(
                        np.array(targets[-self.verbosityBatches:]), 
                        np.array(epochProbs[-self.verbosityBatches:])
                    )
                    currentTPR = batchSetMetrics[self.fprReportingIdx, 0]
                    currentF1 = batchSetMetrics[self.fprReportingIdx, 1]
                    metricsReport = f"FPR {self.falsePositiveRates[self.fprReportingIdx]} -> "
                    metricsReport += f"TPR {currentTPR:.4f} & F1 {currentF1:.4f}"
                    metricsReportList.append(metricsReport)
                    try: # calculate auc for last set of batches
                        batch_auc = roc_auc_score(
                            targets[-self.verbosityBatches:], 
                            epochProbs[-self.verbosityBatches:]
                        )
                    # Only one class present in y_true. ROC AUC score is not defined in that case.
                    except ValueError:
                        batch_auc = np.nan
                    metricsReportList.append(f"AUC {batch_auc:.4f}")
                logging.warning(" [*] {}: Train Epoch: {} [{:^5}/{:^5} ({:^2.0f}%)] | ".format(
                    time.ctime().split()[3], epochId, batchIdx * len(data), len(trainLoader.dataset),
                100. * batchIdx / len(trainLoader)) + " | ".join(metricsReportList))
                now = time.time()

        if target.shape[1] == 1: # calculate TRP & F1 only for binary classification
            epochMetrics = self.getMetrics(np.array(targets), np.array(epochProbs))
            epoch_tprs, epoch_f1s = epochMetrics[:,0], epochMetrics[:,1]
            # calculate auc 
            epoch_auc = roc_auc_score(targets, epochProbs)
            return epoch_losses, epoch_tprs, epoch_f1s, epoch_auc
        else:
            return epoch_losses, None, None, None

    def fit(self, X, y, epochs=10, dump_model_every_epoch=False, overwrite_epoch_idx=False, reporting_timestamp=None):
        if reporting_timestamp:
            self.reporting_timestamp = reporting_timestamp

        trainLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batchSize,
            shuffle=True
        )
        self.model.train()
        self.train_tprs = np.empty(shape=(epochs, len(self.falsePositiveRates)))
        self.train_f1s = np.empty(shape=(epochs, len(self.falsePositiveRates)))
        self.train_losses = []
        self.training_time = []
        self.auc = []
        self.globalBatchCounter = 0
        try:
            for epochIdx in range(1, epochs + 1):
                if overwrite_epoch_idx:
                    assert isinstance(overwrite_epoch_idx, int), "overwrite_epoch_idx must be an integer"
                    epochIdx = overwrite_epoch_idx + epochIdx

                epochStartTime = time.time()
                logging.warning(f" [*] Started epoch: {epochIdx}")
                epoch_train_loss, epoch_tprs, epoch_f1s, epoch_auc = self.trainOneEpoch(trainLoader, epochIdx)

                if overwrite_epoch_idx:
                    self.train_tprs[epochIdx-1-overwrite_epoch_idx] = epoch_tprs
                    self.train_f1s[epochIdx-1-overwrite_epoch_idx] = epoch_f1s
                else:
                    self.train_tprs[epochIdx-1] = epoch_tprs
                    self.train_f1s[epochIdx-1] = epoch_f1s

                self.train_losses.extend(epoch_train_loss)
                self.auc.append(epoch_auc)
                timeElapsed = time.time() - epochStartTime
                self.training_time.append(timeElapsed)
                
                metricsReportList = [f"{epochIdx:^7}", f"Tr.loss: {np.nanmean(epoch_train_loss):.6f}", f"Elapsed: {timeElapsed:^9.2f}s"]
                if epoch_tprs is not None and epoch_f1s is not None and epoch_auc is not None:
                    metricsReport = f"FPR {self.falsePositiveRates[self.fprReportingIdx]} -> "
                    metricsReport += f"TPR: {epoch_tprs[self.fprReportingIdx]:.2f} & F1: {epoch_f1s[self.fprReportingIdx]:.2f}"
                    metricsReportList.append(metricsReport)
                    metricsReportList.append(f"AUC: {epoch_auc:.4f}")
                logging.warning(f" [*] {time.ctime()}: " + " | ".join(metricsReportList))
                if dump_model_every_epoch:
                    self.dumpResults(prefix=f"epoch_{epochIdx}_", model_only=True)
            if self.outputFolder:
                self.dumpResults()
        except KeyboardInterrupt:
            if self.outputFolder:
                self.dumpResults()

    def train(self, X, y, epochs=10):
        self.fit(X, y, epochs=epochs)

    def evaluate(self, X, y, metrics="array"):
        testLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batchSize,
            shuffle=True
        )
        self.model.eval()

        self.testLoss = []
        self.predProbs = []
        self.trueLabels = []
        for data, target in tqdm(testLoader):
            self.trueLabels.extend(target.cpu().numpy())
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                logits = self.model.forwardPass(data)

            loss = self.lossFunction(logits, target.float().reshape(-1,1))
            self.testLoss.append(loss.item())

            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            self.predProbs.extend(predProbs)
        
        metricsValues = self.getMetrics(np.array(self.trueLabels), np.array(self.predProbs))
        self.test_tprs, self.test_f1s = metricsValues[:,0], metricsValues[:,1]
        self.test_auc = roc_auc_score(self.trueLabels, self.predProbs)
        if metrics == "array":
            return self.testLoss, self.test_tprs, self.test_f1s, self.test_auc
        elif metrics == "json":
            return self.metricsToJSON([self.testLoss, self.test_tprs, self.test_f1s, self.test_auc])
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
        for batch_idx, data in enumerate(loader):
            with torch.no_grad():
                logits = self.model(torch.Tensor(data[0]).long().to(self.device))
                out = torch.vstack([out, logits])
            if batch_idx % self.verbosityBatches == 0:
                print(f" [*] Predicting batch: {batch_idx}/{len(loader)}", end="\r")

        return torch.sigmoid(out).clone().detach().cpu().numpy()
    
    def predict(self, arr, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.predict_proba(arr)
        return (logits > threshold).astype(np.int_)
    
    def metricsToJSON(self, metrics):
        assert len(metrics) == 4, "metricsToJSON(): metrics must be a list of length 4"
        tprs = dict(zip(["fpr_"+str(x) for x in self.falsePositiveRates], metrics[1]))
        f1s = dict(zip(["fpr_"+str(x) for x in self.falsePositiveRates], metrics[2]))
        metricDict = DataFrame([tprs, f1s], index=["tpr", "f1"]).to_dict()
        metricDict['loss'] = metrics[0]
        metricDict['auc'] = metrics[3]
        return metricDict


class PEHybridClassifier(nn.Module):
    def __init__(self,
                    speakeasyConfig=None,
                    vocab=None,
                    tokenizerModelPath=None,
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
            vocabFile = os.path.join(os.path.dirname(nebula.__file__), "objects", "speakeasy_BPE_50000_vocab.json")
            with open(vocabFile, "r") as f:
                vocab = json.load(f)
        if tokenizerModelPath is None:
            tokenizerModelPath = os.path.join(os.path.dirname(nebula.__file__), "objects", "speakeasy_BPE_50000.model")
            self.tokenizer = JSONTokenizerBPE(model_path=tokenizerModelPath)
            self.tokenizer.load_vocab()

        self.staticExtractor = PEStaticFeatureExtractor()
        # TODO: add configuration for dynamic extractor during init
        self.dynamicExtractor = PEDynamicFeatureExtractor(
            speakeasyConfig=speakeasyConfig,
            emulationOutputFolder=outputFolder,
        )
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
