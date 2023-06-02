import nebula
from nebula.preprocessing import JSONTokenizerBPE, PEDynamicFeatureExtractor, PEStaticFeatureExtractor
from nebula.optimization import OptimSchedulerGPT, OptimSchedulerStep, CosineSchedule, TriangularSchedule, OneCycleSchedule
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
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Linear
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

class ModelTrainer(object):
    def __init__(self, 
                    model,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                    loss_function=BCEWithLogitsLoss(),
                    optimizer_class=AdamW,
                    optimizer_config={"lr": 2.5e-4, "weight_decay": 1e-2},
                    optim_scheduler=None,
                    batchSize=64,
                    verbosity_n_batches=100,
                    outputFolder=None,
                    falsePositiveRates=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                    modelForwardPass=None,
                    stateDict = None,
                    timestamp = None,
                    clip_grad_norm = 0.5,
                    optim_step_budget = 1000,
                    time_budget = None,
                    n_batches_grad_update = 1,
                    n_output_classes = None):
        self.model = model 
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_config)
        self.loss_function = loss_function
        self.verbosity_n_batches = verbosity_n_batches
        self.batch_size = batchSize
        self.reporting_timestamp = timestamp
        self.clip_grad_norm = clip_grad_norm
        self.global_batch_idx = 0
        self.n_batches_grad_update = n_batches_grad_update
        if n_output_classes is not None:
            self.n_output_classes = n_output_classes
        else:
            self.n_output_classes = [x for x in self.model.children() if isinstance(x, Linear)][-1].out_features

        # lr scheduling setup, for visulaizations see:
        # https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
        if optim_scheduler is None:
            self.optim_scheduler = None
        elif optim_scheduler == "gpt":
            # TODO: broken
            self.optim_scheduler = OptimSchedulerGPT(
                self.optimizer, 
                max_lr=optimizer_config['lr'],
                half_cycle_batches=optim_step_budget//2
            )
        elif optim_scheduler == "step":
            nr_of_steps = 3
            self.optim_scheduler = OptimSchedulerStep(
                self.optimizer, 
                step_size=optim_step_budget//nr_of_steps
            )
        elif optim_scheduler == "triangular":
            self.optim_scheduler = TriangularSchedule(
                self.optimizer,
                base_lr=optimizer_config['lr']/10,
                max_lr=optimizer_config['lr'],
                step_size_up=optim_step_budget//2
            )
        elif optim_scheduler == "onecycle":
            self.optim_scheduler = OneCycleSchedule(
                self.optimizer,
                max_lr=optimizer_config['lr'],
                total_steps=optim_step_budget
            )
        elif optim_scheduler == "cosine":
            self.optim_scheduler = CosineSchedule(
                self.optimizer,
                T_max=optim_step_budget,
                eta_min=optimizer_config['lr']/10
            )
        else:
            self.optim_scheduler = optim_scheduler(self.optimizer)
        
        self.learning_rates = []
        self.time_budget = time_budget

        self.fp_rates = falsePositiveRates
        self.fp_reporting_idx = 1 # what FPR stats to report every epoch

        self.device = device
        self.model.to(self.device)

        self.output_folder = outputFolder
        if self.output_folder is not None:
            self.trainingFileFolder = os.path.join(self.output_folder, "training_files")
            os.makedirs(self.output_folder, exist_ok=True)
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

            if self.learning_rates:
                np.save(f"{prefix}-learning_rates.npy", np.array(self.learning_rates))
                dumpString += f"\n\t\tlearning rates: {os.path.basename(prefix)}-learning_rates.npy"
        logging.warning(dumpString)
    
    def getMetrics(self, true_labels, predicted_probs):
        assert isinstance(true_labels, np.ndarray) and isinstance(predicted_probs, np.ndarray)
        assert len(true_labels) == len(predicted_probs)
        metrics = []
        fprs, tprs, thresholds = roc_curve(true_labels, predicted_probs)
        for fpr in self.fp_rates:
            tpr = tprs[fprs <= fpr][-1]
            threshold = thresholds[fprs <= fpr][-1]
            f1_at_fpr = f1_score(true_labels, predicted_probs >= threshold)
            metrics.append([tpr, f1_at_fpr])
        return np.array(metrics)

    def updateMetricsReportList(self, metricsReportList, tprs=None, f1s=None, batch_metrics=None, auc=None):
        if tprs is not None and f1s is not None:
            tpr = tprs[self.fp_reporting_idx]
            f1 = f1s[self.fp_reporting_idx]
            metricsReport = f"FPR {self.fp_rates[self.fp_reporting_idx]} -> "
            metricsReport += f"TPR: {tpr:.2f} & F1: {f1:.2f}"
            metricsReportList.append(metricsReport)
        if auc is not None:
            metricsReportList.append(f"AUC: {auc:.4f}")
        return metricsReportList

    def logBatchMetrics(self, loss, targets, probs,):
        metricsReportList = [f"Loss: {loss.item():.6f}", f"Elapsed: {time.time() - self.tick:.2f}s"]

        # report TPR | F1 | AUC only for binary classification
        if self.n_output_classes == 1:

            # metrics for set of batches since last reporting
            batchMetrics = self.getMetrics(
                np.array(targets[-self.verbosity_n_batches:]), 
                np.array(probs[-self.verbosity_n_batches:])
            )
            batch_tprs, batch_f1s = batchMetrics[:,0], batchMetrics[:,1]

            try: # calculate auc for last set of batches
                batch_auc = roc_auc_score(
                    targets[-self.verbosity_n_batches:], 
                    probs[-self.verbosity_n_batches:]
                )
            # batch might contain only one class, then auc fails
            except ValueError:
                batch_auc = np.nan

            metricsReportList = self.updateMetricsReportList(
                metricsReportList,
                tprs=batch_tprs,
                f1s=batch_f1s,
                auc=batch_auc
            )
        else: # multiclass
            # TODO: implement multiclass roc
            pass
        msg = " [*] {}: Train Epoch: {} [{:^5}/{:^5} ({:^2.0f}%)] | ".format(
            time.ctime().split()[3],
            self.epoch_idx,
            self.batch_idx,
            self.trainset_size,
            100. * self.batch_idx / self.trainset_size
        )
        msg += " | ".join(metricsReportList)
        logging.warning(msg)
        self.tick = time.time()

    def train_one_epoch(self, train_loader):
        epoch_losses = []
        epochProbs = []
        targets = []
        self.tick = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            self.batch_idx = batch_idx
            targets.extend(target)
            data, target = data.to(self.device), target.to(self.device).float()

            logits = self.model.forwardPass(data).float()

            if isinstance(self.loss_function, BCEWithLogitsLoss) and target.dim() == 1:
                target = target.reshape(-1, 1)
            elif isinstance(self.loss_function, CrossEntropyLoss) and target.dim() != 1:
                target = target.squeeze()
            
            try:
                loss = self.loss_function(logits, target)
            except RuntimeError:
                # NOTE: might need to transform to .long() for CrossEntropyLoss if this error occurs:
                # "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'
                target = target.long()
                loss = self.loss_function(logits, target)

            loss.backward() # derivatives
            epoch_losses.append(loss.item())

            # +1 to skip update at first batch since not yet accumulated gradients
            if ((batch_idx + 1) % self.n_batches_grad_update == 0) or \
                (batch_idx + 1 == len(train_loader)): # to update at last batch

                if self.clip_grad_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # learning rate update
            self.global_batch_idx += 1
            if self.optim_scheduler is not None:
                self.optim_scheduler.step(self.global_batch_idx)
                self.learning_rates.append(self.optim_scheduler.get_lr())
            
            predProbs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            epochProbs.extend(predProbs)
            
            if batch_idx % self.verbosity_n_batches == 0:
                self.logBatchMetrics(loss, targets, epochProbs)
            
            if self.time_budget is not None and (time.time() - self.training_start_time) > self.time_budget:
                logging.warning(" [!] Time budget exceeded, training stopped.")
                raise KeyboardInterrupt

        if self.n_output_classes == 1: # calculate TRP & F1 only for binary classification
            epochMetrics = self.getMetrics(np.array(targets), np.array(epochProbs))
            epoch_tprs, epoch_f1s = epochMetrics[:,0], epochMetrics[:,1]
            # calculate auc 
            epoch_auc = roc_auc_score(targets, epochProbs)
            return epoch_losses, epoch_tprs, epoch_f1s, epoch_auc
        else:
            return epoch_losses, None, None, None

    def fit(
            self,
            X,
            y,
            epochs=10,
            time_budget=None,
            dump_model_every_epoch=False,
            overwrite_epoch_idx=False,
            reporting_timestamp=None
    ):
        assert (epochs is not None) ^ (time_budget is not None), \
            "Either epochs or time_budget should be specified"
        if time_budget or self.time_budget:
            self.time_budget = time_budget if time_budget else self.time_budget
            assert isinstance(self.time_budget, int), "time_budget must be an integer"
            self.training_start_time = time.time()
            logging.warning(f" [*] Training time budget set: {self.time_budget/60} min")
        self.epochs = int(1e4) if self.time_budget else epochs

        if reporting_timestamp:
            self.reporting_timestamp = reporting_timestamp

        trainLoader = DataLoader(
            TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.trainset_size = len(trainLoader)

        self.model.train()
        self.train_tprs = np.empty(shape=(self.epochs, len(self.fp_rates)))
        self.train_f1s = np.empty(shape=(self.epochs, len(self.fp_rates)))
        self.train_losses = []
        self.training_time = []
        self.auc = []
        try:
            for epoch_idx in range(1, epochs + 1):
                if overwrite_epoch_idx:
                    assert isinstance(overwrite_epoch_idx, int), "overwrite_epoch_idx must be an integer"
                    epoch_idx = overwrite_epoch_idx + epoch_idx
                self.epoch_idx = epoch_idx

                epochStartTime = time.time()
                logging.warning(f" [*] Started epoch: {epoch_idx}")
                epoch_train_loss, epoch_tprs, epoch_f1s, epoch_auc = self.train_one_epoch(trainLoader)

                if overwrite_epoch_idx:
                    self.train_tprs[epoch_idx-1-overwrite_epoch_idx] = epoch_tprs
                    self.train_f1s[epoch_idx-1-overwrite_epoch_idx] = epoch_f1s
                else:
                    self.train_tprs[epoch_idx-1] = epoch_tprs
                    self.train_f1s[epoch_idx-1] = epoch_f1s

                self.train_losses.extend(epoch_train_loss)
                self.auc.append(epoch_auc)
                timeElapsed = time.time() - epochStartTime
                self.training_time.append(timeElapsed)
                
                metricsReportList = [f"{epoch_idx:^7}", f"Tr.loss: {np.nanmean(epoch_train_loss):.6f}", f"Elapsed: {timeElapsed:^9.2f}s"]
                metricsReportList = self.updateMetricsReportList(
                    metricsReportList,
                    tprs=epoch_tprs,
                    f1s=epoch_f1s,
                    auc=epoch_auc
                )
                logging.warning(f" [*] {time.ctime()}: " + " | ".join(metricsReportList))
                if dump_model_every_epoch:
                    self.dumpResults(prefix=f"epoch_{epoch_idx}_", model_only=True)
            if self.output_folder:
                self.dumpResults()
        except KeyboardInterrupt:
            if self.output_folder:
                self.dumpResults()

    def train(self, X, y, epochs=10, time_budget=None, dump_model_every_epoch=False, overwrite_epoch_idx=False, reporting_timestamp=None):
        self.fit(
            X, y, 
            epochs=epochs, 
            time_budget=time_budget, 
            dump_model_every_epoch=dump_model_every_epoch, 
            overwrite_epoch_idx=overwrite_epoch_idx, 
            reporting_timestamp=reporting_timestamp
        )

    def evaluate(self, X, y, metrics="array"):
        testLoader = DataLoader(
            TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float()),
            batch_size=self.batch_size,
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

            loss = self.loss_function(logits, target.float().reshape(-1,1))
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
        out = torch.empty((0,self.n_output_classes)).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(arr).long()),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        for batch_idx, data in enumerate(loader):
            with torch.no_grad():
                logits = self.model(torch.Tensor(data[0]).long().to(self.device))
                out = torch.vstack([out, logits])
            if batch_idx % self.verbosity_n_batches == 0:
                print(f" [*] Predicting batch: {batch_idx}/{len(loader)}", end="\r")
        if self.n_output_classes == 1:
            return torch.sigmoid(out).clone().detach().cpu().numpy()
        else:
            return torch.softmax(out, axis=1).clone().detach().cpu().numpy()
    
    def predict(self, arr, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            prob = self.model.predict_proba(arr)
        return (prob > threshold).astype(np.int8)
    
    def metricsToJSON(self, metrics):
        assert len(metrics) == 4, "metricsToJSON(): metrics must be a list of length 4"
        tprs = dict(zip(["fpr_"+str(x) for x in self.fp_rates], metrics[1]))
        f1s = dict(zip(["fpr_"+str(x) for x in self.fp_rates], metrics[2]))
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
            speakeasyConfigFile = os.path.join(os.path.dirname(nebula.__file__), "objects", "speakeasy_config.json")
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
