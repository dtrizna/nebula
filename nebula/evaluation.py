import os
import json
import logging
import numpy as np
from collections import defaultdict

from time import sleep, time
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, roc_auc_score

from nebula import ModelTrainer
from nebula.misc import get_tpr_at_fpr, clear_cuda_cache


class SelfSupervisedPretraining:
    def __init__(self, 
                    modelClass,
                    modelConfig,
                    pretrainingTaskClass,
                    pretrainingTaskConfig,
                    device = 'auto',
                    training_types=['pretrained', 'non_pretrained', 'full_data'],
                    falsePositiveRates=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                    unlabeledDataSize=0.8,
                    randomState=None,
                    pretraingEpochs=5,
                    downstreamEpochs=5,
                    batchSize=256,
                    verbosityBatches=100,
                    optimizerStep=5000,
                    outputFolder=None,
                    dump_model_every_epoch=False,
                    dump_data_splits=True,
                    remask_every_epoch=False,
                    downsample_unlabeled_data=False):
        self.modelClass = modelClass
        self.modelConfig = modelConfig
        self.pretrainingTask = pretrainingTaskClass(**pretrainingTaskConfig)
        self.device = device
        self.training_types = training_types
        self.falsePositiveRates = falsePositiveRates
        self.unlabeledDataSize = unlabeledDataSize
        self.randomState = randomState
        self.pretrainingEpochs = pretraingEpochs
        self.downstreamEpochs = downstreamEpochs
        self.batchSize = batchSize
        self.verbosityBatches = verbosityBatches
        self.optimizerStep = optimizerStep
        self.outputFolder = outputFolder
        self.dump_model_every_epoch = dump_model_every_epoch
        self.dump_data_splits = dump_data_splits
        self.remask_every_epoch = remask_every_epoch
        self.downsample_unlabeled_data = downsample_unlabeled_data
        if self.downsample_unlabeled_data:
            assert isinstance(downsample_unlabeled_data, float) and 0 < downsample_unlabeled_data < 1

    def run_one_split(self, x, y, x_test, y_test):
        models = {k: None for k in self.training_types}
        modelInterfaces = {k: None for k in self.training_types}
        metrics = {k: None for k in self.training_types}
        timestamp = int(time())
        
        # split x and y into train and validation sets
        unlabeled_data, labeled_x, _, labeled_y = train_test_split(
            x, y, train_size=self.unlabeledDataSize, random_state=self.randomState
        )
        if self.downsample_unlabeled_data:
            # sample N random samples from unlabeled data which is numpy array
            unlabeled_size = unlabeled_data.shape[0]
            indices = np.random.choice(unlabeled_size, int(self.downsample_unlabeled_data*unlabeled_size), replace=False)
            unlabeled_data = unlabeled_data[indices].copy()
        if self.dump_data_splits:
            splitData = f"dataset_splits_{timestamp}.npz"
            np.savez_compressed(
                os.path.join(self.outputFolder, splitData),
                unlabeled_data=unlabeled_data,
                labeled_x=labeled_x,
                labeled_y=labeled_y
            )
            logging.warning(f" [!] Saved dataset splits to {splitData}")

        logging.warning(' [!] Pre-training model...')
        # create a pretraining model and task
        models['pretrained'] = self.modelClass(**self.modelConfig).to(self.device)
        modelTrainerConfig = {
            "device": self.device,
            "model": models['pretrained'],
            "modelForwardPass": models['pretrained'].pretrain,
            "lossFunction": CrossEntropyLoss(),
            "optimizerClass": Adam,
            "optimizerConfig": {"lr": 2.5e-4},
            "optimSchedulerClass": "step",
            "optim_step_size": self.optimizerStep,
            "verbosityBatches": self.verbosityBatches,
            "batchSize": self.batchSize,
            "falsePositiveRates": self.falsePositiveRates,
        }
        if self.outputFolder:
            modelTrainerConfig["outputFolder"] = os.path.join(self.outputFolder, "preTraining")

        self.pretrainingTask.pretrain(
            unlabeled_data, 
            modelTrainerConfig, 
            pretrainEpochs=self.pretrainingEpochs,
            dump_model_every_epoch=self.dump_model_every_epoch,
            remask_every_epoch=self.remask_every_epoch
        )

        # downstream task for pretrained model
        modelTrainerConfig['lossFunction'] = BCEWithLogitsLoss()
        modelTrainerConfig['modelForwardPass'] = None

        for model in self.training_types:
            logging.warning(f' [!] Training {model} model on downstream task...')
            if model != 'pretrained': # if not pretrained -- create a new model
                models[model] = self.modelClass(**self.modelConfig).to(self.device)
            modelTrainerConfig['model'] = models[model]
            
            if self.outputFolder:
                modelTrainerConfig["outputFolder"] = os.path.join(self.outputFolder, f"downstreamTask_{model}")
            
            modelInterfaces[model] = ModelTrainer(**modelTrainerConfig)
            if model == 'full_data':
                modelTrainerConfig['optim_step_size'] = self.optimizerStep//2
                modelInterfaces[model].fit(x, y, self.downstreamEpochs, reporting_timestamp=timestamp)
            else:
                modelTrainerConfig['optim_step_size'] = self.optimizerStep//10
                modelInterfaces[model].fit(labeled_x, labeled_y, self.downstreamEpochs, reporting_timestamp=timestamp)
        
        for model in models:
            logging.warning(f' [*] Evaluating {model} model on test set...')
            metrics[model] = modelInterfaces[model].evaluate(x_test, y_test, metrics="json")
            logging.warning(f"\t[!] Test set AUC: {metrics[model]['auc']:.4f}")
            for reportingFPR in self.falsePositiveRates:
                f1 = metrics[model]["fpr_"+str(reportingFPR)]["f1"]
                tpr = metrics[model]["fpr_"+str(reportingFPR)]["tpr"]
                logging.warning(f'\t[!] Test set scores at FPR: {reportingFPR:>6} --> TPR: {tpr:.4f} | F1: {f1:.4f}')
        del models, modelInterfaces # cleanup to not accidentaly reuse 
        clear_cuda_cache()
        return metrics

    def run_splits(self, x, y, x_test, y_test, nSplits=5, rest=None):
        metrics = {k: {"fpr_"+str(fpr): {"f1": [], "tpr": []} for fpr in self.falsePositiveRates} for k in self.training_types}
        for trainingType in self.training_types:
            metrics[trainingType]['auc'] = []
        # collect metrics for number of iterations
        for i in range(nSplits):
            self.randomState += i # to get different splits
            logging.warning(f' [!] Running pre-training split {i+1}/{nSplits}')
            splitMetrics = self.run_one_split(x, y, x_test, y_test)
            for trainingType in self.training_types:
                metrics[trainingType]['auc'].append(splitMetrics[trainingType]['auc'])
                for fpr in self.falsePositiveRates:
                    metrics[trainingType]["fpr_"+str(fpr)]["f1"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["f1"])
                    metrics[trainingType]["fpr_"+str(fpr)]["tpr"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["tpr"])
            if rest:
                sleep(rest)

        # compute mean and std for each metric
        for trainingType in self.training_types:
            for fpr in self.falsePositiveRates:
                metrics[trainingType]["fpr_"+str(fpr)]["f1_mean"] = np.nanmean(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["f1_std"] = np.nanstd(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_mean"] = np.nanmean(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_std"] = np.nanstd(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
        return metrics


class CrossValidation(object):
    def __init__(self,
                 modelInterfaceClass,
                 modelInterfaceConfig,
                 modelClass,
                 modelConfig,
                 outputRootFolder,
                 dump_data_splits=True):
        self.modelInterfaceClass = modelInterfaceClass
        self.modelInterfaceConfig = modelInterfaceConfig
        self.modelClass = modelClass
        self.modelConfig = modelConfig
        self.dump_data_splits = dump_data_splits

        self.outputRootFolder = outputRootFolder
        os.makedirs(self.outputRootFolder, exist_ok=True)
        
        self.metrics_val = defaultdict(lambda: defaultdict(list))
        self.aucs_val = []
        self.metrics_train = defaultdict(lambda: defaultdict(list))
        self.aucs_train = []

    def run_folds(self, X, y, folds=3, epochs=5, fprValues=[0.0001, 0.001, 0.01, 0.1], random_state=42):
        """
        Cross validate a model on a dataset and return the mean metrics over all folds depending on specific False Positive Rate (FPR).
        Returns: {fpr1: [mean_tpr, mean_f1], fpr2: [mean_tpr, mean_f1], ...], "avg_epoch_time": N seconds}
        """
        self.fprValues = fprValues
        self.epochs = epochs
        self.trainSize = X.shape[0]
        if folds == 1:
            # kfold requires at least 2 folds -- this is rude shortcut 
            # to get a single run with training on 80%, and validating on 20% of the data
            self.nFolds = 5
            singleFold = True
        else:
            self.nFolds = folds
            singleFold = False

        logging.warning(f" [!] Epochs per fold: {epochs} | Model config: {self.modelConfig}")

        kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=random_state)
        kf.get_n_splits(X)

        # Create the output
        self.training_time = np.empty(shape=(folds, epochs))
        # Iterate over the folds
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            timestamp = int(time())
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            logging.warning(f"\n[!!!] Fold {i+1}/{self.nFolds} | Train set size: {len(X_train)}, Validation set size: {len(X_test)}")
            if self.dump_data_splits:
                splitData = f"dataset_splits_{timestamp}.npz"
                np.savez_compressed(
                    os.path.join(self.outputRootFolder, splitData),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                logging.warning(f" [!] Saved dataset splits to {splitData}")
            # Create the model
            model = self.modelClass(**self.modelConfig)

            modelInterfaceConfig = self.modelInterfaceConfig
            modelInterfaceConfig["model"] = model
            modelInterfaceConfig["outputFolder"] = self.outputRootFolder
            modelTrainer = self.modelInterfaceClass(**modelInterfaceConfig)

            # Train the model
            try:
                modelTrainer.fit(X_train, y_train, epochs=self.epochs, reporting_timestamp=timestamp)
                self.training_time[i,:] = np.pad(modelTrainer.training_time, (0, epochs-len(modelTrainer.training_time)), 'constant', constant_values=(0,0))
            except RuntimeError as e: # cuda outofmemory
                logging.warning(f" [!] Exception: {e}")
                break
            
            # collect stats of this run
            aucs_train, tprs_train, f1_train = self.collect_stats(modelTrainer, X_train, y_train, name="training")
            aucs_val, tprs_val, f1_val = self.collect_stats(modelTrainer, X_test, y_test, name="validation")
            self.aucs_train.append(aucs_train)
            self.aucs_val.append(aucs_val)
            for fpr in self.fprValues:
                self.metrics_train[fpr]["tpr"].append(tprs_train[fpr])
                self.metrics_train[fpr]["f1"].append(f1_train[fpr])
                self.metrics_val[fpr]["tpr"].append(tprs_val[fpr])
                self.metrics_val[fpr]["f1"].append(f1_val[fpr])

            # cleanup
            del model, modelTrainer, modelInterfaceConfig
            clear_cuda_cache()
            if singleFold:
                break
        # calculate average metrics
        self.average_stats()

    def average_stats(self):
        # add AUC to metrics
        self.metrics_val["auc"] = self.aucs_val
        self.metrics_val["auc_avg"] = np.nanmean(self.aucs_val)
        self.metrics_val["auc_std"] = np.nanstd(self.aucs_val)
        self.metrics_train["auc"] = self.aucs_train
        self.metrics_train["auc_avg"] = np.nanmean(self.aucs_train)
        self.metrics_train["auc_std"] = np.nanstd(self.aucs_train)

        # take means over all folds        
        for fpr in self.fprValues:
            self.metrics_val[fpr]["tpr_avg"] = np.nanmean(self.metrics_val[fpr]["tpr"])
            self.metrics_val[fpr]["tpr_std"] = np.nanstd(self.metrics_val[fpr]["tpr"])
            self.metrics_val[fpr]["f1_avg"] = np.nanmean(self.metrics_val[fpr]["f1"])
            self.metrics_val[fpr]["f1_std"] = np.nanstd(self.metrics_val[fpr]["f1"])
            self.metrics_train[fpr]["tpr_avg"] = np.nanmean(self.metrics_train[fpr]["tpr"])
            self.metrics_train[fpr]["tpr_std"] = np.nanstd(self.metrics_train[fpr]["tpr"])
            self.metrics_train[fpr]["f1_avg"] = np.nanmean(self.metrics_train[fpr]["f1"])
            self.metrics_train[fpr]["f1_std"] = np.nanstd(self.metrics_train[fpr]["f1"])

        # Add the average epoch time
        self.metrics_val["epoch_time_avg"] = np.nanmean(self.training_time)

    def collect_stats(self, model_trainer, X, y, name="validation"):
        logging.warning(f"\n [!] Evaluating model on {name} set...")
        probs = model_trainer.predict_proba(X)
        logging.warning(f" [!] This fold metrics on {name} set:")
        # calculate auc 
        auc = roc_auc_score(y, probs)
        logging.warning(f"\tAUC: {auc:.4f}")
        trps, f1s = {}, {}
        for fpr in self.fprValues:
            tpr, threshold = get_tpr_at_fpr(y, probs, fpr)
            f1 = f1_score(y, probs >= threshold)
            trps[fpr], f1s[fpr] = tpr, f1
            logging.warning(f"\tFPR: {fpr} | TPR: {tpr:.4f} | F1: {f1:.4f}")
        return auc, trps, f1s

    def dump_metrics(self, prefix="", suffix=""):
        prefix = prefix.rstrip('_').lstrip('_') + '_' if prefix else ''
        suffix = f"_{suffix.rstrip('_').lstrip('_')}" if suffix else ''
        
        metricFilename = f"{prefix}metrics_validation{suffix}.json"
        metricFileFullpath = os.path.join(self.outputRootFolder, metricFilename)
        with open(metricFileFullpath, "w") as f:
            json.dump(self.metrics_val, f, indent=4)
        logging.warning(f" [!] Metrics saved to {metricFileFullpath}")

        metricFilename = f"{prefix}metrics_training{suffix}.json"
        metricFileFullpath = os.path.join(self.outputRootFolder, metricFilename)
        with open(metricFileFullpath, "w") as f:
            json.dump(self.metrics_train, f, indent=4)
        logging.warning(f" [!] Metrics saved to {metricFileFullpath}")        

    def log_avg_metrics(self):
        msg = f" [!] Average epoch time: {self.metrics_val['epoch_time_avg']:.2f}s | Mean values over {self.nFolds} folds:\n"
        msg += f"\tAUC: {self.metrics_val['auc_avg']:.4f}\n"
        for fpr in self.fprValues:
            msg += f"\tFPR: {fpr:>6} -- TPR: {self.metrics_val[fpr]['tpr_avg']:.4f} -- F1: {self.metrics_val[fpr]['f1_avg']:.4f}\n"
        logging.warning(msg)
