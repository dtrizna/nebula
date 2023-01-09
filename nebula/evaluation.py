import os
import json
import logging
import numpy as np
from collections import defaultdict
from torch import cuda
import gc

from time import sleep
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score

from nebula import ModelTrainer
from nebula.misc import dictToString, get_tpr_at_fpr


class SelfSupervisedPretraining:
    def __init__(self, 
                    vocab,
                    modelClass,
                    modelConfig,
                    pretrainingTaskClass,
                    pretrainingTaskConfig,
                    device = 'auto',
                    falsePositiveRates=[0.001, 0.003, 0.01, 0.03, 0.1],
                    unlabeledDataSize=0.8,
                    randomState=None,
                    pretraingEpochs=5,
                    downstreamEpochs=5,
                    batchSize=256,
                    verbosityBatches=100):
        self.vocab = vocab
        self.modelClass = modelClass
        self.modelConfig = modelConfig
        self.falsePositiveRates = falsePositiveRates
        self.unlabeledDataSize = unlabeledDataSize
        self.randomState = randomState
        self.batchSize = batchSize
        self.pretrainingEpochs = pretraingEpochs
        self.downstreamEpochs = downstreamEpochs
        self.device = device
        self.verbosityBatches = verbosityBatches
        self.pretrainingTask = pretrainingTaskClass(**pretrainingTaskConfig)

        self.trainingTypes = ['pretrained', 'non_pretrained', 'full_data']

    def run(self, x, y, x_test, y_test, outputFolder=None):
        models = {k: None for k in self.trainingTypes}
        modelInterfaces = {k: None for k in self.trainingTypes}
        metrics = {k: None for k in self.trainingTypes}
        
        # split x and y into train and validation sets
        U, L_x, _, L_y = train_test_split(x, y, train_size=self.unlabeledDataSize, random_state=self.randomState)

        # create a pretraining model and task
        logging.warning(' [!] Pre-training model...')
        models['pretrained'] = self.modelClass(**self.modelConfig).to(self.device)
        modelInterfaceConfig = {
            "device": self.device,
            "model": models['pretrained'],
            "modelForwardPass": models['pretrained'].pretrain,
            "lossFunction": CrossEntropyLoss(),
            "optimizerClass": Adam,
            "optimizerConfig": {"lr": 0.001},
            "verbosityBatches": self.verbosityBatches,
            "batchSize": self.batchSize,
            "falsePositiveRates": self.falsePositiveRates,
        }
        if outputFolder:
            modelInterfaceConfig["outputFolder"] = os.path.join(outputFolder, "preTraining")

        self.pretrainingTask.pretrain(U, modelInterfaceConfig, pretrainEpochs=self.pretrainingEpochs)
        
        # downstream task for pretrained model
        modelInterfaceConfig['lossFunction'] = BCEWithLogitsLoss()
        modelInterfaceConfig['modelForwardPass'] = None

        for model in self.trainingTypes:
            logging.warning(f' [!] Training {model} model on downstream task...')
            if model != 'pretrained':
                models[model] = self.modelClass(**self.modelConfig).to(self.device)
                modelInterfaceConfig['model'] = models[model]
            
            if outputFolder:
                modelInterfaceConfig["outputFolder"] = os.path.join(outputFolder, "downstreamTask_{}".format(model))
            
            modelInterfaces[model] = ModelTrainer(**modelInterfaceConfig)
            if model == 'full_data':
                modelInterfaces[model].fit(x, y, self.downstreamEpochs)
            else:
                modelInterfaces[model].fit(L_x, L_y, self.downstreamEpochs)
        
        for model in models:
            logging.warning(f' [*] Evaluating {model} model on test set...')
            metrics[model] = modelInterfaces[model].evaluate(x_test, y_test, metrics="json")
            reportingFPR = self.falsePositiveRates[modelInterfaces[model].fprReportingIdx]
            logging.warning(f' [!] Test F1 score for {model} model at {reportingFPR} FPR : {metrics[model]["fpr_"+str(reportingFPR)]["f1"]:.4f}')
            logging.warning(f' [!] Test TPR score for {model} model at {reportingFPR} FPR: {metrics[model]["fpr_"+str(reportingFPR)]["tpr"]:.4f}')
        del models, modelInterfaces # cleanup to not accidentaly reuse 
        return metrics

    def runSplits(self, x, y, x_test, y_test, outputFolder=None, nSplits=5, rest=None):
        metrics = {k: {"fpr_"+str(fpr): {"f1": [], "tpr": []} for fpr in self.falsePositiveRates} for k in self.trainingTypes}
        # collect metrics for number of iterations
        for i in range(nSplits):
            self.randomState += i # to get different splits
            logging.warning(f' [!] Running pre-training split {i+1}/{nSplits}')
            splitMetrics = self.run(x, y, x_test, y_test, outputFolder=outputFolder)
            for trainingType in self.trainingTypes:
                for fpr in self.falsePositiveRates:
                    metrics[trainingType]["fpr_"+str(fpr)]["f1"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["f1"])
                    metrics[trainingType]["fpr_"+str(fpr)]["tpr"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["tpr"])
            if rest:
                sleep(rest)

        # compute mean and std for each metric
        for trainingType in self.trainingTypes:
            for fpr in self.falsePositiveRates:
                metrics[trainingType]["fpr_"+str(fpr)]["f1_mean"] = np.mean(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["f1_std"] = np.std(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_mean"] = np.mean(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_std"] = np.std(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
        return metrics


class CrossValidation(object):
    def __init__(self,
                 modelInterfaceClass,
                 modelInterfaceConfig,
                 modelClass,
                 modelConfig,
                 outputRootFolder):
        self.modelInterfaceClass = modelInterfaceClass
        self.modelInterfaceConfig = modelInterfaceConfig
        self.modelClass = modelClass
        self.modelConfig = modelConfig

        self.outputRootFolder = outputRootFolder
        os.makedirs(self.outputRootFolder, exist_ok=True)
        self.outputFolderTrainingFiles = os.path.join(self.outputRootFolder, "trainingFiles")

        self.metrics = defaultdict(lambda: defaultdict(list))

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
            # will split 2/3 trainig set; 1/3 test set and run only one fold
            self.nFolds = 3
            singleFold = True
        else:
            self.nFolds = folds
            singleFold = False
        # Create the folds
        logging.warning(f" [!] Epochs per fold: {epochs} | Model config: {self.modelConfig}")

        kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=random_state)
        kf.get_n_splits(X)

        # Create the output
        trainingTime = np.empty(shape=(folds, epochs))
        # Iterate over the folds
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            logging.warning(f" [!] Fold {i+1}/{self.nFolds} | Train set size: {len(X_train)}, Validation set size: {len(X_test)}")
            # Create the model
            model = self.modelClass(**self.modelConfig)

            modelInterfaceConfig = self.modelInterfaceConfig
            modelInterfaceConfig["model"] = model
            modelInterfaceConfig["outputFolder"] = self.outputFolderTrainingFiles
            modelTrainer = self.modelInterfaceClass(**modelInterfaceConfig)

            # Train the model
            try:
                modelTrainer.fit(X_train, y_train, epochs=self.epochs)
                trainingTime[i,:] = np.pad(modelTrainer.trainingTime, (0, epochs-len(modelTrainer.trainingTime)), 'constant', constant_values=(0,0))
            except RuntimeError as e: # usually cuda outofmemory
                logging.warning(f" [!] Exception: {e}")
                break

            # Evaluate the model
            logging.warning(f" [!] Evaluating model on validation set...")
            predicted_probs = modelTrainer.predict_proba(X_test)
            logging.warning(f" [!] This fold metrics on validation set:")
            for fpr in self.fprValues:
                tpr, threshold = get_tpr_at_fpr(y_test, predicted_probs, fpr)
                f1 = f1_score(y[test_index], predicted_probs >= threshold)
                self.metrics[fpr]["tpr"].append(tpr)
                self.metrics[fpr]["f1"].append(f1)       
                logging.warning(f" [!] FPR: {fpr} | TPR: {tpr} | F1: {f1}")
            
            # cleanup
            del model, modelTrainer, modelInterfaceConfig
            cuda.empty_cache()
            gc.collect()

            if singleFold:
                break

        # take means over all folds        
        for fpr in self.fprValues:
            self.metrics[fpr]["tpr_avg"] = np.mean(self.metrics[fpr]["tpr"])
            self.metrics[fpr]["f1_avg"] = np.mean(self.metrics[fpr]["f1"])

        # Add the average epoch time
        self.metrics["epoch_time_avg"] = np.mean(trainingTime)

    def dump_metrics(self, prefix="", suffix=""):
        configStr = dictToString(self.modelConfig)
        metricFilename = f"metrics_trainSize_{self.trainSize}_ep_{self.epochs}_cv_{self.nFolds}_{prefix}{configStr}{suffix}.json"
        metricFileFullpath = os.path.join(self.outputRootFolder, metricFilename)
        with open(metricFileFullpath, "w") as f:
            json.dump(self.metrics, f, indent=4)
        logging.warning(f" [!] Metrics saved to {metricFileFullpath}")

    def log_avg_metrics(self):
        msg = f" [!] Average epoch time: {self.metrics['epoch_time_avg']:.2f}s | Mean values over {self.nFolds} folds:\n"
        for fpr in self.fprValues:
            msg += f"\tFPR: {fpr:>6} -- TPR: {self.metrics[fpr]['tpr_avg']:.4f} -- F1: {self.metrics[fpr]['f1_avg']:.4f}\n"
        logging.warning(msg)
