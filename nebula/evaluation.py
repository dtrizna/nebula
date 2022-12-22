from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, f1_score
from collections import defaultdict
import numpy as np
import json
import os
import logging
from nebula.misc import dictToString

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

    @staticmethod
    def get_tpr_at_fpr(predicted_probs, true_labels, fprNeeded):
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
        tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
        threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
        return tpr_at_fpr, threshold_at_fpr

    def run_folds(self, X, y, folds=3, epochs=5, fprValues=[0.0001, 0.001, 0.01, 0.1], random_state=42):
        """
        Cross validate a model on a dataset and return the mean metrics over all folds depending on specific False Positive Rate (FPR).
        Returns: {fpr1: [mean_tpr, mean_f1], fpr2: [mean_tpr, mean_f1], ...], "avg_epoch_time": N seconds}
        """
        self.fprValues = fprValues
        self.nFolds = folds
        self.epochs = epochs
        self.trainSize = X.shape[0]
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
            modelTrainer.fit(X_train, y_train, epochs=self.epochs)
            trainingTime[i,:] = np.pad(modelTrainer.trainingTime, (0, epochs-len(modelTrainer.trainingTime)), 'constant', constant_values=(0,0))

            # Evaluate the model
            predicted_probs = modelTrainer.predict_proba(X_test)

            for fpr in self.fprValues:
                tpr, threshold = self.get_tpr_at_fpr(predicted_probs, y_test, fpr)
                f1 = f1_score(y[test_index], predicted_probs >= threshold)
                self.metrics[fpr]["tpr"].append(tpr)
                self.metrics[fpr]["f1"].append(f1)
            
            # cleanup to avoid reusing the same model for the next fold
            del model, modelTrainer, modelInterfaceConfig

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
