import os
import json
import logging
import numpy as np
from collections import defaultdict
from pandas import DataFrame

from time import sleep, time
from torch.optim import Adam, AdamW
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
                    false_positive_rates=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                    unlabeledDataSize=0.8,
                    randomState=None,
                    pretraingEpochs=5,
                    downstreamEpochs=5,
                    batchSize=256,
                    verbosity_n_batches=100,
                    optim_scheduler=None,
                    optim_step_budget=5000,
                    outputFolder=None,
                    dump_model_every_epoch=False,
                    dump_data_splits=True,
                    remask_epochs=False,
                    downsample_unlabeled_data=False):
        self.device = device
        self.training_types = training_types
        self.false_positive_rates = false_positive_rates
        self.unlabeledDataSize = unlabeledDataSize
        self.randomState = randomState
        self.pretrainingEpochs = pretraingEpochs
        self.downstreamEpochs = downstreamEpochs
        self.batchSize = batchSize
        self.verbosity_n_batches = verbosity_n_batches
        self.optim_step_budget = optim_step_budget
        self.optim_scheduler = optim_scheduler
        self.output_folder = outputFolder
        self.dump_model_every_epoch = dump_model_every_epoch
        self.dump_data_splits = dump_data_splits
        self.remask_epochs = remask_epochs
        self.downsample_unlabeled_data = downsample_unlabeled_data
        if self.downsample_unlabeled_data:
            assert isinstance(downsample_unlabeled_data, float) and 0 < downsample_unlabeled_data < 1
        
        self.model_class = modelClass
        self.model_config = modelConfig
        self.pretraining_task_class = pretrainingTaskClass
        self.pretraining_task_config = pretrainingTaskConfig

    def run_one_split(self, x, y, x_test, y_test):
        models = {k: None for k in self.training_types}
        model_trainer = {k: None for k in self.training_types}
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
                os.path.join(self.output_folder, splitData),
                unlabeled_data=unlabeled_data,
                labeled_x=labeled_x,
                labeled_y=labeled_y
            )
            logging.warning(f" [!] Saved dataset splits to {splitData}")

        logging.warning(' [!] Pre-training model...')

        # create a pretraining model and task
        models['pretrained'] = self.model_class(**self.model_config).to(self.device)
        model_trainer_config = {
            "device": self.device,
            "model": models['pretrained'],
            "modelForwardPass": models['pretrained'].pretrain,
            "loss_function": CrossEntropyLoss(),
            "optimizer_class": AdamW,
            "optimizer_config": {"lr": 2.5e-4},
            "optim_scheduler": self.optim_scheduler,
            "optim_step_budget": self.optim_step_budget,
            "verbosity_n_batches": self.verbosity_n_batches,
            "batchSize": self.batchSize,
            "falsePositiveRates": self.false_positive_rates,
        }
        if self.output_folder:
            model_trainer_config["outputFolder"] = os.path.join(self.output_folder, "preTraining")

        self.pretraining_task_config['model_trainer_config'] = model_trainer_config
        self.pretraining_task = self.pretraining_task_class(
            **self.pretraining_task_config
        )
        self.pretraining_task.pretrain(
            unlabeled_data,
            epochs=self.pretrainingEpochs,
            dump_model_every_epoch=self.dump_model_every_epoch,
            remask_epochs=self.remask_epochs
        )

        # downstream task for pretrained model
        model_trainer_config['loss_function'] = BCEWithLogitsLoss()
        model_trainer_config['modelForwardPass'] = None

        for model in self.training_types:
            logging.warning(f' [!] Training {model} model on downstream task...')
            if model != 'pretrained': # if not pretrained -- create a new model
                models[model] = self.model_class(**self.model_config).to(self.device)
            model_trainer_config['model'] = models[model]
            
            if self.output_folder:
                model_trainer_config["outputFolder"] = os.path.join(self.output_folder, f"downstreamTask_{model}")
            
            model_trainer[model] = ModelTrainer(**model_trainer_config)
            # TODO: don't like how step is calculated here
            # use size of x and self.unlabeledDataSize to calculate steps
            if model == 'full_data':
                model_trainer_config['optim_step_budget'] = self.optim_step_budget//2
                model_trainer[model].fit(x, y, self.downstreamEpochs, reporting_timestamp=timestamp)
            else:
                model_trainer_config['optim_step_budget'] = self.optim_step_budget//10
                model_trainer[model].fit(labeled_x, labeled_y, self.downstreamEpochs, reporting_timestamp=timestamp)
        
        for model in models:
            logging.warning(f' [*] Evaluating {model} model on test set...')
            metrics[model] = model_trainer[model].evaluate(x_test, y_test, metrics="json")
            logging.warning(f"\t[!] Test set AUC: {metrics[model]['auc']:.4f}")
            for reportingFPR in self.false_positive_rates:
                f1 = metrics[model]["fpr_"+str(reportingFPR)]["f1"]
                tpr = metrics[model]["fpr_"+str(reportingFPR)]["tpr"]
                logging.warning(f'\t[!] Test set scores at FPR: {reportingFPR:>6} --> TPR: {tpr:.4f} | F1: {f1:.4f}')
        del models, model_trainer # cleanup to not accidentaly reuse 
        clear_cuda_cache()
        return metrics

    def run_splits(self, x, y, x_test, y_test, nSplits=5, rest=None):
        metrics = {k: {"fpr_"+str(fpr): {"f1": [], "tpr": []} for fpr in self.false_positive_rates} for k in self.training_types}
        for trainingType in self.training_types:
            metrics[trainingType]['auc'] = []
        # collect metrics for number of iterations
        for i in range(nSplits):
            self.randomState += i # to get different splits
            logging.warning(f' [!] Running pre-training split {i+1}/{nSplits}')
            splitMetrics = self.run_one_split(x, y, x_test, y_test)
            for trainingType in self.training_types:
                metrics[trainingType]['auc'].append(splitMetrics[trainingType]['auc'])
                for fpr in self.false_positive_rates:
                    metrics[trainingType]["fpr_"+str(fpr)]["f1"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["f1"])
                    metrics[trainingType]["fpr_"+str(fpr)]["tpr"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["tpr"])
            if rest:
                sleep(rest)

        # compute mean and std for each metric
        for trainingType in self.training_types:
            for fpr in self.false_positive_rates:
                metrics[trainingType]["fpr_"+str(fpr)]["f1_mean"] = np.nanmean(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["f1_std"] = np.nanstd(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_mean"] = np.nanmean(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_std"] = np.nanstd(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
        return metrics


class CrossValidation(object):
    def __init__(self,
                 model_trainer_class,
                 model_trainer_config,
                 model_class,
                 model_config,
                 output_folder_root,
                false_positive_rates=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                 dump_data_splits=True):
        self.model_trainer_class = model_trainer_class
        self.model_trainer_config = model_trainer_config
        self.model_class = model_class
        self.model_config = model_config
        self.dump_data_splits = dump_data_splits
        self.false_positive_rates = false_positive_rates

        self.output_folder_root = output_folder_root
        os.makedirs(self.output_folder_root, exist_ok=True)
        
        self.metrics_val = defaultdict(lambda: defaultdict(list))
        self.aucs_val = []
        self.metrics_train = defaultdict(lambda: defaultdict(list))
        self.aucs_train = []

    def run_folds(
            self, 
            X,
            y,
            folds=3,
            epochs=5,
            false_positive_rates=None,
            random_state=42
    ):
        """
        Cross validate a model on a dataset and return the mean metrics over all folds depending on specific False Positive Rate (FPR).
        Returns: {fpr1: [mean_tpr, mean_f1], fpr2: [mean_tpr, mean_f1], ...], "avg_epoch_time": N seconds}
        """
        if false_positive_rates is not None:
            self.false_positive_rates = false_positive_rates
        self.trainSize = X.shape[0]
        if folds == 1:
            # kfold requires at least 2 folds -- this is rude shortcut 
            # to get a single run with training on 80%, and validating on 20% of the data
            self.nFolds = 5
            singleFold = True
        else:
            self.nFolds = folds
            singleFold = False

        if self.model_trainer_config["time_budget"] is not None:
            self.epochs = int(1e4)
            msg = f" [!] Training time budget: {self.model_trainer_config['time_budget']}s"
            logging.warning(msg)
        else:
            self.epochs = epochs
            logging.warning(f" [!] Epochs per fold: {self.epochs}")
        logging.warning(f" [!] Model config: {self.model_config}")

        kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=random_state)
        kf.get_n_splits(X)

        # Create the output
        self.training_time = np.empty(shape=(folds, self.epochs))
        # Iterate over the folds
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            timestamp = int(time())
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            logging.warning(f" [{i+1}/{self.nFolds}] Train set size: {len(X_train)}, Validation set size: {len(X_test)}")
            if self.dump_data_splits:
                splitData = f"dataset_splits_{timestamp}.npz"
                np.savez_compressed(
                    os.path.join(self.output_folder_root, splitData),
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                logging.warning(f" [!] Saved dataset splits to {splitData}")
            # Create the model
            model = self.model_class(**self.model_config)

            model_trainer_config = self.model_trainer_config
            model_trainer_config["model"] = model
            model_trainer_config["outputFolder"] = self.output_folder_root
            model_trainer = self.model_trainer_class(**model_trainer_config)

            # Train the model
            try:
                model_trainer.fit(X_train, y_train, epochs=self.epochs, reporting_timestamp=timestamp)
                self.training_time[i,:] = np.pad(model_trainer.training_time, (0, self.epochs-len(model_trainer.training_time)), 'constant', constant_values=(0,0))
            except RuntimeError as e: # cuda outofmemory
                logging.warning(f" [!] RuntimeError exception: {e}")
                break
            
            # collect stats of this run
            aucs_train, tprs_train, f1_train = self.collect_stats(model_trainer, X_train, y_train, name="training")
            aucs_val, tprs_val, f1_val = self.collect_stats(model_trainer, X_test, y_test, name="validation")
            self.aucs_train.append(aucs_train)
            self.aucs_val.append(aucs_val)
            for fpr in self.false_positive_rates:
                self.metrics_train[fpr]["tpr"].append(tprs_train[fpr])
                self.metrics_train[fpr]["f1"].append(f1_train[fpr])
                self.metrics_val[fpr]["tpr"].append(tprs_val[fpr])
                self.metrics_val[fpr]["f1"].append(f1_val[fpr])

            # cleanup
            del model, model_trainer, model_trainer_config
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
        for fpr in self.false_positive_rates:
            self.metrics_val[fpr]["tpr_avg"] = np.nanmean(self.metrics_val[fpr]["tpr"])
            self.metrics_val[fpr]["tpr_std"] = np.nanstd(self.metrics_val[fpr]["tpr"])
            self.metrics_val[fpr]["f1_avg"] = np.nanmean(self.metrics_val[fpr]["f1"])
            self.metrics_val[fpr]["f1_std"] = np.nanstd(self.metrics_val[fpr]["f1"])
            self.metrics_train[fpr]["tpr_avg"] = np.nanmean(self.metrics_train[fpr]["tpr"])
            self.metrics_train[fpr]["tpr_std"] = np.nanstd(self.metrics_train[fpr]["tpr"])
            self.metrics_train[fpr]["f1_avg"] = np.nanmean(self.metrics_train[fpr]["f1"])
            self.metrics_train[fpr]["f1_std"] = np.nanstd(self.metrics_train[fpr]["f1"])

        # Add the average epoch time
        self.metrics_train["epoch_time_avg"] = np.nanmean(self.training_time)

    def collect_stats(self, model_trainer, X, y, name="validation"):
        logging.warning(f" [!] Evaluating model on {name} set...")
        probs = model_trainer.predict_proba(X)
        logging.warning(f" [!] This fold metrics on {name} set:")
        try: # calculate auc
            auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        except ValueError as e: # only one class
            logging.warning(f" [!] AUC calculation exception: {e}")
            auc = np.nan
        logging.warning(f"\tAUC: {auc:.4f}")
        trps, f1s = {}, {}
        for fpr in self.false_positive_rates:
            tpr, threshold = get_tpr_at_fpr(y, probs, fpr)
            f1 = f1_score(y, probs >= threshold)
            trps[fpr], f1s[fpr] = tpr, f1
            logging.warning(f"\tFPR: {fpr} | TPR: {tpr:.4f} | F1: {f1:.4f}")
        if not self.false_positive_rates:
            f1 = f1_score(y, np.argmax(probs, axis=1), average='macro')
            logging.warning(f"\tF1: {f1:.4f}")
        return auc, trps, f1s

    def dump_metrics(self, prefix="", suffix=""):
        prefix = prefix.rstrip('_').lstrip('_') + '_' if prefix else ''
        suffix = f"_{suffix.rstrip('_').lstrip('_')}" if suffix else ''
        
        metricFilename = f"{prefix}metrics_validation{suffix}.json"
        metricFileFullpath = os.path.join(self.output_folder_root, metricFilename)
        with open(metricFileFullpath, "w") as f:
            json.dump(self.metrics_val, f, indent=4)
        logging.warning(f" [!] Metrics saved to {metricFileFullpath}")

        metricFilename = f"{prefix}metrics_training{suffix}.json"
        metricFileFullpath = os.path.join(self.output_folder_root, metricFilename)
        with open(metricFileFullpath, "w") as f:
            json.dump(self.metrics_train, f, indent=4)
        logging.warning(f" [!] Metrics saved to {metricFileFullpath}")        

    def log_avg_metrics(self):
        msg = f" [!] Average epoch time: {self.metrics_train['epoch_time_avg']:.2f}s | Mean values over {self.nFolds} folds:\n"
        msg += f"\tAUC: {self.metrics_val['auc_avg']:.4f}\n"
        for fpr in self.false_positive_rates:
            msg += f"\tFPR: {fpr:>6} -- TPR: {self.metrics_val[fpr]['tpr_avg']:.4f} -- F1: {self.metrics_val[fpr]['f1_avg']:.4f}\n"
        logging.warning(msg)


def read_cv_metrics_file(file):
    with open(file, "r") as f:
            metrics = json.load(f)
    cols = ["tpr_avg", "tpr_std", "f1_avg", "f1_std"]
    dfMetrics = DataFrame(columns=cols)
    timeValue = None
    aucs = None
    for key in metrics:
        if key in ("avg_epoch_time", "epoch_time_avg"):
            timeValue = metrics[key]
        elif key.startswith("auc"):
            if key == "auc":
                aucs = metrics[key]
        else:
            # false positive rate reporting
            if isinstance(metrics[key], dict):
                # take mean, ignore 0.
                tprs = np.array(metrics[key]['tpr'])
                tprs[tprs == 0] = np.nan
                tpr_avg = np.nanmean(tprs)
                tpr_std = np.nanstd(tprs)
                # same for f1
                f1s = metrics[key]['f1']
                f1s[f1s == 0] = np.nan
                f1_avg = np.nanmean(f1s)
                f1_std = np.nanstd(f1s)
            else: # legacy trp/f1 reporting
                arr = np.array(metrics[key]).squeeze()
                if arr.ndim == 1:
                    tpr_avg = arr[0]
                    tpr_std = 0
                    f1_avg = arr[1]
                    f1_std = 0
                elif arr.ndim == 2:
                    tpr_avg = np.nanmean(arr[:, 0])
                    tpr_std = np.nanstd(arr[:, 0])
                    f1_avg = np.nanmean(arr[:, 1])
                    f1_std = np.nanstd(arr[:, 1])
            dfMetrics.loc[key] = [tpr_avg, tpr_std, f1_avg, f1_std]
    return dfMetrics, aucs, timeValue


def read_cv_metrics_folder(folder, key_lambda, extra_file_filter=None):
    """Example:
        def training_filter(file):
            return file.endswith("training.json")

        def validation_filter(file):
            return file.endswith("validation.json")
        
        def key_extractor(file):
            return "_".join(file.split("_")[0:3])
        
        training_metrics = read_cv_metrics_folder(".", key_extractor, training_filter)
        validation_metrics = read_cv_metrics_folder(".", key_extractor, validation_filter)
    """
    if extra_file_filter:
        metricFiles = [x for x in os.listdir(folder) if x.endswith(".json") and extra_file_filter(x)]
    else:
        metricFiles = [x for x in os.listdir(folder) if x.endswith(".json")]
    key_to_file = dict(zip(['_'.join(key_lambda(x)) for x in metricFiles], metricFiles))
    try:
        key_to_file = {k: v for k, v in sorted(key_to_file.items(), key=lambda item: int(item[0]))}
    except ValueError:
        key_to_file = {k: v for k, v in sorted(key_to_file.items(), key=lambda item: item[0])}

    metric_dict = dict()
    for key in key_to_file:
        metrics, _, _ = read_cv_metrics_file(os.path.join(folder, key_to_file[key]))
        metric_dict[key] = metrics
        
    return metric_dict


def read_cv_data_split(npz_file):
    with np.load(npz_file) as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def read_cv_data_splits(folder, splits=3):
    fold_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(".npz")][0:splits]
    fold_sets = [read_cv_data_split(f) for f in fold_files]
    return fold_sets
