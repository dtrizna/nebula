import torch
import numpy as np
from tqdm import tqdm
from pandas import DataFrame, concat
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import defaultdict
from torch.nn import Linear

from ..misc import read_files_from_log
from ..misc.plots import plot_roc_curves

def get_preds(model, x, y, model_name=None, batch_size=64, device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x).long()#.to(device)
    y = torch.from_numpy(y).float()#.to(device)

    # getting predictions
    testLoader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True)

    # get count of trainable model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        y_pred = []
        y_true = []
        if model_name:
            print(f"Evaluating {model_name} | Size: {total_params}...")
        else:
            print(f"Evaluating model | Size: {total_params}")
        for X, y in tqdm(testLoader):
            y_hat = model(X.to(device))
            y_pred.append(y_hat.cpu().numpy())
            y_true.append(y.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return y_true, y_pred


def get_roc(y_true, y_pred, metrics_full=False, threshold=0.5):
    # calculating ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    if metrics_full:
        # compute f1, precision, recall
        y_pred_labels = np.where(y_pred > threshold, 1, 0)
        f1 = f1_score(y_true, y_pred_labels)
        recall = recall_score(y_true, y_pred_labels)
        precision = precision_score(y_true, y_pred_labels)
        acc = accuracy_score(y_true, y_pred_labels)
        return fpr, tpr, roc_auc, f1, recall, precision, acc
    else:
        return fpr, tpr, roc_auc

def get_model_rocs(run_types, model_class, model_config, data_splits, model_files=None, logfile=None, splits=3, verbose=True):
    if model_files is None:
        model_files = read_files_from_log(logfile)
    assert len(model_files) == splits*len(run_types),\
        f"Number of model files does not match number of models: {len(model_files)}, {splits*len(run_types)}"
    assert len(data_splits) == splits,\
        f"Number of data splits does not match provided models: {len(data_splits)}, {splits}"
    types_to_model_files = {k: model_files[i*splits:(i+1)*splits] for i, k in enumerate(run_types)}
    if verbose:
        msg = "Model files:\n"
        for k, v in types_to_model_files.items():
            msg += f"\t{k}: {v}\n"
        print(msg)
    metrics = defaultdict(list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_type, model_files in types_to_model_files.items():
        for i, model_file in enumerate(model_files):
            model = model_class(**model_config).to(device)
            model.load_state_dict(torch.load(model_file))
            model.eval()
            n_output_classes = [x for x in model.children() if isinstance(x, Linear)][-1].out_features
            if n_output_classes == 1:
                fpr, tpr, roc_auc = get_roc(
                    model, 
                    data_splits[i]["X_test"],
                    data_splits[i]["y_test"],
                    model_name=f"{run_type}_split_{i}"
                )
                metrics[run_type].append((fpr, tpr, roc_auc))
            else: # multiclass
                # TODO: implement multiclass roc
                pass
    return metrics

def allign_metrics(metrics, base_fpr_scale=50001):
    base_fpr = np.linspace(0, 1, base_fpr_scale)
    trps_same = defaultdict(dict)
    for run_type in metrics.keys():
        for i, (fpr, tpr, *_) in enumerate(metrics[run_type]):
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            trps_same[run_type][i] = tpr
    # take mean and std of all tprs
    tprs_mean = {k: np.mean([v[i] for i in v.keys()], axis=0) for k, v in trps_same.items()}
    tprs_std = {k: np.std([v[i] for i in v.keys()], axis=0) for k, v in trps_same.items()}
    return base_fpr, tprs_mean, tprs_std

def report_alligned_metrics(base_fpr, tprs_mean, tprs_std, metrics=None, metrics_full=False, 
                            fprs=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], 
                            xlim=[-0.0005, 0.3], 
                            ylim=[0.6, 1]):
    tprsdf = DataFrame()
    axs = None        
    for i, model_type in enumerate(tprs_mean.keys()):
        stds = tprs_std[model_type] if tprs_std else None
        if metrics is not None:
            auc_mean = np.mean([metrics[model_type][i][2] for i in range(len(metrics[model_type]))])
        else:
            auc_mean = None
        axs = plot_roc_curves(
            base_fpr, 
            tprs_mean[model_type], 
            tpr_std=stds,
            model_name=f"{model_type}",
            axs=axs,
            roc_auc=auc_mean,
            xlim=xlim,
            ylim=ylim
        )
        tprs = defaultdict(list)
        for fpr in fprs:
            tpr = tprs_mean[model_type][np.argmin(np.abs(base_fpr - fpr))]
            tprs[fpr].append(tpr)
        tprsdf = concat([tprsdf, DataFrame(tprs, index=[model_type])])
    if axs is not None:
        [ax.grid() for ax in axs]
        [ax.legend(loc='lower right') for ax in axs]

    # add other metrics to dataframe
    if metrics is not None:
        aucdf = DataFrame([np.mean([y[2] for y in metrics[x]]) for x in metrics], index=metrics.keys(), columns=["AUC"])
        if metrics_full:
            f1df = DataFrame([np.mean([y[3] for y in metrics[x]]) for x in metrics], index=metrics.keys(), columns=["F1"])
            recall_df = DataFrame([np.mean([y[4] for y in metrics[x]]) for x in metrics], index=metrics.keys(), columns=["Recall"])
            precision_df = DataFrame([np.mean([y[5] for y in metrics[x]]) for x in metrics], index=metrics.keys(), columns=["Precision"])
            acc_df = DataFrame([np.mean([y[6] for y in metrics[x]]) for x in metrics], index=metrics.keys(), columns=["Accuracy"])
            tprsdf = concat([tprsdf, aucdf, f1df, recall_df, precision_df, acc_df], axis=1)
        else:
            tprsdf = concat([tprsdf, aucdf], axis=1)

    return tprsdf, axs
    