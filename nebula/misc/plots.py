import numpy as np
import matplotlib.pyplot as plt
from . import read_files_from_log

def plot3D(arr, labels, title):
    #from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arr[:,0], arr[:,1], arr[:,2], c=labels)
    plt.title(title)
    plt.show()

def tSneReductionTo3D(arr):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(arr)
    return tsne_results

def plotCounterCountsLineplot(
        counter, 
        xlim=None,
        mostCommon=None, 
        outfile=None,
        figSize=(12,8),
        ax=None,
        ylogscale=True
    ):
    counts = [x[1] for x in counter.most_common(mostCommon)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figSize)
    ax.plot(np.arange(len(counts)), counts)
    if ylogscale:
        ax.set_yscale("log")
    # add ticks and grid to plot
    ax.grid(which="both")
    # save to file
    if xlim:
        ax.set_xlim(xlim)
    if outfile:
        plt.savefig(outfile)
    return counts, ax

def plotListElementLengths(list_of_strings, xlim=None, outfile=None, bins=100, ax=None):
    lengths = [len(x) for x in list_of_strings]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.hist(lengths, bins=bins)
    ax.set_title("Histogram of list element lengths")
    if xlim:
        ax.set_xlim(xlim)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log)")
    ax.grid(which="both")
    if outfile:
        plt.tight_layout()
        plt.savefig(outfile)
    return lengths, ax

def plot_cv_metrics_dict(metric_dict, key='tpr', ax=None, legendTitle=None, legendValues=None, savePath=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, diff in enumerate(metric_dict):
        x = metric_dict[diff].index.values
        y = metric_dict[diff][f"{key}_avg"]
        yerr = metric_dict[diff][f"{key}_std"]
        if (yerr > 0).any():
            capsize=10
        else:
            capsize=0
        if legendValues:
            diff = legendValues[i]
        ax.errorbar(x, y, yerr=yerr, label=diff, marker="o", capsize=capsize)
    ax.set_xlabel("FPR")
    ax.set_ylabel(f"{key}".upper())
    ax.set_title(f"{key.upper()} vs FPR")
    ax.legend(title=legendTitle)
    # add grid
    ax.grid()
    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)
    return ax

def plot_roc_curves(fpr, tpr, tpr_std=None, model_name="", axs=None, roc_auc=None, xlim=[-0.0005, 0.003], ylim=[0.3, 1.0], plot_random=False):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].set_title("ROC Curve")
        axs[1].set_title("Zoomed in ROC Curve")
        axs[0].set_xlabel("False Positive Rate")
        axs[0].set_ylabel("True Positive Rate")
        axs[1].set_xlabel("False Positive Rate")
        axs[1].set_ylabel("True Positive Rate")
    # plotting ROC curve
    if roc_auc:
        label = f"{model_name} (AUC = {roc_auc:.6f})"
    else:
        label = model_name
    axs[0].plot(fpr, tpr, lw=2, label=label)
    if tpr_std is not None:
        tprs_upper = np.minimum(tpr + tpr_std, 1)
        tprs_lower = tpr - tpr_std
        color = axs[0].lines[-1].get_color()
        color = color[:-2] + "33"
        axs[0].fill_between(fpr, tprs_lower, tprs_upper, alpha=.2)
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    
    # plot zoomed in ROC curve x-axis from 0 to 0.2
    axs[1].plot(fpr, tpr, lw=2, label=label)
    if tpr_std is not None:
        tprs_upper = np.minimum(tpr + tpr_std, 1)
        tprs_lower = tpr - tpr_std
        color = axs[1].lines[-1].get_color()
        color = color[:-2] + "33"
        axs[1].fill_between(fpr, tprs_lower, tprs_upper, alpha=.2)
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)
    
    if plot_random:
        axs[0].plot([0, 1], [0, 1], lw=2, linestyle="--")
        axs[1].plot([0, 1], [0, 1], lw=2, linestyle="--")
    return axs

def plot_roc_curve(fpr, tpr, model_name, ax, xlim=[-0.0005, 0.003], ylim=[0.3, 1.0], roc_auc=None):
    if roc_auc:
        label = f"{model_name} (AUC = {roc_auc:.6f})"
    else:
        label = model_name
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def plot_losses(run_types, logfile=None, folder='.', splits=3, start=100, n=20, ax=None, spare_ticks=2, skip_types=[], only_first_loss=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    losses = read_files_from_log(logfile=logfile, folder=folder, pattern="trainLosses")
    types_to_losses = {k: losses[i*splits:(i+1)*splits] for i, k in enumerate(run_types)}
    if only_first_loss:
        types_to_losses = {k: [v[0]] for k, v in types_to_losses.items()}
    
    for run_type in types_to_losses:
        if run_type in skip_types:
            continue
        loss_stack = np.vstack([np.load(x) for x in types_to_losses[run_type]])
        loss_mean = np.mean(loss_stack, axis=0)[start:]
        loss_std = np.std(loss_stack, axis=0)[start:]
        # take every Nth value from loss_mean and loss_std
        loss_mean = loss_mean[::n]
        loss_std = loss_std[::n]

        ax.plot(loss_mean, label=run_type)
        ax.fill_between(np.arange(len(loss_mean)), loss_mean-loss_std, loss_mean+loss_std, alpha=0.2)
    
    ax.set_xlabel("Batches")
    ax.set_ylabel("Loss")
    # set xticklabels to account sampling every N-th value
    xticks = len(ax.get_xticklabels()) - spare_ticks
    ax.set_xticklabels([0] + list(np.arange(start, len(loss_mean)*n, len(loss_mean)*n//xticks)))
    
    return ax
