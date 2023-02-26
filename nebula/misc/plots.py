import numpy as np
import matplotlib.pyplot as plt

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

def plot_roc_curves(fpr, tpr, tpr_std=None, model_name="", axs=None, roc_auc=None, xlim=[-0.0005, 0.003], ylim=[0.3, 1.0]):
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
    axs[0].plot([0, 1], [0, 1], lw=2, linestyle="--")
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
    axs[1].plot([0, 1], [0, 1], lw=2, linestyle="--")
    if tpr_std is not None:
        tprs_upper = np.minimum(tpr + tpr_std, 1)
        tprs_lower = tpr - tpr_std
        color = axs[1].lines[-1].get_color()
        color = color[:-2] + "33"
        axs[1].fill_between(fpr, tprs_lower, tprs_upper, alpha=.2)
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)
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
