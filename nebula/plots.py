import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pandas import DataFrame

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
        figSize=(12,8)
    ):
    counts = [x[1] for x in counter.most_common(mostCommon)]
    
    plt.figure(figsize=figSize)
    plt.plot(np.arange(len(counts)), counts)
    plt.yscale("log")
    # add ticks and grid to plot
    plt.grid(which="both")
    # save to file
    if xlim:
        plt.xlim(xlim)
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

def plotListElementLengths(list_of_strings, xlim=None, outfile=None):
    lengths = [len(x) for x in list_of_strings]
    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.hist(lengths, bins=100)
    plt.title("Histogram of list element lengths")
    if xlim:
        plt.xlim(xlim)
    plt.yscale("log")
    plt.ylabel("Count (log)")
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    return lengths

def readCrossValidationFolder(folder, diffExtractor, extraFileFilter=None):
    if extraFileFilter:
        metricFiles = [x for x in os.listdir(folder) if x.endswith(".json") and extraFileFilter(x)]
    else:
        metricFiles = [x for x in os.listdir(folder) if x.endswith(".json")]
    fileToField = dict(zip(metricFiles, [diffExtractor(x) for x in metricFiles]))
    # sort fileToFiel based on values
    fileToField = {k: v for k, v in sorted(fileToField.items(), key=lambda item: int(item[1]))}

    cols = ["tpr_avg", "tpr_std", "f1_avg", "f1_std"]
    dfDict = dict()
    timeDict = dict()
    for file in fileToField:
        with open(os.path.join(folder, file), "r") as f:
            metrics = json.load(f)
        dfDict[fileToField[file]] = DataFrame(columns=cols)
        for fpr in metrics:
            if fpr in ("avg_epoch_time", "epoch_time_avg"):
                timeDict[fileToField[file]] = metrics[fpr]
            else:
                if isinstance(metrics[fpr], dict):
                    tpr_avg = metrics[fpr]["tpr_avg"]
                    tpr_std = np.std(metrics[fpr]["tpr"])
                    f1_avg = metrics[fpr]["f1_avg"]
                    f1_std = np.std(metrics[fpr]["f1"])
                else:
                    arr = np.array(metrics[fpr]).squeeze()
                    if arr.ndim == 1:
                        tpr_avg = arr[0]
                        tpr_std = 0
                        f1_avg = arr[1]
                        f1_std = 0
                    elif arr.ndim == 2:
                        tpr_avg = arr[:, 0].mean()
                        tpr_std = arr[:, 0].std()
                        f1_avg = arr[:, 1].mean()
                        f1_std = arr[:, 1].std()
                dfDict[fileToField[file]].loc[fpr] = [tpr_avg, tpr_std, f1_avg, f1_std]
    return dfDict, timeDict

def plotCrossValidationDict(dfDict, key, ax=None, legendTitle=None, savePath=None):
    if ax is None:
        fig, ax = plt.subplots()
    for diff in dfDict:
        x = dfDict[diff].index.values
        y = dfDict[diff][f"{key}_avg"]
        yerr = dfDict[diff][f"{key}_std"]
        if (yerr > 0).any():
            capsize=10
        else:
            capsize=0
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

def plotCrossValidationTrainingTime(timeDict, ax=None, savePath=None, xlabel=""):
    if ax is None:
        fig, ax = plt.subplots()
    if any([x for x in timeDict if "_" in x]):
        x = [x for x in timeDict]
    else:
        x = [int(x) for x in timeDict]
    y = [timeDict[str(value)] for value in x]
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"Training Time vs {xlabel}")
    # set y limit to be -10 + 10 of min max y values
    ax.set_ylim(min(y) - 10, max(y) + 10)
    # add grid
    ax.grid()
    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)

def plotCrossValidationFolder(folder, field, diffExtractor, extraFileFilter=None, savePath=None, title=None, figSize=(18, 6)):
    dfDict, timeDict = readCrossValidationFolder(folder, diffExtractor, extraFileFilter=extraFileFilter)

    fig, ax = plt.subplots(1, 3, figsize=figSize)
    if title is None:
        title = folder
    plt.suptitle(title, fontsize=16)

    plotCrossValidationDict(dfDict, "tpr", legendTitle=field, ax=ax[0])
    plotCrossValidationDict(dfDict, "f1", legendTitle=field, ax=ax[1])
    plotCrossValidationTrainingTime(timeDict, ax=ax[2], xlabel=field)

    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)

def plotCrossValidationFieldvsKeys(dfDict, field, keyName, fpr, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    keys = dfDict.keys()
    y = [dfDict[key].loc[str(fpr)][f"{field}_avg"] for key in keys]
    yerr = [dfDict[key].loc[str(fpr)][f"{field}_std"] for key in keys]
    ax.errorbar(keys, y, yerr=yerr, marker="o", capsize=10)
    ax.set_xlabel("{keyName}")
    ax.set_ylabel(f"{field}".upper())
    ax.set_title(f"{field.upper()} vs {keyName} at FPR={fpr}")
    # add grid
    ax.grid()

def plotVocabSizeMaxLenTests(inFolder, plotOutFolder):
    field = "vocabSize"
    diffExtractor = lambda x: x.split(field)[1].split("_")[1]

    myFilter = "maxLen_1024_"
    extraFileFilter = lambda x: myFilter in x

    title = f"{inFolder.split('_')[0]} {field}; {'='.join(myFilter.split('_')[:-1])}"
    pngPath = plotOutFolder + title.replace(" ", "_").replace("=", "_").replace(";", "")+".png"
    plotCrossValidationFolder(inFolder, field, diffExtractor, extraFileFilter, title=title, savePath=pngPath)

    field = "maxLen"
    diffExtractor = lambda x: x.split(field)[1].split("_")[1]

    myFilter = "vocabSize_2000_"
    extraFileFilter = lambda x: myFilter in x

    title = f"{inFolder.split('_')[0]} {field}; {'='.join(myFilter.split('_')[:-1])}"
    pngPath = plotOutFolder + title.replace(" ", "_").replace("=", "_").replace(";", "")+".png"
    plotCrossValidationFolder(inFolder, field, diffExtractor, extraFileFilter, title=title, savePath=pngPath)