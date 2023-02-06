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
        figSize=(12,8),
        ax=None
    ):
    counts = [x[1] for x in counter.most_common(mostCommon)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figSize)
    ax.plot(np.arange(len(counts)), counts)
    ax.set_yscale("log")
    # add ticks and grid to plot
    ax.grid(which="both")
    # save to file
    if xlim:
        ax.set_xlim(xlim)
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

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
    try:
        if any([x for x in timeDict if "_" in x]):
            x = [x for x in timeDict]
        else:
            x = [int(x) for x in timeDict]
    except ValueError:
        x = [x for x in timeDict]
    y = [timeDict[str(value)] for value in x]
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"Average Epoch Time vs {xlabel}")
    # set y limit to be -10 + 10 of min max y values
    #ax.set_ylim(min(y) - 10, max(y) + 10)
    # add grid
    ax.grid()
    ax.set_yscale('log')
    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)

def plotCrossValidation_TPR_F1_Time(dfDict, timeDict, title=None, legentTitle=None, savePath=None, figSize=(18, 6)):
    fig, ax = plt.subplots(1, 3, figsize=figSize)
    plt.suptitle(title, fontsize=16)

    plotCrossValidationDict(dfDict, "tpr", legendTitle=legentTitle, ax=ax[0])
    plotCrossValidationDict(dfDict, "f1", legendTitle=legentTitle, ax=ax[1])
    plotCrossValidationTrainingTime(timeDict, ax=ax[2], xlabel=legentTitle)

    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)
    return fig, ax

def plotCrossValidationFolder(folder, field, diffExtractor, extraFileFilter=None, savePath=None, title=None, figSize=(18, 6)):
    dfDict, timeDict = readCrossValidationFolder(folder, diffExtractor, extraFileFilter=extraFileFilter)
    if title is None:
        title = folder
    plotCrossValidation_TPR_F1_Time(dfDict, timeDict, title=title, legentTitle=field, savePath=savePath, figSize=figSize) 

def plotCrossValidationFieldvsKeys(dfDict, field, keyName, fpr, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    keys = dfDict.keys()
    y = [dfDict[key].loc[str(fpr)][f"{field}_avg"] for key in keys]
    yerr = [dfDict[key].loc[str(fpr)][f"{field}_std"] for key in keys]
    ax.errorbar(keys, y, yerr=yerr, marker="o", capsize=10)
    ax.set_xlabel(f"{keyName}")
    ax.set_ylabel(f"{field}".upper())
    ax.set_title(f"{field.upper()} vs {keyName} at FPR={fpr}")
    # add grid
    ax.grid()

def plotVocabSizeMaxLenTests(inFolder, plotOutFolder, maxLen=1024, vocabSize=2000):
    field = "vocabSize"
    diffExtractor = lambda x: x.split(field)[1].split("_")[1]

    myFilter = f"maxLen_{maxLen}_"
    extraFileFilter = lambda x: myFilter in x

    title = f"{inFolder.split('_')[0]} {field}; {'='.join(myFilter.split('_')[:-1])}"
    pngPath = os.path.join(plotOutFolder, title.replace(" ", "_").replace("=", "_").replace(";", "").replace("crossValidation", "")+".png")
    plotCrossValidationFolder(inFolder, field, diffExtractor, extraFileFilter, title=title, savePath=pngPath)

    field = "maxLen"
    diffExtractor = lambda x: x.split(field)[1].split("_")[1]

    myFilter = f"vocabSize_{vocabSize}_"
    extraFileFilter = lambda x: myFilter in x

    title = f"{inFolder.split('_')[0]} {field}; {'='.join(myFilter.split('_')[:-1])}"
    pngPath = os.path.join(plotOutFolder, title.replace(" ", "_").replace("=", "_").replace(";", "").replace("crossValidation", "")+".png")
    plotCrossValidationFolder(inFolder, field, diffExtractor, extraFileFilter, title=title, savePath=pngPath)

def plotVocabSizeMaxLenArchComparison(vmFolders, maxLen=512, vocabSize=2000, savePath=None, legendTitle=None, title="Architecture", figSize=(18, 6), legendValues=None, fprs=['0.0001', '0.001', '0.01', '0.1']):
    metricDict = dict()
    timeDict = dict()
    for folder in vmFolders:
        key = folder.split("\\")[-1].rstrip("_VocabSize_maxLen").rstrip("_PositEmbFix")
        try:
            metricFile = [x for x in os.listdir(folder) if f"maxLen_{maxLen}_vocabSize_{vocabSize}" in x][0]
        except IndexError:
            metricFile = [x for x in os.listdir(folder) if f"metrics" in x][0]
        metricFile = os.path.join(folder, metricFile)
        dfDict, timeValue = readCrossValidationMetricFile(metricFile)
        dfDict = dfDict[dfDict.index.isin(fprs)]
        metricDict[key] = dfDict
        timeDict[key] = timeValue
    plotTitle = f"{title} Comparison"
    fig, ax = plotCrossValidation_TPR_F1_Time(metricDict, timeDict, legentTitle=legendTitle, savePath=savePath, title=plotTitle, figSize=figSize)
    if legendValues:
                # modify legend for first plot
        handles, _ = ax[0].get_legend_handles_labels()
        ax[0].legend(handles, legendValues[0], title=title)
        # do the saame for ax[1]
        handles, _ = ax[1].get_legend_handles_labels()
        ax[1].legend(handles, legendValues[0], title=title)
        # set basePaths as xticklabels for ax[2]
        ax[2].set_xticks(ax[2].get_xticks())
        # wrap labels to fit correctly in plot
        ax[2].set_xlabel("")
        _ = ax[2].set_xticklabels(legendValues[1])
    return metricDict, timeDict, fig, ax

def plotHeatmap(df, title, rangeL=None, xlabel=None, ylabel=None, savePath=None, figSize=(12, 4), ax=None, cmap=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figSize)
    # plot heatmap with vmin and vmax -5 and +5 from min and max values in df
    if rangeL:
        vmin = df.min().min() - rangeL
        vmax = df.max().max() + rangeL
    else:
        vmin = None
        vmax = None
    # heatmap with blue colormap
    sns.heatmap(df, ax=ax, annot=True, fmt=".3f", vmin=vmin, vmax=vmax, cmap=cmap)    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savePath:
        #plt.tight_layout()
        plt.savefig(savePath)

def plotVocabSizeMaxLenHeatmap(inFolder, fpr, savePath=None, rangeL=None, figSize=(14, 5), supTitle=None, vocabSizes=None, maxLens=None, axs=None):
    metricFiles = [file for file in os.listdir(inFolder) if file.endswith(".json")]

    dfHeatTPR = DataFrame()
    dfHeatF1 = DataFrame()
    for file in metricFiles:
        vocabSize = int(file.split("vocabSize")[1].split("_")[1])
        maxLen = int(file.split("maxLen")[1].split("_")[1])
        if vocabSizes and vocabSize not in vocabSizes:
            continue
        if maxLens and maxLen not in maxLens:
            continue
        dfMetric, _ = readCrossValidationMetricFile(os.path.join(inFolder, file))
        dfHeatTPR.loc[vocabSize, maxLen] = dfMetric["tpr_avg"][fpr]
        dfHeatF1.loc[vocabSize, maxLen] = dfMetric["f1_avg"][fpr]
    dfHeatF1 = dfHeatF1.sort_index().transpose().sort_index(ascending=False)
    dfHeatTPR = dfHeatTPR.sort_index().transpose().sort_index(ascending=False)

    # create 2 plots
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=figSize)
        if not supTitle:
            plt.suptitle(f"{inFolder.split('_')[0]} with FPR={fpr} for different vocabSize and sequence length", fontsize=14)
        else:
            plt.suptitle(f"{supTitle} with FPR={fpr} for different vocabSize and sequence length", fontsize=14)

    plotHeatmap(dfHeatTPR, f"True-Positive Rate", rangeL=rangeL, xlabel="vocabSize", ylabel="sequence length", ax=axs[0])
    plotHeatmap(dfHeatF1, f"F1-score", rangeL=rangeL, xlabel="vocabSize", ylabel="sequence length", cmap="Blues", ax=axs[1])

    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)

    return axs


def readCrossValidationMetricFile(file):
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
                tprs = metrics[key]['tpr']
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

def readCrossValidationFolder(folder, diffExtractor, extraFileFilter=None):
    if extraFileFilter:
        metricFiles = [x for x in os.listdir(folder) if x.endswith(".json") and extraFileFilter(x)]
    else:
        metricFiles = [x for x in os.listdir(folder) if x.endswith(".json")]
    fileToField = dict(zip(metricFiles, [diffExtractor(x) for x in metricFiles]))
    # sort fileToFiel based on values
    fileToField = {k: v for k, v in sorted(fileToField.items(), key=lambda item: int(item[1]))}

    dfDict = dict()
    timeDict = dict()
    for file in fileToField:
        dfMetrics, avgEpochTime = readCrossValidationMetricFile(os.path.join(folder, file))
        dfDict[fileToField[file]] = dfMetrics
        timeDict[fileToField[file]] = avgEpochTime
        
    return dfDict, timeDict
