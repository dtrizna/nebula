import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pandas import DataFrame
from nebula.evaluation import readCrossValidationFolder, readCrossValidationMetricFile

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
    ax.set_ylim(min(y) - 10, max(y) + 10)
    # add grid
    ax.grid()
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

def plotVocabSizeMaxLenArchComparison(maxLen=512, vocabSize=2000, savePath=None, legendTitle=None, figSize=(18, 6)):
    vmFolders = [x for x in os.listdir() if "VocabSize_maxLen" in x]

    metricDict = dict()
    timeDict = dict()
    for folder in vmFolders:
        key = folder.rstrip("_VocabSize_maxLen")
        metricFile = [x for x in os.listdir(folder) if f"maxLen_{maxLen}_vocabSize_{vocabSize}" in x][0]
        metricFile = os.path.join(folder, metricFile)
        dfDict, timeValue = readCrossValidationMetricFile(metricFile)
        metricDict[key] = dfDict
        timeDict[key] = timeValue
    title = f"Architecture Comparison; maxLen={maxLen}, vocabSize={vocabSize}"
    plotCrossValidation_TPR_F1_Time(metricDict, timeDict, legentTitle=legendTitle, savePath=savePath, title=title, figSize=figSize)
    return metricDict, timeDict

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
    sns.heatmap(df, ax=ax, annot=True, fmt=".2f", vmin=vmin, vmax=vmax, cmap=cmap)    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)

def plotVocabSizeMaxLenHeatmap(inFolder, fpr, savePath=None, rangeL=None, figSize=(14, 5)):
    metricFiles = [file for file in os.listdir(inFolder) if file.endswith(".json")]

    dfHeatTPR = DataFrame()
    dfHeatF1 = DataFrame()
    for file in metricFiles:
        vocabSize = int(file.split("vocabSize")[1].split("_")[1])
        maxLen = int(file.split("maxLen")[1].split("_")[1])
        dfMetric, timeDict = readCrossValidationMetricFile(os.path.join(inFolder, file))
        dfHeatTPR.loc[vocabSize, maxLen] = dfMetric["tpr_avg"][fpr]
        dfHeatF1.loc[vocabSize, maxLen] = dfMetric["f1_avg"][fpr]
    dfHeatF1 = dfHeatF1.sort_index().transpose().sort_index(ascending=False)
    dfHeatTPR = dfHeatTPR.sort_index().transpose().sort_index(ascending=False)

    # create 2 plots
    fig, ax = plt.subplots(1, 2, figsize=figSize)

    plotHeatmap(dfHeatTPR, f"True-Positive Rate", rangeL=rangeL, xlabel="vocabSize", ylabel="sequence length", ax=ax[0])
    plotHeatmap(dfHeatF1, f"F1-score", rangeL=rangeL, xlabel="vocabSize", ylabel="sequence length", ax=ax[1], cmap="Blues")
    _ = fig.suptitle(f"{inFolder.split('_')[0]} with FPR={fpr} for different vocabSize and sequence length", fontsize=14)

    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)
