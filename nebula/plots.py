import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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
        counter, xlim=None,
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
