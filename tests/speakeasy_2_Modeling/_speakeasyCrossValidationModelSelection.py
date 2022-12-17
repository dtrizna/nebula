from sklearn.model_selection import KFold
import numpy as np
import torch
import logging
import sys
sys.path.extend(['..', '.'])
from nebula.models import Cnn1DLinear, ModelAPI

def getCrossValidationMetrics(modelClass, modelConfig, X, y, folds=3):
    """
    Cross validate a model on a dataset and return the metrics for each fold.
    """
    # Create the folds
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(X)
    
    # Create the model
    model = modelClass(**modelConfig)

    # Create the output
    metrics = []

    # Iterate over the folds
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        logging.warning(f" [!] Fold {i+1} started...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y_train).long()),
            batch_size=32,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_test).long(), torch.from_numpy(y_test).long()),
            batch_size=32,
            shuffle=True
        )

        # Train the model
        model.fit(train_loader, epochs=3)

        # Evaluate the model
        evalStats = model.evaluate(test_loader)
        metrics.append(evalStats)

    return metrics


train_limit = 1000
x_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048.npy"
x_train = np.load(x_train)[:train_limit]

y_train = r"C:\Users\dtrizna\Code\nebula\data\data_filtered\speakeasy_trainset\speakeasy_VocabSize_1500_maxLen_2048_y.npy"
y_train = np.load(y_train)[:train_limit]

print("Trainset size: ", len(x_train))

VOCAB_SIZE = 1500
model = Cnn1DLinear(vocabSize=VOCAB_SIZE)
modelConfig = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "model": model,
    "lossFunction": torch.nn.BCEWithLogitsLoss(),
    "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
    "outputFolder": None, #r".\tests\speakeasy_2_Modeling\Cnn1DLinear",
    "verbosityBatches": 100
}

metrics = getCrossValidationMetrics(ModelAPI, modelConfig, x_train, y_train, folds=3)

# metrics -- [fold1, fold2, fold3]
# folds -- [lossValues, [accuracy, f1_score]]
mean_f1 =  np.array([x[1] for x in metrics]).squeeze().mean(axis=0)[0]

print("Mean F1 over all folds:", mean_f1)
import pdb;pdb.set_trace()