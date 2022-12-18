from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, f1_score
from collections import defaultdict
import numpy as np
import logging

def get_tpr_at_fpr(predicted_probs, true_labels, fprNeeded):
  fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
  tpr_at_fpr = tpr[fpr <= fprNeeded][-1]
  threshold_at_fpr = thresholds[fpr <= fprNeeded][-1]
  return tpr_at_fpr, threshold_at_fpr


def getCrossValidationMetrics(modelClass, modelConfig, X, y, folds=3, epochs=3, fprs=[0.1, 0.01, 0.001], mean=True):
    """
    Cross validate a model on a dataset and return the metrics for each fold.
    """
    # Create the folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    
    # Create the model
    model = modelClass(**modelConfig)

    # Create the output
    metrics = defaultdict(list)

    # Iterate over the folds
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        logging.warning(f" [!] Fold {i+1} started...")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        model.fit(X_train, y_train, epochs=epochs)

        # Evaluate the model
        preds_probs = model.predict_proba(X_test)

        for fpr in fprs:
            tprs, threshold = get_tpr_at_fpr(preds_probs, y_test, fpr)
            f1 = f1_score(y_test, preds_probs > threshold)
            metrics[fpr].append([tprs, f1])
    
    if mean:
        for fpr in fprs:
            metrics[fpr] = np.mean(metrics[fpr], axis=0)

    return dict(metrics)