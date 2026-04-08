import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def evaluate_model(model, X_test, y_test):

    y_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    pr_auc = average_precision_score(y_test, y_proba)

    return {
        "pr_auc": pr_auc,
        "best_threshold": best_thresh
    }