"""Candidate classifiers.

Class imbalance is handled once, by re-weighting the loss (class_weight /
scale_pos_weight). SMOTE on top of that double-corrects and pushes predicted
probabilities far from real fraud rates; calibration in main.py then maps
scores back to probabilities.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_candidate_models(pos_weight: float, random_state: int = 42) -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=random_state,
        ),
    }
