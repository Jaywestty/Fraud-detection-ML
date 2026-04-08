from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', use_label_encoder=False, random_state=42)
    }