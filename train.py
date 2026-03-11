# train.py

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # use imblearn pipeline to support SMOTE
from pathlib import Path
from datetime import datetime
import joblib

# Define target and feature lists
TARGET_COLUMN = "is_fraud"

NUM_COLS = [
    "amount_usd",
    "fee",
    "exchange_rate_src_to_dest",
    "ip_risk_score",
    "account_age_days",
    "device_trust_score",
    "chargeback_history_count",
    "risk_score_internal",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk"
]

CAT_COLS = [
    "home_country",
    "source_currency",
    "dest_currency",
    "channel",
    "kyc_tier",
    "new_device",
    "location_mismatch",
    "cross_border",
    "device_risk"
]

def main():
    # Load dataset
    df = pd.read_csv("nova_pay_transcations.csv") 

    # Feature Engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["transaction_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["cross_border"] = (df["home_country"] != df["ip_country"]).astype(int)
    df["device_risk"] = ((df["new_device"] == True) & (df["device_trust_score"] < 0.5)).astype(int)

    df["velocity_ratio"] = df["txn_velocity_1h"] / (df["txn_velocity_24h"] + 1)
    df["velocity_x_amount"] = df["velocity_ratio"] * df["amount_usd"].fillna(0)
    df["new_account_flag"] = (df["account_age_days"] < 60).astype(int)
    df["young_account_velocity"] = df["new_account_flag"] * df["txn_velocity_1h"]
    df["risk_pile_up"] = (
        (df["risk_score_internal"] > df["risk_score_internal"].quantile(0.75)).astype(int) +
        (df["txn_velocity_1h"] >= 2).astype(int) +
        df["location_mismatch"].astype(int) +
        df["new_device"].astype(int) +
        df["device_risk"].astype(int)
    )

    # Drop unnecessary columns
    drop_cols = ["transaction_id", "customer_id", "timestamp", "ip_address", "day_of_week"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Split X / y
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing pipelines with imputers
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, [c for c in NUM_COLS if c in X_train.columns]),
        ("cat", categorical_transformer, [c for c in CAT_COLS if c in X_train.columns])
    ])

    # Pipeline with SMOTE and Logistic Regression
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.3)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    y_pred_tuned = (y_proba >= best_thresh).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)

    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Best threshold (max F1): {best_thresh:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_tuned, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))

    # Save model
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"fraud_model_{timestamp}.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()