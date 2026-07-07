# main.py
import joblib
from pathlib import Path
from datetime import datetime

import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import feature_engineering
from src.pipeline import build_pipeline
from src.models import get_candidate_models
from src.train import train_model
from src.evaluate import evaluate_model

TARGET = "is_fraud"

def main():
    # Paths
    PROJECT_DIR = Path(__file__).parent.resolve()
    DATA_PATH = PROJECT_DIR / "data" / "nova_pay_transcations.csv"
    MODELS_DIR = PROJECT_DIR / "models"
    MLFLOW_DIR = PROJECT_DIR / "mlruns"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    # MLflow tracking
    tracking_uri = f"file:///{MLFLOW_DIR.as_posix()}"  # forward slashes + file:///
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection")

    # Load and preprocess data
    df = load_data(DATA_PATH)
    df = feature_engineering(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Candidate models
    candidate_models = get_candidate_models()
    best_model = None
    best_score = 0
    best_name = ""

    # Start MLflow run
    all_model_metrics = {}

    with mlflow.start_run():
        for name, classifier in candidate_models.items():
            print(f"\nTraining {name}...")
            pipeline = build_pipeline(X_train, classifier)
            pipeline = train_model(pipeline, X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            print(f"{name} PR-AUC: {metrics['pr_auc']:.4f}")
            mlflow.log_metric(f"{name}_pr_auc", metrics["pr_auc"])

            all_model_metrics[name] = {
                "pr_auc": float(metrics["pr_auc"]),
                "best_threshold": float(metrics["best_threshold"])
            }

            if metrics["pr_auc"] > best_score:
                best_score = metrics["pr_auc"]
                best_model = pipeline
                best_name = name

        print(f"\nBest model: {best_name} with PR-AUC = {best_score:.4f}")
        mlflow.log_param("best_model", best_name)
        mlflow.sklearn.log_model(best_model, "best_model")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_model_path = MODELS_DIR / f"fraud_model_{timestamp}.pkl"
        prod_model_path = MODELS_DIR / "best_model.pkl"

        joblib.dump(best_model, versioned_model_path)
        joblib.dump(best_model, prod_model_path)

        print(f"Saved versioned model to {versioned_model_path}")
        print(f"Saved production model to {prod_model_path}")

        metrics_report = {
            "best_model": best_name,
            "best_pr_auc": float(best_score),
            "trained_at": timestamp,
            "models": all_model_metrics
        }
        metrics_path = MODELS_DIR / "best_run_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_report, f, indent=2)

        print(f"Saved run metrics to {metrics_path}")

if __name__ == "__main__":
    main()