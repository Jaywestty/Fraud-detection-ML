"""Train, select, calibrate and evaluate the fraud model.

Protocol (each split is used for exactly one job):
  train      -> fit candidates (with internal CV for probability calibration)
  validation -> pick the best candidate and its decision threshold
  test       -> one final, untouched estimate of the chosen model
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.data import load_data, time_split
from src.evaluate import bootstrap_pr_auc, choose_threshold, evaluate
from src.models import get_candidate_models
from src.pipeline import build_pipeline
from src.schema import RAW_FEATURES, TARGET

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_PATH = PROJECT_DIR / "data" / "nova_pay_transcations.csv"
MODELS_DIR = PROJECT_DIR / "models"
MLFLOW_DIR = PROJECT_DIR / "mlruns"
SEED = 42


def calibrated(classifier):
    return CalibratedClassifierCV(
        build_pipeline(classifier),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("fraud-detection")

    df = load_data(DATA_PATH)
    train, val, test = time_split(df)
    X_train, y_train = train[RAW_FEATURES], train[TARGET]
    X_val, y_val = val[RAW_FEATURES], val[TARGET]
    X_test, y_test = test[RAW_FEATURES], test[TARGET]

    def describe(part):
        ts = pd.to_datetime(part["timestamp"], errors="coerce", utc=True, format="mixed")
        return {
            "rows": int(len(part)),
            "fraud": int(part[TARGET].sum()),
            "from": str(ts.min().date()),
            "to": str(ts.max().date()),
        }

    split_info = {name: describe(p) for name, p in (("train", train), ("validation", val), ("test", test))}
    for name, s in split_info.items():
        print(f"{name:>10}: {s['rows']:>5} rows, {s['fraud']:>3} fraud, {s['from']} -> {s['to']}")

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    candidates = get_candidate_models(pos_weight, random_state=SEED)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results, fitted = {}, {}
    with mlflow.start_run(run_name=f"train-{version}"):
        mlflow.log_params({"seed": SEED, "split": "time 60/20/20", "calibration": "sigmoid-cv3"})

        for name, clf in candidates.items():
            with mlflow.start_run(run_name=name, nested=True):
                print(f"\nTraining {name}...")
                model = calibrated(clf).fit(X_train, y_train)
                val_scores = model.predict_proba(X_val)[:, 1]
                threshold = choose_threshold(y_val, val_scores)
                metrics = evaluate(y_val, val_scores, threshold)

                mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                print(f"  val PR-AUC {metrics['pr_auc']:.4f} | ROC-AUC {metrics['roc_auc']:.4f}"
                      f" | threshold {threshold:.4f}")

                results[name] = {**metrics, "best_threshold": threshold}
                fitted[name] = model

        best_name = max(results, key=lambda n: results[n]["pr_auc"])
        best_model = fitted[best_name]
        threshold = results[best_name]["best_threshold"]

        test_scores = best_model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_scores, threshold)
        ci_low, ci_high = bootstrap_pr_auc(y_test, test_scores)

        print(f"\nBest model (by validation PR-AUC): {best_name}")
        print(f"  test PR-AUC {test_metrics['pr_auc']:.4f} (95% CI {ci_low:.3f}-{ci_high:.3f}),"
              f" base rate {test_metrics['base_rate']:.4f}")
        print(f"  at threshold {threshold:.4f}: precision {test_metrics['precision_at_threshold']:.3f},"
              f" recall {test_metrics['recall_at_threshold']:.3f}")

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({"test_pr_auc_ci_low": ci_low, "test_pr_auc_ci_high": ci_high})

        example = X_val.dropna().head(5)  # fully populated rows so MLflow can infer types
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=infer_signature(example, best_model.predict_proba(example)[:, 1]),
            input_example=example,
        )

        report = {
            "model_version": version,
            "trained_at": version,
            "best_model": best_name,
            "selected_on": "validation_pr_auc",
            "threshold": threshold,
            "best_pr_auc": test_metrics["pr_auc"],
            "test": {**test_metrics, "pr_auc_ci_95": [ci_low, ci_high]},
            "splits": split_info,
            "models": results,
        }
        metrics_path = MODELS_DIR / "best_run_metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(metrics_path))

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    print(f"\nSaved model to {MODELS_DIR / 'best_model.pkl'} and report to {metrics_path}")


if __name__ == "__main__":
    main()
