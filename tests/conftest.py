import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import load_data
from src.pipeline import build_pipeline
from src.schema import RAW_FEATURES, TARGET

DATA_PATH = Path(__file__).parent.parent / "data" / "nova_pay_transcations.csv"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_data(DATA_PATH)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, raw_df):
    """A small model trained on the real data, saved the same way main.py saves it."""
    d = tmp_path_factory.mktemp("models")
    model = build_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    model.fit(raw_df[RAW_FEATURES], raw_df[TARGET])
    joblib.dump(model, d / "best_model.pkl")
    report = {
        "model_version": "test", "trained_at": "test", "best_model": "LogisticRegression",
        "threshold": 0.7, "best_pr_auc": 0.1,
        "test": {"base_rate": 0.02, "pr_auc_ci_95": [0.05, 0.15]},
        "models": {"LogisticRegression": {"pr_auc": 0.1, "roc_auc": 0.7, "best_threshold": 0.7}},
    }
    (d / "best_run_metrics.json").write_text(json.dumps(report))
    return d
