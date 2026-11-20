import html
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.schema import RAW_FEATURES

logger = logging.getLogger("fraud_api")

PROJECT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = Path(os.getenv("MODEL_DIR", PROJECT_DIR / "models"))
MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_PATH = MODELS_DIR / "best_run_metrics.json"
DEFAULT_THRESHOLD = 0.5

state: dict = {}


def load_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python main.py` first.")
    state["model"] = joblib.load(MODEL_PATH)
    report = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    state["report"] = report
    state["threshold"] = float(report.get("threshold", DEFAULT_THRESHOLD))
    state["model_version"] = report.get("model_version", "unknown")
    if "threshold" not in report:
        logger.warning("No tuned threshold in %s; falling back to %.2f", METRICS_PATH, DEFAULT_THRESHOLD)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    state.clear()


app = FastAPI(title="Fraud Detection API", version="2.0", lifespan=lifespan)


class TransactionInput(BaseModel):
    """Raw transaction fields. Feature engineering happens inside the model pipeline."""

    amount_usd: float = Field(gt=0, description="Transaction amount in USD")
    fee: float = Field(ge=0, description="Fee charged, in USD")
    exchange_rate_src_to_dest: Optional[float] = Field(None, gt=0)
    source_currency: Literal["USD", "GBP", "CAD"]
    dest_currency: Literal["USD", "CAD", "GBP", "EUR", "MXN", "NGN", "INR", "PHP", "CNY"]
    channel: Literal["mobile", "web", "atm"]
    home_country: Literal["US", "UK", "CA"]
    ip_country: Optional[Literal["US", "UK", "CA"]] = Field(
        None, description="Country of the IP address; used for the cross-border feature"
    )
    kyc_tier: Literal["low", "standard", "enhanced"]
    account_age_days: int = Field(ge=0)
    chargeback_history_count: int = Field(ge=0)
    new_device: bool
    location_mismatch: bool
    ip_risk_score: float = Field(ge=0, le=1)
    device_trust_score: float = Field(ge=0, le=1)
    risk_score_internal: float = Field(ge=0, le=1)
    corridor_risk: float = Field(ge=0, le=1)
    txn_velocity_1h: int = Field(ge=0)
    txn_velocity_24h: int = Field(ge=0)
    timestamp: Optional[datetime] = Field(None, description="Defaults to now (UTC)")


class Prediction(BaseModel):
    fraud_probability: float
    prediction: int
    threshold: float
    risk_level: Literal["low", "medium", "high"]
    model_version: str


def to_frame(data: TransactionInput) -> pd.DataFrame:
    row = data.model_dump()
    row["timestamp"] = (row["timestamp"] or datetime.now(timezone.utc)).isoformat()
    return pd.DataFrame([row], columns=RAW_FEATURES)


def risk_level(proba: float, threshold: float) -> str:
    if proba >= threshold:
        return "high"
    if proba >= threshold / 2:
        return "medium"
    return "low"


@app.get("/")
def home():
    return {"status": "API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "model_version": state.get("model_version")}


@app.post("/predict", response_model=Prediction)
def predict(data: TransactionInput):
    try:
        proba = float(state["model"].predict_proba(to_frame(data))[0, 1])
    except Exception as exc:  # surface model errors as 500 with a clear message
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    threshold = state["threshold"]
    logger.info("prediction proba=%.4f version=%s", proba, state["model_version"])
    return Prediction(
        fraud_probability=proba,
        prediction=int(proba >= threshold),
        threshold=threshold,
        risk_level=risk_level(proba, threshold),
        model_version=state["model_version"],
    )


@app.get("/model-info")
def model_info():
    if not state.get("report"):
        return {"error": "No training metrics found. Run main.py to generate a training report."}
    return state["report"]


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    report = state.get("report")
    if not report:
        return "<h2>No training metrics found. Run main.py to generate a report.</h2>"

    rows = ""
    for name, m in report["models"].items():
        highlight = "background-color:#e6ffe6;" if name == report["best_model"] else ""
        rows += (
            f'<tr style="{highlight}"><td>{html.escape(name)}</td>'
            f"<td>{m['pr_auc']:.4f}</td><td>{m.get('roc_auc', float('nan')):.4f}</td>"
            f"<td>{m['best_threshold']:.4f}</td></tr>"
        )
    test = report.get("test", {})
    ci = test.get("pr_auc_ci_95", [float("nan"), float("nan")])
    return f"""
    <html>
    <head><title>Fraud Detection Model Dashboard</title></head>
    <body style="font-family:sans-serif; padding:2rem; max-width:860px;">
        <h1>Fraud Detection Model Dashboard</h1>
        <p>Selected model: <strong>{html.escape(report['best_model'])}</strong>
           (version {html.escape(str(report.get('model_version')))})</p>
        <p>Test PR-AUC: <strong>{report['best_pr_auc']:.4f}</strong>
           (95% CI {ci[0]:.3f}-{ci[1]:.3f}; base rate {test.get('base_rate', float('nan')):.4f})</p>
        <p>Decision threshold: {report.get('threshold', DEFAULT_THRESHOLD):.4f}</p>
        <h3>Validation results (used for model selection)</h3>
        <table border="1" cellpadding="8" cellspacing="0">
            <tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Threshold</th></tr>
            {rows}
        </table>
    </body>
    </html>
    """
