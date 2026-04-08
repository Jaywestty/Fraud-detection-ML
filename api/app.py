from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Fraud Detection API")

# Absolute path for model
PROJECT_DIR = Path(__file__).parent.parent.resolve()
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pkl"
model = joblib.load(MODEL_PATH)

# --- Input Schema ---
class TransactionInput(BaseModel):
    amount_usd: float
    fee: float
    source_currency: str
    dest_currency: str
    channel: str
    home_country: str
    kyc_tier: str
    account_age_days: int
    chargeback_history_count: int
    new_device: bool
    location_mismatch: bool
    ip_risk_score: float
    device_trust_score: float
    risk_score_internal: float
    corridor_risk: float
    txn_velocity_1h: int
    txn_velocity_24h: int

# --- Helper to build features ---
def build_features(data: TransactionInput) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])

    # Flags
    df["new_device"] = df["new_device"].astype(int)
    df["location_mismatch"] = df["location_mismatch"].astype(int)

    # Derived features
    df["cross_border"] = (df["home_country"] != "US").astype(int)
    df["device_risk"] = ((df["new_device"] == 1) & (df["device_trust_score"] < 0.5)).astype(int)
    df["velocity_ratio"] = df["txn_velocity_1h"] / (df["txn_velocity_24h"] + 1)
    df["velocity_x_amount"] = df["velocity_ratio"] * df["amount_usd"]
    df["new_account_flag"] = (df["account_age_days"] < 60).astype(int)
    df["young_account_velocity"] = df["new_account_flag"] * df["txn_velocity_1h"]
    df["risk_pile_up"] = (
        (df["risk_score_internal"] > 0.75).astype(int) +
        (df["txn_velocity_1h"] >= 2).astype(int) +
        df["location_mismatch"] +
        df["new_device"] +
        df["device_risk"]
    )

    # Align columns with trained model
    try:
        MODEL_COLUMNS = model.named_steps["preprocessor"].transformers_[0][2] + model.named_steps["preprocessor"].transformers_[1][2]
    except Exception:
        # fallback if preprocessor not found
        MODEL_COLUMNS = df.columns.tolist()

    for col in MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df

# --- Endpoints ---
@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: TransactionInput):
    df = build_features(data)
    proba = model.predict_proba(df)[0, 1]
    prediction = int(proba >= 0.5)
    return {"fraud_probability": float(proba), "prediction": prediction}