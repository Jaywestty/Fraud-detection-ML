# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
MODEL_PATH = Path("models/fraud_model_20260311_164054.pkl")
model = joblib.load(MODEL_PATH)

# Streamlit UI
st.title("Fraud Detection App")
st.write("Enter transaction details to predict fraud Score.")

# User Inputs
amount_usd = st.number_input("Amount (USD)", min_value=0.0, value=100.0)
fee = st.number_input("Fee", min_value=0.0, value=0.0)
exchange_rate_src_to_dest = st.number_input("Exchange Rate (Src -> Dest)", min_value=0.0, value=1.0)
ip_risk_score = st.number_input("IP Risk Score", min_value=0.0, max_value=1.0, value=0.0)
account_age_days = st.number_input("Account Age (Days)", min_value=0, value=30)
device_trust_score = st.number_input("Device Trust Score", min_value=0.0, max_value=1.0, value=0.5)
chargeback_history_count = st.number_input("Customer Chargebacks", min_value=0, value=0)
risk_score_internal = st.number_input("Internal Risk Score", min_value=0.0, max_value=1.0, value=0.0)
txn_velocity_1h = st.number_input("Transactions in last 1h", min_value=0, value=0)
txn_velocity_24h = st.number_input("Transactions in last 24h", min_value=0, value=0)
corridor_risk = st.number_input("Corridor Risk", min_value=0.0, max_value=1.0, value=0.0)

home_country = st.text_input("Customer Home Country", "US")
source_currency = st.text_input("Source Currency", "USD")
dest_currency = st.text_input("Destination Currency", "USD")
channel = st.selectbox("Transaction Channel", ["web", "mobile", "ATM"])
kyc_tier = st.selectbox("KYC Tier", ["low", "standard", "enhanced"])
new_device = st.selectbox("Is it a new device?", [0, 1])
location_mismatch = st.selectbox("Location mismatch?", [0, 1])

# Construct DataFrame
input_df = pd.DataFrame([{
    "amount_usd": amount_usd,
    "fee": fee,
    "exchange_rate_src_to_dest": exchange_rate_src_to_dest,
    "ip_risk_score": ip_risk_score,
    "account_age_days": account_age_days,
    "device_trust_score": device_trust_score,
    "chargeback_history_count": chargeback_history_count,
    "risk_score_internal": risk_score_internal,
    "txn_velocity_1h": txn_velocity_1h,
    "txn_velocity_24h": txn_velocity_24h,
    "corridor_risk": corridor_risk,
    "home_country": home_country,
    "source_currency": source_currency,
    "dest_currency": dest_currency,
    "channel": channel,
    "kyc_tier": kyc_tier,
    "new_device": new_device,
    "location_mismatch": location_mismatch
}])

# Feature Engineering
input_df["cross_border"] = (input_df["home_country"] != "US").astype(int)  # Example: compare to your default IP country if needed
input_df["device_risk"] = ((input_df["new_device"] == 1) & (input_df["device_trust_score"] < 0.5)).astype(int)
input_df["velocity_ratio"] = input_df["txn_velocity_1h"] / (input_df["txn_velocity_24h"] + 1)
input_df["velocity_x_amount"] = input_df["velocity_ratio"] * input_df["amount_usd"]
input_df["new_account_flag"] = (input_df["account_age_days"] < 60).astype(int)
input_df["young_account_velocity"] = input_df["new_account_flag"] * input_df["txn_velocity_1h"]
input_df["risk_pile_up"] = (
    (input_df["risk_score_internal"] > 0.75).astype(int) +
    (input_df["txn_velocity_1h"] >= 2).astype(int) +
    input_df["location_mismatch"] +
    input_df["new_device"] +
    input_df["device_risk"]
)

# Fill missing columns the model expects
MODEL_COLUMNS = model.named_steps["preprocessor"].transformers_[0][2] + model.named_steps["preprocessor"].transformers_[1][2]
for col in MODEL_COLUMNS:
    if col not in input_df.columns:
        input_df[col] = 0

# Predict
if st.button("Predict Fraud"):
    # Direct prediction
    pred = model.predict(input_df)[0]  # 0 or 1
    pred_label = "Yes" if pred == 1 else "No"
    st.write(f"Fraud: {pred_label}")