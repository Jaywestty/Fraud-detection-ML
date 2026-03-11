# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
MODEL_PATH = Path("models/fraud_model_20260311_164054.pkl")
model = joblib.load(MODEL_PATH)

st.title("Fraud Detection App")
st.write("Enter transaction details to predict fraud (Yes/No).")

# Dropdown options
HOME_COUNTRIES = ['US', 'CA', 'UK', 'unknown']
SOURCE_CURRENCIES = ['USD', 'CAD', 'GBP']
DEST_CURRENCIES = ['CAD', 'MXN', 'CNY', 'EUR', 'INR', 'GBP', 'PHP', 'NGN', 'USD']
CHANNELS = ['web', 'mobile', 'ATM']
KYC_TIERS = ['low', 'standard', 'enhanced']

# Numeric inputs
amount_usd = st.number_input("Amount (USD)", 0.0, 1_000_000.0, 100.0)
fee = st.number_input("Fee", 0.0, 1000.0, 0.0)
exchange_rate_src_to_dest = st.number_input("Exchange Rate (Src -> Dest)", 0.0, 1000.0, 1.0)
ip_risk_score = st.number_input("IP Risk Score", 0.0, 1.0, 0.0)
account_age_days = st.number_input("Account Age (Days)", 0, 3650, 30)
device_trust_score = st.number_input("Device Trust Score", 0.0, 1.0, 0.5)
chargeback_history_count = st.number_input("Customer Chargebacks", 0, 100, 0)
risk_score_internal = st.number_input("Internal Risk Score", 0.0, 1.0, 0.0)
txn_velocity_1h = st.number_input("Transactions last 1h", 0, 1000, 0)
txn_velocity_24h = st.number_input("Transactions last 24h", 0, 10_000, 0)
corridor_risk = st.number_input("Corridor Risk", 0.0, 1.0, 0.0)

# Dropdowns for categorical
home_country = st.selectbox("Customer Home Country", HOME_COUNTRIES)
source_currency = st.selectbox("Source Currency", SOURCE_CURRENCIES)
dest_currency = st.selectbox("Destination Currency", DEST_CURRENCIES)
channel = st.selectbox("Transaction Channel", CHANNELS)
kyc_tier = st.selectbox("KYC Tier", KYC_TIERS)
new_device = st.selectbox("New Device?", ["No", "Yes"])
location_mismatch = st.selectbox("Location Mismatch?", ["No", "Yes"])

# Convert Yes/No to 0/1
new_device_flag = 1 if new_device == "Yes" else 0
location_mismatch_flag = 1 if location_mismatch == "Yes" else 0

# Build input dataframe
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
    "new_device": new_device_flag,
    "location_mismatch": location_mismatch_flag
}])

# Feature Engineering
input_df["cross_border"] = (input_df["home_country"] != "US").astype(int)
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

# Fill missing model columns
MODEL_COLUMNS = model.named_steps["preprocessor"].transformers_[0][2] + model.named_steps["preprocessor"].transformers_[1][2]
for col in MODEL_COLUMNS:
    if col not in input_df.columns:
        input_df[col] = 0

# Prediction
if st.button("Predict Fraud"):
    pred = model.predict(input_df)[0]  
    prob = model.predict_proba(input_df)[0, 1]  

    if pred == 1:
        label = "Yes"
        message = f"⚠️ This transaction is likely fraudulent. Be cautious! (Probability: {prob:.2%})"
    else:
        label = "No"
        message = f"✅ This transaction seems safe. (Probability of fraud: {prob:.2%})"

    st.subheader(f"Fraud Prediction: {label}")
    st.write(message)