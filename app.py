import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page Config
st.set_page_config(page_title="Fraud Detection Pro", page_icon="🛡️", layout="centered")

# Load trained model
MODEL_PATH = Path("models/fraud_model_20260311_164054.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Header
st.title("🛡️ Fraud Detection Analysis")
st.markdown("---")

# --- INPUT SECTION ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Transaction")
    amount_usd = st.number_input("Amount (USD)", 0.0, 1_000_000.0, 100.0)
    source_currency = st.selectbox("Source Currency", ['USD', 'CAD', 'GBP'])
    dest_currency = st.selectbox("Destination Currency", ['CAD', 'MXN', 'CNY', 'EUR', 'INR', 'GBP', 'PHP', 'NGN', 'USD'])
    channel = st.selectbox("Channel", ['web', 'mobile', 'ATM'])
    fee = st.number_input("Transaction Fee", 0.0, 1000.0, 0.0)

with col2:
    st.subheader("👤 Customer")
    home_country = st.selectbox("Home Country", ['US', 'CA', 'UK', 'unknown'])
    kyc_tier = st.selectbox("KYC Tier", ['low', 'standard', 'enhanced'])
    account_age_days = st.number_input("Account Age (Days)", 0, 3650, 30)
    chargeback_history_count = st.number_input("Prev. Chargebacks", 0, 100, 0)
    # Using horizontal radio for better UX
    new_device = st.radio("New Device?", ["No", "Yes"], horizontal=True)
    location_mismatch = st.radio("Location Mismatch?", ["No", "Yes"], horizontal=True)

st.markdown("---")

# Risk Scores Section
st.subheader("📊 Risk & Velocity Metrics")
r_col1, r_col2 = st.columns(2)

with r_col1:
    ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.0)
    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
    risk_score_internal = st.slider("Internal Risk Score", 0.0, 1.0, 0.0)

with r_col2:
    corridor_risk = st.slider("Corridor Risk", 0.0, 1.0, 0.0)
    txn_velocity_1h = st.number_input("Transactions (Last 1h)", 0, 1000, 0)
    txn_velocity_24h = st.number_input("Transactions (Last 24h)", 0, 10_000, 0)

# Fixed logic variables (not displayed to user to keep UI clean)
exchange_rate_src_to_dest = 1.0 
new_device_flag = 1 if new_device == "Yes" else 0
location_mismatch_flag = 1 if location_mismatch == "Yes" else 0

# Build Dataframe
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

# Feature Engineering (Preserving all logic)
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

# Align with model
MODEL_COLUMNS = model.named_steps["preprocessor"].transformers_[0][2] + model.named_steps["preprocessor"].transformers_[1][2]
for col in MODEL_COLUMNS:
    if col not in input_df.columns:
        input_df[col] = 0

# --- PREDICTION ---
st.markdown("---")
if st.button("Run Fraud Analysis", use_container_width=True, type="primary"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0, 1]

    if pred == 1:
        st.error(f"### 🚩 Prediction: FRAUD DETECTED")
        st.metric("Fraud Probability", f"{prob:.2%}")
        st.write("This transaction matches high-risk fraud patterns.")
    else:
        st.success(f"### ✅ Prediction: CLEAR")
        st.metric("Fraud Probability", f"{prob:.2%}")
        st.write("This transaction appears legitimate.")