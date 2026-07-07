import streamlit as st
import requests
import os
from datetime import datetime
import plotly.graph_objects as go

from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
MODEL_INFO_URL = API_URL.rsplit("/predict", 1)[0] + "/model-info"


def format_trained_at(raw_timestamp):
    try:
        parsed = datetime.strptime(raw_timestamp, "%Y%m%d_%H%M%S")
        return parsed.strftime("%B %d, %Y at %I:%M %p")
    except (ValueError, TypeError):
        return raw_timestamp


st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("Fraud Detection")
st.write(
    "This app checks a financial transaction for fraud risk in real time. "
    "A trained classifier scores the transaction and returns a fraud probability. "
    "Predictions are served by a FastAPI backend hosted on Hugging Face Spaces."
)

with st.expander("Model performance dashboard", expanded=False):
    try:
        info_response = requests.get(MODEL_INFO_URL, timeout=10)
        info_response.raise_for_status()
        info = info_response.json()

        if "error" in info:
            st.warning(info["error"])
        else:
            col1, col2, col3 = st.columns(3)
            metric_style = "font-size:1rem; font-weight:600; margin-top:2px;"

            col1.markdown(f"**Best Model**<div style='{metric_style}'>{info['best_model']}</div>", unsafe_allow_html=True)
            col2.markdown(f"**Best PR-AUC**<div style='{metric_style}'>{info['best_pr_auc']:.4f}</div>", unsafe_allow_html=True)
            col3.markdown(f"**Last Trained**<div style='{metric_style}'>{format_trained_at(info['trained_at'])}</div>", unsafe_allow_html=True)

            model_names = list(info["models"].keys())
            pr_auc_values = [info["models"][m]["pr_auc"] for m in model_names]
            colors = [
                "#2ecc71" if m == info["best_model"] else "#3498db"
                for m in model_names
            ]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=model_names,
                        y=pr_auc_values,
                        marker_color=colors,
                        text=[f"{v:.4f}" for v in pr_auc_values],
                        textposition="outside"
                    )
                ]
            )
            fig.update_layout(
                title="PR-AUC by Model",
                yaxis_title="PR-AUC",
                showlegend=False,
                height=280,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load model metrics: {e}")

st.divider()
st.subheader("Transaction details")
st.caption("Enter the transaction information below. Hover the (?) icons for field guidance.")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount_usd = st.number_input(
            "Amount (USD)", min_value=0.0, value=100.0,
            help="Total value of the transaction in US dollars."
        )
        fee = st.number_input(
            "Fee", min_value=0.0, value=1.0,
            help="Transaction fee charged, in US dollars."
        )
        source_currency = st.text_input(
            "Source Currency", value="USD",
            help="Currency the sender is transacting in, e.g. USD, NGN, EUR."
        )
        dest_currency = st.text_input(
            "Destination Currency", value="USD",
            help="Currency the recipient will receive, e.g. USD, NGN, EUR."
        )
        channel = st.text_input(
            "Channel", value="mobile",
            help="Platform used to initiate the transaction, e.g. mobile, web, atm."
        )
        home_country = st.text_input(
            "Home Country", value="US",
            help="Country the sender's account is registered in."
        )
        kyc_tier = st.text_input(
            "KYC Tier", value="tier_1",
            help="Know Your Customer verification level of the sender's account."
        )
        account_age_days = st.number_input(
            "Account Age (days)", min_value=0, value=180,
            help="Number of days since the sender's account was created."
        )

    with col2:
        chargeback_history_count = st.number_input(
            "Chargeback History Count", min_value=0, value=0,
            help="Number of past chargebacks or disputes on this account."
        )
        new_device = st.checkbox(
            "New Device",
            help="Check if this transaction is from a device not previously used on this account."
        )
        location_mismatch = st.checkbox(
            "Location Mismatch",
            help="Check if the transaction location differs from the account's usual location."
        )
        ip_risk_score = st.slider(
            "IP Risk Score", 0.0, 1.0, 0.1,
            help="Risk score of the IP address used, from 0 (safe) to 1 (high risk)."
        )
        device_trust_score = st.slider(
            "Device Trust Score", 0.0, 1.0, 0.8,
            help="Trust score of the device used, from 0 (untrusted) to 1 (fully trusted)."
        )
        risk_score_internal = st.slider(
            "Internal Risk Score", 0.0, 1.0, 0.2,
            help="Internal risk score assigned by upstream fraud systems."
        )
        corridor_risk = st.slider(
            "Corridor Risk", 0.0, 1.0, 0.1,
            help="Risk score associated with this specific source-destination country pair."
        )
        txn_velocity_1h = st.number_input(
            "Transactions (1h)", min_value=0, value=1,
            help="Number of transactions made by this account in the last hour."
        )
        txn_velocity_24h = st.number_input(
            "Transactions (24h)", min_value=0, value=3,
            help="Number of transactions made by this account in the last 24 hours."
        )

    submitted = st.form_submit_button("Check for Fraud")

if submitted:
    payload = {
        "amount_usd": amount_usd,
        "fee": fee,
        "source_currency": source_currency,
        "dest_currency": dest_currency,
        "channel": channel,
        "home_country": home_country,
        "kyc_tier": kyc_tier,
        "account_age_days": account_age_days,
        "chargeback_history_count": chargeback_history_count,
        "new_device": new_device,
        "location_mismatch": location_mismatch,
        "ip_risk_score": ip_risk_score,
        "device_trust_score": device_trust_score,
        "risk_score_internal": risk_score_internal,
        "corridor_risk": corridor_risk,
        "txn_velocity_1h": txn_velocity_1h,
        "txn_velocity_24h": txn_velocity_24h
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.divider()
        proba = result["fraud_probability"]

        if result["prediction"] == 1:
            st.error(f"Fraud detected. Probability: {proba:.2%}")
        else:
            st.success(f"Transaction looks legitimate. Fraud probability: {proba:.2%}")

        st.progress(min(proba, 1.0))
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")