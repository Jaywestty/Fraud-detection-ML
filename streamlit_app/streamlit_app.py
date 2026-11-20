import os
from datetime import datetime, time, timezone

import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Must match src/schema.py (tests/test_api.py enforces this). Kept as a copy so
# the Streamlit Cloud deploy doesn't need the training code or its dependencies.
COUNTRIES = ["US", "UK", "CA"]
SOURCE_CURRENCIES = ["USD", "GBP", "CAD"]
DEST_CURRENCIES = ["USD", "CAD", "GBP", "EUR", "MXN", "NGN", "INR", "PHP", "CNY"]
CHANNELS = ["mobile", "web", "atm"]
KYC_TIERS = ["low", "standard", "enhanced"]

PRESETS = {
    "Typical customer": dict(
        amount_usd=150.0, fee=3.3, source_currency="USD", dest_currency="CAD", channel="mobile",
        home_country="US", ip_country="US", kyc_tier="standard", account_age_days=420,
        chargeback_history_count=0, new_device=False, location_mismatch=False,
        ip_risk_score=0.25, device_trust_score=0.85, risk_score_internal=0.20,
        corridor_risk=0.0, txn_velocity_1h=0, txn_velocity_24h=1, txn_hour=14,
    ),
    "Suspicious pattern": dict(
        amount_usd=2400.0, fee=12.0, source_currency="GBP", dest_currency="NGN", channel="web",
        home_country="UK", ip_country="US", kyc_tier="low", account_age_days=12,
        chargeback_history_count=1, new_device=True, location_mismatch=True,
        ip_risk_score=0.85, device_trust_score=0.15, risk_score_internal=0.50,
        corridor_risk=0.25, txn_velocity_1h=5, txn_velocity_24h=7, txn_hour=3,
    ),
}


def get_api_base() -> str | None:
    url = os.getenv("API_URL")
    if not url:
        try:
            url = st.secrets.get("API_URL")
        except Exception:
            url = None
    if not url:
        return None
    url = url.rstrip("/")
    return url[: -len("/predict")] if url.endswith("/predict") else url


@st.cache_data(ttl=300, show_spinner=False)
def fetch_json(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def apply_preset():
    for key, value in PRESETS[st.session_state.preset].items():
        st.session_state[key] = value


def gauge(proba: float, threshold: float) -> go.Figure:
    top = max(0.25, min(1.0, threshold * 4, proba * 1.25))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, top * 100], "ticksuffix": "%"},
            "bar": {"color": "#1f2937"},
            "steps": [
                {"range": [0, threshold * 50], "color": "#d1fae5"},
                {"range": [threshold * 50, threshold * 100], "color": "#fef3c7"},
                {"range": [threshold * 100, top * 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#dc2626", "width": 3}, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=230, margin=dict(t=20, b=0, l=30, r=30))
    return fig


def input_signals(p: dict) -> list[str]:
    """Plain-language red flags present in the input (not a model explanation)."""
    s = []
    if p["new_device"] and p["device_trust_score"] < 0.5:
        s.append("New device with low trust score")
    elif p["new_device"]:
        s.append("New device")
    if p["location_mismatch"]:
        s.append("Location differs from the account's usual location")
    if p.get("ip_country") and p["ip_country"] != p["home_country"]:
        s.append(f"IP country ({p['ip_country']}) differs from home country ({p['home_country']})")
    if p["account_age_days"] < 60:
        s.append(f"Young account ({p['account_age_days']} days)")
    if p["txn_velocity_1h"] >= 2:
        s.append(f"{p['txn_velocity_1h']} transactions in the last hour")
    if p["ip_risk_score"] >= 0.7:
        s.append("High-risk IP address")
    if p["chargeback_history_count"] > 0:
        s.append("Previous chargebacks")
    if p["kyc_tier"] == "low":
        s.append("Low KYC verification tier")
    return s


st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

api_base = get_api_base()
if not api_base:
    st.error("`API_URL` is not set. Add it to a `.env` file or Streamlit secrets, e.g. "
             "`API_URL=http://127.0.0.1:8000`.")
    st.stop()

for key, value in PRESETS["Typical customer"].items():
    st.session_state.setdefault(key, value)

# ---------------- Sidebar: service + model card ----------------
with st.sidebar:
    st.header("Model")
    try:
        health = fetch_json(f"{api_base}/health")
        st.success(f"API online · model `{health.get('model_version')}`")
    except requests.RequestException:
        health = None
        st.error("API unreachable. It may be waking up; retry in a minute.")

    info = None
    try:
        info = fetch_json(f"{api_base}/model-info")
    except requests.RequestException:
        pass

    if info and "error" not in info:
        test = info.get("test", {})
        st.metric("Selected model", info["best_model"])
        c1, c2 = st.columns(2)
        c1.metric("Test PR-AUC", f"{info['best_pr_auc']:.3f}")
        c2.metric("Base rate", f"{test.get('base_rate', 0):.2%}")
        ci = test.get("pr_auc_ci_95")
        if ci:
            st.caption(f"95% CI for PR-AUC: {ci[0]:.3f}–{ci[1]:.3f} "
                       f"({test.get('n_fraud', '?')} fraud cases in the test period).")
        if "recall_at_top_5pct" in test:
            st.caption(f"Reviewing the riskiest 5% of transactions catches "
                       f"**{test['recall_at_top_5pct']:.0%}** of fraud in the test period.")
        st.caption(f"Alert threshold: {info.get('threshold', 0.5):.1%}")
    st.divider()
    st.markdown(f"[API docs]({api_base}/docs) · [Dashboard]({api_base}/dashboard)")

# ---------------- Main ----------------
st.title("🔍 Transaction fraud check")
st.write("Score a single transaction with the deployed model. Start from a preset, adjust any field, "
         "then run the check.")

st.radio("Start from a preset", list(PRESETS), key="preset", horizontal=True, on_change=apply_preset)

with st.form("transaction_form"):
    tx, acct = st.columns(2, gap="large")
    dev, risk = st.columns(2, gap="large")

    with tx:
        st.subheader("Transaction")
        st.number_input("Amount (USD)", min_value=0.01, step=10.0, key="amount_usd")
        st.number_input("Fee (USD)", min_value=0.0, step=0.5, key="fee")
        st.selectbox("Source currency", SOURCE_CURRENCIES, key="source_currency")
        st.selectbox("Destination currency", DEST_CURRENCIES, key="dest_currency")
        st.selectbox("Channel", CHANNELS, key="channel", format_func=str.upper)
        st.slider("Hour of day (UTC)", 0, 23, key="txn_hour")

    with acct:
        st.subheader("Account")
        st.selectbox("Home country", COUNTRIES, key="home_country")
        st.selectbox("KYC tier", KYC_TIERS, key="kyc_tier", format_func=str.title,
                     help="Know-Your-Customer verification level.")
        st.number_input("Account age (days)", min_value=0, step=1, key="account_age_days")
        st.number_input("Past chargebacks", min_value=0, step=1, key="chargeback_history_count")

    with dev:
        st.subheader("Device & network")
        st.selectbox("IP country", COUNTRIES, key="ip_country")
        st.toggle("New device", key="new_device")
        st.toggle("Location mismatch", key="location_mismatch")
        st.slider("IP risk score", 0.0, 1.0, step=0.01, key="ip_risk_score",
                  help="0 = clean IP, 1 = known-bad IP.")
        st.slider("Device trust score", 0.0, 1.0, step=0.01, key="device_trust_score",
                  help="0 = untrusted, 1 = fully trusted device.")

    with risk:
        st.subheader("Risk & velocity")
        st.slider("Internal risk score", 0.0, 1.0, step=0.01, key="risk_score_internal")
        st.slider("Corridor risk", 0.0, 1.0, step=0.01, key="corridor_risk",
                  help="Risk of this source → destination corridor.")
        st.number_input("Transactions in last 1h", min_value=0, step=1, key="txn_velocity_1h")
        st.number_input("Transactions in last 24h", min_value=0, step=1, key="txn_velocity_24h")

    submitted = st.form_submit_button("Check transaction", type="primary", use_container_width=True)

if submitted:
    ss = st.session_state
    when = datetime.combine(datetime.now(timezone.utc).date(), time(ss.txn_hour), tzinfo=timezone.utc)
    payload = {
        k: ss[k] for k in PRESETS["Typical customer"] if k != "txn_hour"
    } | {"timestamp": when.isoformat()}

    if payload["txn_velocity_24h"] < payload["txn_velocity_1h"]:
        st.warning("Transactions in the last 24h can't be fewer than in the last hour.")
        st.stop()

    try:
        with st.spinner("Scoring..."):
            response = requests.post(f"{api_base}/predict", json=payload, timeout=15)
        if response.status_code == 422:
            st.error("The API rejected the input:")
            st.json(response.json().get("detail"))
            st.stop()
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        st.stop()

    proba, threshold = result["fraud_probability"], result["threshold"]
    level = result.get("risk_level", "high" if result["prediction"] else "low")

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(proba, threshold), use_container_width=True)
    with right:
        verdict = {
            "high": ("🚨 Flag for review", st.error),
            "medium": ("⚠️ Elevated risk, monitor", st.warning),
            "low": ("✅ Looks legitimate", st.success),
        }[level]
        verdict[1](f"**{verdict[0]}** — fraud probability {proba:.1%} (alert threshold {threshold:.1%})")
        base = (info or {}).get("test", {}).get("base_rate")
        if base:
            st.metric("Risk vs. an average transaction", f"{proba / base:.1f}×",
                      help=f"Average fraud rate in the test period: {base:.2%}")

        signals = input_signals(payload)
        st.markdown("**Red flags in this input**")
        if signals:
            st.markdown("\n".join(f"- {s}" for s in signals))
        else:
            st.caption("None of the common red flags are present.")
        st.caption("Red flags are simple rules on your inputs, shown for context. "
                   "The score itself comes from the model.")

if info and "error" not in info:
    with st.expander("How the candidate models compared"):
        names = list(info["models"])
        values = [info["models"][m]["pr_auc"] for m in names]
        fig = go.Figure(go.Bar(
            x=names, y=values, text=[f"{v:.3f}" for v in values], textposition="outside",
            marker_color=["#16a34a" if m == info["best_model"] else "#94a3b8" for m in names],
        ))
        base = info.get("test", {}).get("base_rate")
        if base:
            fig.add_hline(y=base, line_dash="dot", annotation_text="random guess (base rate)")
        fig.update_layout(yaxis_title="Validation PR-AUC", height=300, showlegend=False,
                          margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Models are compared on a validation period. The selected model is then "
                   "scored once on a later, untouched test period.")
