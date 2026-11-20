import importlib

import pytest
from fastapi.testclient import TestClient

VALID = {
    "amount_usd": 250.0, "fee": 3.0, "source_currency": "USD", "dest_currency": "NGN",
    "channel": "mobile", "home_country": "US", "ip_country": "US", "kyc_tier": "standard",
    "account_age_days": 400, "chargeback_history_count": 0, "new_device": False,
    "location_mismatch": False, "ip_risk_score": 0.2, "device_trust_score": 0.8,
    "risk_score_internal": 0.2, "corridor_risk": 0.0, "txn_velocity_1h": 0, "txn_velocity_24h": 1,
}


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    import api.app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True and body["model_version"] == "test"


def test_predict_returns_probability_and_uses_tuned_threshold(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["threshold"] == 0.7
    assert body["prediction"] == int(body["fraud_probability"] >= 0.7)
    assert body["risk_level"] in {"low", "medium", "high"}


def test_optional_fields_can_be_omitted(client):
    payload = {k: v for k, v in VALID.items() if k != "ip_country"}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("kyc_tier", "tier_1"), ("channel", "ATM"), ("ip_risk_score", 1.5), ("amount_usd", -5),
])
def test_invalid_inputs_are_rejected(client, field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_riskier_transaction_scores_higher(client):
    risky = {**VALID, "new_device": True, "location_mismatch": True, "ip_country": "UK",
             "device_trust_score": 0.1, "risk_score_internal": 0.55, "txn_velocity_1h": 6,
             "account_age_days": 5, "ip_risk_score": 0.9, "amount_usd": 5000.0}
    p_safe = client.post("/predict", json=VALID).json()["fraud_probability"]
    p_risky = client.post("/predict", json=risky).json()["fraud_probability"]
    assert p_risky > p_safe


def test_model_info_and_dashboard(client):
    assert client.get("/model-info").json()["best_model"] == "LogisticRegression"
    assert "Fraud Detection Model Dashboard" in client.get("/dashboard").text


def test_streamlit_options_match_training_schema():
    """The Streamlit form keeps its own copy of the categories; keep them in sync."""
    import ast
    from pathlib import Path

    from src import schema

    src = (Path(__file__).parent.parent / "streamlit_app" / "streamlit_app.py").read_text(encoding="utf-8")
    consts = {
        t.id: ast.literal_eval(node.value)
        for node in ast.parse(src).body if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()
        and isinstance(node.value, ast.List)
    }
    for name in ("COUNTRIES", "SOURCE_CURRENCIES", "DEST_CURRENCIES", "CHANNELS", "KYC_TIERS"):
        assert consts[name] == getattr(schema, name), name
