---
title: Fraud Detection API
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---

# Fraud Detection ML System

An end-to-end fraud detection system that scores financial transactions in real time. It covers the full path from raw, messy data to a deployed model: cleaning, leakage-free feature engineering, time-aware evaluation, experiment tracking, a typed prediction API, and a user-facing app, all deployed as connected services with tests gating every deploy.

- Live prediction app (Streamlit): https://fraud-detect-mll.streamlit.app
- Live API (FastAPI on Hugging Face Spaces): https://jayywestty-frauddetection-ml.hf.space ([docs](https://jayywestty-frauddetection-ml.hf.space/docs))
- Model dashboard: https://jayywestty-frauddetection-ml.hf.space/dashboard

## For non-technical readers

This is a fraud checker. You enter details about a money transfer (amount, device, account age, and so on) and the system estimates how likely it is to be fraud. It compares that estimate with an alert threshold and lists any obvious red flags in the input. The sidebar shows how well the model performed on transactions it had never seen, so the score is not a black box.

## Results

The model is trained on the oldest 60% of transactions, selected and tuned on the next 20%, and scored **once** on the most recent 20%. That mirrors deployment, where the model always faces future transactions.

| Split | Period | Rows | Fraud |
|---|---|---|---|
| Train | Oct 2022 → Jul 2024 | 6,000 | 127 |
| Validation | Jul 2024 → Mar 2025 | 2,000 | 37 |
| Test | Mar 2025 → Dec 2025 | 2,000 | 31 |

Selected model: **XGBoost** (highest validation PR-AUC). On the untouched test period:

| Metric | Value |
|---|---|
| PR-AUC | 0.073 (95% bootstrap CI 0.029–0.178) |
| Base fraud rate (a random model's PR-AUC) | 0.0155 |
| ROC-AUC | 0.73 |
| Fraud caught when reviewing the riskiest 5% | 23% |
| Precision / recall at the tuned threshold | 0.13 / 0.16 |

The model ranks fraud about **4.7× better than random**, but the signal in this dataset is weak, and with only 31 fraud cases in the test period the estimate is noisy (see the CI). The numbers are reported honestly rather than inflated by a random split. The previous version used one, which let the same customers appear in both train and test and tuned the threshold on the test set. See [Limitations](#limitations-and-next-steps).

## For technical reviewers

### How a prediction flows

```
raw transaction (JSON)
  -> FastAPI: Pydantic validation (allowed categories, value ranges)
  -> pickled sklearn Pipeline
       FraudFeatures   clean (typos, casing, sentinels) + engineer features
       ColumnTransformer   impute (+ missing indicators), scale, one-hot
       XGBoost         wrapped in CalibratedClassifierCV (sigmoid)
  -> probability, compared with the threshold tuned on validation data
  -> {fraud_probability, prediction, threshold, risk_level, model_version}
```

Cleaning and feature engineering live **inside** the pickled pipeline, so the API sends raw fields and gets exactly the features the model was trained on. There is no second, hand-copied implementation that could drift. A test asserts that scoring one row alone gives the same features as scoring it in a batch.

### Key design decisions

| Decision | Why |
|---|---|
| Chronological train / validation / test split | Fraud models always predict the future. A random split leaks customers (115 customers, one with 45% of all fraud) and fraud campaigns across splits. |
| Model selection and threshold on validation, never test | Picking the best of N models on the test set biases the reported score upward. |
| Class weights instead of SMOTE, then probability calibration | Using both double-corrects for imbalance and pushes probabilities far from real rates. Calibrated scores mean "6% ≈ 6% chance". |
| Data-learned cut-offs (e.g. "high internal risk") are fitted on train only | `FraudFeatures.fit` learns them, so nothing leaks from evaluation data. |
| Unknown or invalid categories become `NaN`, never silent zeros | Bad input is either rejected by the API (422) or explicitly imputed. |
| PR-AUC + recall@top-k + bootstrap CI | Accuracy is meaningless at a 2% base rate. Top-k recall answers "if analysts review N alerts, how much fraud do we catch?" |

### Data cleaning

The raw data is deliberately messy. `src/clean.py` handles:
- category typos and casing: `mobille`, `weeb`, `MOBILE`, `' mobile  '` → `mobile`; `standrd` → `standard`
- junk values: `unknown`, `NAN`, `' nan  '` → missing
- sentinel values: `fee = -1 / 9999.99`, `device_trust_score = -0.1` → missing; `txn_velocity_1h = -1` → 0
- malformed timestamps: `N/A`, `0000-00-00T00:00:00Z`, `2025/13/40 25:61:00` → missing
- 200 duplicated transaction rows, dropped before splitting

### Engineered features

Log amount, fee-to-amount ratio, hour of day, weekend and night flags, cross-border (home country ≠ IP country), new device with low trust, velocity ratio, velocity × amount, young-account flag and velocity, a combined risk count, and the currency corridor (`GBP->NGN`). Every feature is listed in `src/features.py`, and a test fails if any is computed but not passed to the model.

### Project layout

```
Fraud-detection-ML/
|-- main.py                    Train, select, calibrate, evaluate, log to MLflow
|-- run.py                     Local FastAPI dev runner
|-- Dockerfile                 API image (serving deps only, non-root, healthcheck)
|-- requirements-api.txt       Serving dependencies
|-- requirements.txt           Training + tracking + tests (includes the API deps)
|
|-- src/
|   |-- schema.py              Raw input columns and allowed category values
|   |-- clean.py               Stateless row cleaning (shared by train and serve)
|   |-- features.py            FraudFeatures sklearn transformer
|   |-- pipeline.py            Features -> preprocessing -> classifier
|   |-- models.py              Candidate models and imbalance handling
|   |-- data.py                Loading, de-duplication, chronological split
|   |-- evaluate.py            PR-AUC, top-k metrics, threshold, bootstrap CI
|
|-- api/app.py                 /predict, /health, /model-info, /dashboard
|-- streamlit_app/             Frontend (deployed separately)
|-- models/                    best_model.pkl + best_run_metrics.json
|-- tests/                     Cleaning, features, train/serve parity, API contract
|-- FRAUD_DETECTION.ipynb      Exploratory analysis
```

### API

| Endpoint | Description |
|---|---|
| `POST /predict` | Raw transaction → probability, decision, threshold, risk level, model version |
| `GET /health` | Liveness and loaded model version |
| `GET /model-info` | Full training report (JSON) |
| `GET /dashboard` | Training report as an HTML page |

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "amount_usd": 250, "fee": 3, "source_currency": "USD", "dest_currency": "NGN",
  "channel": "mobile", "home_country": "US", "ip_country": "US", "kyc_tier": "standard",
  "account_age_days": 400, "chargeback_history_count": 0, "new_device": false,
  "location_mismatch": false, "ip_risk_score": 0.2, "device_trust_score": 0.8,
  "risk_score_internal": 0.2, "corridor_risk": 0.0, "txn_velocity_1h": 0, "txn_velocity_24h": 1
}'
```

`ip_country`, `exchange_rate_src_to_dest` and `timestamp` are optional. Invalid categories (e.g. `"kyc_tier": "tier_1"`) or out-of-range scores return `422`.

### Frontend

The Streamlit app is a pure client of the API. It has presets ("Typical customer", "Suspicious pattern") for quick demos, dropdowns restricted to the categories the model knows, a probability gauge marked with the alert threshold, the risk relative to the base rate, plain-language red flags from the input, and a sidebar model card with test metrics and their confidence interval.

### CI/CD

GitHub Actions runs the test suite on every push and pull request. Pushes to `main` deploy to Hugging Face Spaces only if the tests pass.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py          # train, evaluate, write models/ and mlruns/
pytest                  # run the tests
python run.py           # API on http://127.0.0.1:8000
mlflow ui               # browse experiment runs on http://127.0.0.1:5000
```

Frontend, in a second terminal:

```bash
pip install -r streamlit_app/requirements.txt
echo "API_URL=http://127.0.0.1:8000" > .env
streamlit run streamlit_app/streamlit_app.py
```

## Tech stack

| Layer | Tool |
|---|---|
| Modeling | scikit-learn, XGBoost |
| Experiment tracking | MLflow (nested runs, params, signatures, artifacts) |
| Backend API | FastAPI + Pydantic |
| Frontend | Streamlit + Plotly |
| Testing | pytest |
| Containerization | Docker |
| Hosting | Hugging Face Spaces (API), Streamlit Community Cloud (app) |
| CI/CD | GitHub Actions |

## Limitations and next steps

- **Weak signal, small sample.** 195 fraud cases in total, and one customer accounts for 45% of them. More labelled fraud matters more than model tuning.
- **Behavioural features.** The strongest fraud signals are usually relative to the customer's own history (amount vs. their usual amount, time since last transaction, devices per customer). These need a feature store or history lookup at serving time, which the API does not have yet.
- **Tree models don't extrapolate.** Inputs far outside the training range (e.g. every risk field maxed out) can score lower than you'd expect.
- **Model registry.** Serve from the MLflow Model Registry instead of a committed `.pkl`.
- **Monitoring.** Log prediction inputs and scores and track drift against the training distribution.
- **Auth.** Add an API key on `/predict`.
