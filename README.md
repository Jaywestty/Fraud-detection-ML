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

An end-to-end fraud detection system that scores financial transactions in real time, built to demonstrate a production-style MLOps workflow: experiment tracking, model serving, and a user-facing interface, deployed as three connected services.

Live prediction interface (Streamlit): https://fraud-detect-mll.streamlit.app
Live API and model dashboard (FastAPI on Hugging Face Spaces): https://jayywestty-frauddetection-ml.hf.space
Model performance dashboard: https://jayywestty-frauddetection-ml.hf.space/dashboard

## What this project demonstrates

This project is a walkthrough of taking a fraud detection model from a notebook to a deployed, monitorable system. It's structured the way a data engineering or MLOps team would organize it, not as a single script.

- Experiment tracking with MLflow, every training run logs metrics for multiple candidate models and automatically selects the best performer
- A clean separation between data processing, feature engineering, model training, and evaluation (`src/`)
- A FastAPI backend that serves predictions and exposes model performance through a live dashboard
- A Streamlit frontend, deployed independently, that consumes the FastAPI backend as its data source
- CI/CD via GitHub Actions, auto-deploying to Hugging Face Spaces on every push to `main`
- Class imbalance handled properly with SMOTE, since fraud is a rare-event problem by nature

## For non-technical readers

This is a fraud checker. You enter details about a transaction (amount, device used, account age, and so on), and the system tells you how likely it is to be fraudulent, along with a probability score. Behind the scenes, a machine learning model that was trained and tested on historical transaction data makes that call, and a small dashboard shows how well that model actually performs, so the prediction isn't a black box.

## For technical reviewers

The system is composed of three deployable pieces working together, not one monolithic app:

```
Training (main.py) -> MLflow tracking (mlruns/) -> best_run_metrics.json
                                                          |
                                                          v
                                        FastAPI backend (api/app.py)
                                        - POST /predict     serves the trained model
                                        - GET /model-info   JSON metrics
                                        - GET /dashboard    HTML metrics dashboard
                                                          |
                                                          v
                                        Streamlit frontend (streamlit_app/)
                                        calls the deployed FastAPI backend
```

### Architecture

```
Fraud-detection-ML/
|
|-- main.py                    Training entrypoint, MLflow logging, model selection
|-- run.py                     Local FastAPI dev runner
|-- Dockerfile                 Container spec for the FastAPI backend
|-- requirements.txt           Backend dependencies
|
|-- src/
|   |-- data.py                 Data loading
|   |-- features.py             Feature engineering
|   |-- pipeline.py             Preprocessing, SMOTE, model pipeline
|   |-- models.py               Candidate model definitions
|   |-- train.py                Model fitting
|   |-- evaluate.py             PR-AUC and threshold evaluation
|
|-- api/
|   |-- app.py                  FastAPI app: /predict, /model-info, /dashboard
|
|-- models/
|   |-- best_model.pkl           Production model, loaded by the API
|   |-- best_run_metrics.json    Latest run's metrics, powers the dashboard
|
|-- mlruns/                     Local MLflow tracking store
|
|-- streamlit_app/
|   |-- streamlit_app.py         Frontend, calls the deployed FastAPI backend
|   |-- requirements.txt         Frontend-only dependencies
|
|-- .github/workflows/           CI/CD to Hugging Face Spaces
```

### Model training and tracking

`main.py` trains three candidate models, LogisticRegression, RandomForest, and XGBoost, inside a single MLflow run. Each model's PR-AUC is logged to MLflow, the best-performing model is saved as the production artifact, and a `best_run_metrics.json` file is written alongside it. That file is what the `/model-info` and `/dashboard` endpoints read from, so the dashboard always reflects the most recent training run without any manual step.

PR-AUC (precision-recall area under curve) is used instead of accuracy, since fraud is a heavily imbalanced classification problem, accuracy would be misleading on its own.

### Serving

`api/app.py` loads `models/best_model.pkl` at startup and exposes:
- `POST /predict`, accepts transaction features, returns a fraud probability and binary prediction
- `GET /model-info`, returns the latest training run's metrics as JSON
- `GET /dashboard`, renders those same metrics as an HTML page

### Frontend

`streamlit_app/streamlit_app.py` is a separate app, deployed independently on Streamlit Community Cloud, that calls the deployed FastAPI backend for both predictions and dashboard metrics. It has no model logic of its own, it's a pure client of the API.

### Deployment

- The FastAPI backend is containerized with Docker and deployed to Hugging Face Spaces
- A GitHub Actions workflow (`.github/workflows/`) syncs the repo to the Space automatically on every push to `main`
- The Streamlit frontend is deployed separately on Streamlit Community Cloud, pointed at the live backend URL via an `API_URL` environment variable

## Running locally

Backend

```bash
pip install -r requirements.txt
python main.py
python run.py
```

`main.py` trains the models, logs to MLflow, and saves `best_model.pkl`. `run.py` starts the FastAPI server locally.

Frontend

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Set `API_URL` in a `.env` file at the project root before running the frontend, pointing to your local or deployed backend's `/predict` endpoint.

## Tech stack

| Layer | Tool |
|---|---|
| Experiment tracking | MLflow |
| Modeling | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Backend hosting | Hugging Face Spaces |
| Frontend hosting | Streamlit Community Cloud |
| CI/CD | GitHub Actions |

## Roadmap

- [ ] Add automated model retraining trigger via CI/CD
- [ ] Add authentication on the `/predict` endpoint
- [ ] Track data drift alongside model metrics on the dashboard