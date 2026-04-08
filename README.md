# 🔍 Fraud Detection ML App

A machine learning-powered web application that predicts whether a financial transaction is fraudulent — built with Scikit-learn and deployed via Streamlit.

🚀 **[Live Demo → fraud-detect-ml.streamlit.app](https://fraud-detect-ml.streamlit.app/)**

> ⚠️ **Note:** The app is hosted on Streamlit Community Cloud and may be sleeping due to inactivity. If you see a "This app is sleeping" message, just click **"Wake app"** and it will be back up in a few seconds.

---

## 📌 Overview

Financial fraud is a critical challenge in banking and online payments. Fraudulent transactions are rare compared to legitimate ones, creating a highly imbalanced classification problem.

This project covers the full ML pipeline — from data exploration and preprocessing to model training, evaluation, and deployment — wrapped in an interactive Streamlit interface where users can input transaction details and receive real-time fraud predictions.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data analysis and preprocessing |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training and evaluation |
| `imbalanced-learn` | Handling class imbalance (SMOTE) |
| `joblib` | Model serialization |
| `streamlit` | Interactive web application |

---

## 📊 ML Pipeline

### 1. Data Loading
- Import libraries and load dataset with `pandas`
- Inspect shape, structure, and column types

### 2. Exploratory Data Analysis (EDA)
- Analyze class distribution (fraud vs. non-fraud)
- Statistical summaries and feature visualizations
- **Key finding:** Fraudulent transactions make up a very small fraction of all records — confirming the need for imbalance-handling techniques

### 3. Data Preprocessing
- Handle missing values
- Feature selection
- Split into features (`X`) and target (`y`)
- Train/test split for generalization

### 4. Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset by generating synthetic fraud samples — preventing the model from defaulting to predicting the majority class.

### 5. Model Training
A classification model is trained on the balanced dataset and evaluated using:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- **Note:** Recall is prioritized — missing a fraudulent transaction is more costly than a false alarm

### 6. Model Serialization
```python
import joblib
joblib.dump(model, "fraud_model.pkl")
```
The trained model is saved and loaded directly into the Streamlit app — no retraining required.

---

## 🌐 Streamlit App

The app provides an interactive UI where users can:
1. Input transaction features
2. Click **Predict**
3. Receive an instant fraud risk assessment

The model is loaded efficiently using Streamlit's caching:
```python
import joblib
model = joblib.load("fraud_model.pkl")
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Jaywestty/Fraud-detection-ML.git
cd fraud-detection-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
fraud-detection-ml/
│
├── app.py                # Streamlit web application
├── fraud_model.pkl       # Trained ML model
├── notebook.ipynb        # Model development notebook
├── requirements.txt      # Project dependencies
├── runtime.txt           # Python version for deployment
└── README.md             # Project documentation
```

---

## 🎯 Key Features

- ✅ End-to-end ML pipeline from raw data to deployed app
- ✅ Handles highly imbalanced datasets using SMOTE
- ✅ Optimized for fraud recall — catches more fraud cases
- ✅ Interactive real-time prediction interface
- ✅ Lightweight and easily deployable

---

## 📈 Roadmap

- [ ] Model performance dashboard
- [ ] Probability-based fraud risk scoring
- [ ] Transaction visualization and analytics
- [ ] Docker / cloud deployment support
