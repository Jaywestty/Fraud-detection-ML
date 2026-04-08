from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

NUM_COLS = [
    "amount_usd","fee","exchange_rate_src_to_dest","ip_risk_score",
    "account_age_days","device_trust_score","chargeback_history_count",
    "risk_score_internal","txn_velocity_1h","txn_velocity_24h","corridor_risk"
]

CAT_COLS = [
    "home_country","source_currency","dest_currency","channel",
    "kyc_tier","new_device","location_mismatch","cross_border","device_risk"
]

def build_pipeline(X_train, classifier):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, [c for c in NUM_COLS if c in X_train.columns]),
        ("cat", cat_pipe, [c for c in CAT_COLS if c in X_train.columns])
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.3)),
        ("classifier", classifier)
    ])