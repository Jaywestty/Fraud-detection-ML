"""Feature engineering as a scikit-learn transformer.

Living inside the pickled pipeline means the API sends raw fields and gets the
exact same features the model was trained on - no hand-copied logic to drift.
Any statistic (e.g. the "high internal risk" cut-off) is learned in `fit` on
the training split only, so nothing leaks from validation/test data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.clean import clean
from src.schema import RAW_FEATURES

NUM_FEATURES = [
    "amount_usd", "log_amount_usd", "fee", "fee_ratio", "exchange_rate_src_to_dest",
    "ip_risk_score", "device_trust_score", "account_age_days",
    "chargeback_history_count", "risk_score_internal", "txn_velocity_1h",
    "txn_velocity_24h", "corridor_risk", "velocity_ratio", "velocity_x_amount",
    "young_account_velocity", "risk_pile_up", "transaction_hour",
]

BINARY_FEATURES = [
    "new_device", "location_mismatch", "cross_border", "device_risk",
    "new_account_flag", "is_weekend", "is_night_txn",
]

CAT_FEATURES = [
    "home_country", "source_currency", "dest_currency", "corridor", "channel", "kyc_tier",
]

OUTPUT_FEATURES = NUM_FEATURES + BINARY_FEATURES + CAT_FEATURES


class FraudFeatures(BaseEstimator, TransformerMixin):
    """Clean raw transaction rows and derive model features."""

    def __init__(self, new_account_days: int = 60, high_risk_quantile: float = 0.75):
        self.new_account_days = new_account_days
        self.high_risk_quantile = high_risk_quantile

    def fit(self, X: pd.DataFrame, y=None):
        X = clean(self._ensure_columns(X))
        self.high_risk_cutoff_ = float(X["risk_score_internal"].quantile(self.high_risk_quantile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = clean(self._ensure_columns(X))
        out = pd.DataFrame(index=df.index)

        for col in ("amount_usd", "fee", "exchange_rate_src_to_dest", "ip_risk_score",
                    "device_trust_score", "account_age_days", "chargeback_history_count",
                    "risk_score_internal", "txn_velocity_1h", "txn_velocity_24h",
                    "corridor_risk", "new_device", "location_mismatch"):
            out[col] = df[col]

        out["log_amount_usd"] = np.log1p(df["amount_usd"])
        out["fee_ratio"] = df["fee"] / df["amount_usd"]

        ts = df["timestamp"]
        out["transaction_hour"] = ts.dt.hour.astype(float)
        out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(float).where(ts.notna())
        out["is_night_txn"] = ts.dt.hour.isin([0, 1, 2, 3, 4]).astype(float).where(ts.notna())

        both_known = df["home_country"].notna() & df["ip_country"].notna()
        out["cross_border"] = (df["home_country"] != df["ip_country"]).astype(float).where(both_known)
        out["device_risk"] = (
            (df["new_device"] == 1) & (df["device_trust_score"] < 0.5)
        ).astype(float).where(df["new_device"].notna() & df["device_trust_score"].notna())

        out["velocity_ratio"] = df["txn_velocity_1h"] / (df["txn_velocity_24h"] + 1)
        out["velocity_x_amount"] = out["velocity_ratio"] * df["amount_usd"]
        out["new_account_flag"] = (df["account_age_days"] < self.new_account_days).astype(float)
        out["young_account_velocity"] = out["new_account_flag"] * df["txn_velocity_1h"]
        out["risk_pile_up"] = (
            (df["risk_score_internal"] > self.high_risk_cutoff_).astype(float)
            + (df["txn_velocity_1h"] >= 2).astype(float)
            + df["location_mismatch"].fillna(0)
            + df["new_device"].fillna(0)
            + out["device_risk"].fillna(0)
        )

        for col in ("home_country", "source_currency", "dest_currency", "channel", "kyc_tier"):
            out[col] = df[col]
        known = df["source_currency"].notna() & df["dest_currency"].notna()
        corridor = df["source_currency"].fillna("") + "->" + df["dest_currency"].fillna("")
        out["corridor"] = corridor.where(known, np.nan).astype(object)

        numeric = NUM_FEATURES + BINARY_FEATURES
        out[numeric] = out[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
        return out[OUTPUT_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(OUTPUT_FEATURES, dtype=object)

    @staticmethod
    def _ensure_columns(X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = np.nan
        return X
