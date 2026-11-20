"""Row-level data cleaning shared by training and serving.

Everything here is stateless (no statistics learned from the data), so it is
safe to apply before the train/test split and identically at inference time.
"""

import numpy as np
import pandas as pd

from src.schema import CHANNELS, COUNTRIES, DEST_CURRENCIES, KYC_TIERS, SOURCE_CURRENCIES

# Known typos observed in the raw data -> canonical value (after strip + case fold).
_TYPO_FIXES = {
    "channel": {"mobille": "mobile", "weeb": "web"},
    "kyc_tier": {"standrd": "standard"},
}

# column -> (case function, allowed values). Anything outside the allowed set
# ("unknown", "N/A", "NAN", unseen junk) becomes NaN and is imputed later.
_CATEGORICAL = {
    "home_country": (str.upper, COUNTRIES),
    "ip_country": (str.upper, COUNTRIES),
    "source_currency": (str.upper, SOURCE_CURRENCIES),
    "dest_currency": (str.upper, DEST_CURRENCIES),
    "channel": (str.lower, CHANNELS),
    "kyc_tier": (str.lower, KYC_TIERS),
}

# Placeholder values that mean "missing", not a real measurement.
_SENTINELS = {
    "fee": [-1, 9999.99],
    "device_trust_score": [-0.1],
}

_NUMERIC = [
    "amount_usd", "fee", "exchange_rate_src_to_dest", "ip_risk_score",
    "device_trust_score", "account_age_days", "chargeback_history_count",
    "risk_score_internal", "txn_velocity_1h", "txn_velocity_24h", "corridor_risk",
]

_BOOLEAN = ["new_device", "location_mismatch"]


def _clean_categorical(s: pd.Series, case, allowed) -> pd.Series:
    def norm(v):
        if not isinstance(v, str):
            return np.nan
        v = case(v.strip())
        return v

    out = s.map(norm)
    col_fixes = _TYPO_FIXES.get(s.name, {})
    out = out.replace(col_fixes)
    return out.where(out.isin(allowed), np.nan).astype(object)


def _clean_boolean(s: pd.Series) -> pd.Series:
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    return s.map(
        lambda v: float(v) if isinstance(v, (bool, np.bool_, int, float)) and not pd.isna(v)
        else mapping.get(str(v).strip().lower(), np.nan)
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, (case, allowed) in _CATEGORICAL.items():
        if col in df:
            df[col] = _clean_categorical(df[col], case, allowed)

    for col in _NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col, bad in _SENTINELS.items():
        if col in df:
            df[col] = df[col].replace(bad, np.nan)

    if "txn_velocity_1h" in df:
        # -1 is a logging artefact for "no transactions" (matches the notebook).
        df["txn_velocity_1h"] = df["txn_velocity_1h"].clip(lower=0)
    for col in ("ip_risk_score", "device_trust_score", "risk_score_internal", "corridor_risk"):
        if col in df:
            df[col] = df[col].clip(0, 1)

    for col in _BOOLEAN:
        if col in df:
            df[col] = _clean_boolean(df[col])

    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")

    return df
