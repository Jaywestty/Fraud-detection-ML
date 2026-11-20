import numpy as np
import pandas as pd

from src.clean import clean
from src.schema import CHANNELS, COUNTRIES, KYC_TIERS


def test_categorical_typos_case_and_whitespace_are_normalised():
    df = pd.DataFrame({
        "channel": ["mobille", " mobile  ", "MOBILE", "weeb", "ATm", "unknown"],
        "kyc_tier": ["standrd", "STANDARD", " standard  ", "low", "enhanced", np.nan],
    })
    out = clean(df)
    assert out["channel"].tolist()[:5] == ["mobile", "mobile", "mobile", "web", "atm"]
    assert pd.isna(out["channel"].iloc[5])
    assert out["kyc_tier"].tolist()[:5] == ["standard"] * 3 + ["low", "enhanced"]


def test_country_junk_becomes_missing():
    out = clean(pd.DataFrame({"ip_country": [" US  ", "NAN", " nan  ", "unknown", "UK"]}))
    assert out["ip_country"].iloc[0] == "US"
    assert out["ip_country"].iloc[1:4].isna().all()


def test_sentinels_become_missing():
    out = clean(pd.DataFrame({
        "fee": [-1, 9999.99, 3.5],
        "device_trust_score": [-0.1, 0.5, 0.9],
        "txn_velocity_1h": [-1, 0, 3],
    }))
    assert out["fee"].isna().tolist() == [True, True, False]
    assert np.isnan(out["device_trust_score"].iloc[0])
    assert out["txn_velocity_1h"].tolist() == [0, 0, 3]


def test_bad_timestamps_parse_to_nat():
    out = clean(pd.DataFrame({"timestamp": [
        "2024-01-01 10:00:00+00:00", "N/A", "0000-00-00T00:00:00Z", "2025/13/40 25:61:00", None,
    ]}))
    assert out["timestamp"].notna().tolist() == [True, False, False, False, False]


def test_cleaning_real_data_leaves_only_valid_categories(raw_df):
    out = clean(raw_df)
    assert set(out["channel"].dropna()) <= set(CHANNELS)
    assert set(out["kyc_tier"].dropna()) <= set(KYC_TIERS)
    assert set(out["home_country"].dropna()) <= set(COUNTRIES)
    assert (out["fee"].dropna() < 9999).all()
