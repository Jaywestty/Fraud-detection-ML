import pandas as pd

from src.data import time_split
from src.features import OUTPUT_FEATURES, FraudFeatures
from src.pipeline import build_preprocessor
from src.schema import RAW_FEATURES, TARGET


def _ts(d):
    return pd.to_datetime(d["timestamp"], errors="coerce", utc=True, format="mixed")


def test_every_engineered_feature_reaches_the_model():
    """Guards against features being computed and then silently dropped."""
    pre = build_preprocessor()
    used = [c for _, _, cols in pre.transformers for c in cols]
    assert sorted(used) == sorted(OUTPUT_FEATURES)


def test_transform_output_columns_and_no_infinities(raw_df):
    out = FraudFeatures().fit(raw_df[RAW_FEATURES]).transform(raw_df[RAW_FEATURES])
    assert list(out.columns) == OUTPUT_FEATURES
    num = out.select_dtypes("number")
    assert not num.isin([float("inf"), float("-inf")]).any().any()


def test_cross_border_uses_ip_country_not_a_hardcoded_country(raw_df):
    rows = pd.DataFrame([
        {"home_country": "UK", "ip_country": "UK"},
        {"home_country": "UK", "ip_country": "US"},
        {"home_country": "UK", "ip_country": None},
    ])
    out = FraudFeatures().fit(raw_df[RAW_FEATURES]).transform(rows)
    assert out["cross_border"].iloc[0] == 0
    assert out["cross_border"].iloc[1] == 1
    assert pd.isna(out["cross_border"].iloc[2])


def test_risk_cutoff_learned_from_training_data_only(raw_df):
    train, _, _ = time_split(raw_df)
    ff = FraudFeatures().fit(train[RAW_FEATURES])
    expected = pd.to_numeric(train["risk_score_internal"]).clip(0, 1).quantile(0.75)
    assert abs(ff.high_risk_cutoff_ - expected) < 1e-9


def test_single_row_matches_batch(raw_df):
    """Train/serve parity: scoring one row alone equals scoring it in a batch."""
    ff = FraudFeatures().fit(raw_df[RAW_FEATURES])
    batch = ff.transform(raw_df[RAW_FEATURES].head(50))
    single = ff.transform(raw_df[RAW_FEATURES].iloc[[17]])
    pd.testing.assert_frame_equal(batch.iloc[[17]], single, check_dtype=False)


def test_time_split_is_chronological_and_complete(raw_df):
    train, val, test = time_split(raw_df)
    assert len(train) + len(val) + len(test) == len(raw_df)
    assert _ts(train).max() <= _ts(val).min()
    assert _ts(val).max() <= _ts(test).min()
    assert _ts(val).notna().all() and _ts(test).notna().all()
    assert test[TARGET].sum() > 0


def test_load_data_drops_duplicate_transactions(raw_df):
    assert raw_df["transaction_id"].is_unique
