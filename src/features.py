import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["transaction_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["cross_border"] = (df["home_country"] != df["ip_country"]).astype(int)
    df["device_risk"] = (
        (df["new_device"] == True) & (df["device_trust_score"] < 0.5)
    ).astype(int)

    df["velocity_ratio"] = df["txn_velocity_1h"] / (df["txn_velocity_24h"] + 1)
    df["velocity_x_amount"] = df["velocity_ratio"] * df["amount_usd"].fillna(0)

    df["new_account_flag"] = (df["account_age_days"] < 60).astype(int)
    df["young_account_velocity"] = df["new_account_flag"] * df["txn_velocity_1h"]

    df["risk_pile_up"] = (
        (df["risk_score_internal"] > df["risk_score_internal"].quantile(0.75)).astype(int)
        + (df["txn_velocity_1h"] >= 2).astype(int)
        + df["location_mismatch"].astype(int)
        + df["new_device"].astype(int)
        + df["device_risk"].astype(int)
    )

    drop_cols = ["transaction_id", "customer_id", "timestamp", "ip_address", "day_of_week"]
    return df.drop(columns=drop_cols, errors="ignore")