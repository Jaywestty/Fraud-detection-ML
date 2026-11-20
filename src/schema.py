"""Single source of truth for raw input columns and their valid values.

Training, the API schema, and the Streamlit form all derive from these
constants, so a category can't exist in one place and not another.
"""

TARGET = "is_fraud"

# Valid (canonical) values for categorical inputs, after cleaning.
COUNTRIES = ["US", "UK", "CA"]
SOURCE_CURRENCIES = ["USD", "GBP", "CAD"]
DEST_CURRENCIES = ["USD", "CAD", "GBP", "EUR", "MXN", "NGN", "INR", "PHP", "CNY"]
CHANNELS = ["mobile", "web", "atm"]
KYC_TIERS = ["low", "standard", "enhanced"]

# Raw fields the model consumes. Anything else in the CSV (ids, ip_address,
# device_id, amount_src) is ignored by the pipeline.
RAW_FEATURES = [
    "timestamp",
    "home_country",
    "ip_country",
    "source_currency",
    "dest_currency",
    "channel",
    "kyc_tier",
    "amount_usd",
    "fee",
    "exchange_rate_src_to_dest",
    "new_device",
    "location_mismatch",
    "ip_risk_score",
    "device_trust_score",
    "account_age_days",
    "chargeback_history_count",
    "risk_score_internal",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk",
]
