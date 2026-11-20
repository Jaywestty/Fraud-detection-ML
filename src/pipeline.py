from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import BINARY_FEATURES, CAT_FEATURES, NUM_FEATURES, FraudFeatures


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    # remainder="drop" is explicit on purpose: every column the model uses must
    # be listed in src/features.py, and tests assert the lists are complete.
    return ColumnTransformer(
        [
            ("num", num_pipe, NUM_FEATURES),
            ("bin", bin_pipe, BINARY_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """Raw transaction rows in -> fraud probability out."""
    return Pipeline([
        ("features", FraudFeatures()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
