import pandas as pd

from src.schema import TARGET


def load_data(path) -> pd.DataFrame:
    """Load the raw CSV, drop exact duplicates and rows without a label."""
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="transaction_id", keep="first")
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


def time_split(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2):
    """Chronological train / validation / test split.

    A fraud model is always deployed on transactions newer than the ones it
    was trained on, so a random split flatters it (the same customers and
    fraud campaigns end up on both sides). Rows with an unparseable timestamp
    are kept in the training split, never in evaluation.
    """
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    order = ts.sort_values(na_position="first").index
    df = df.loc[order].reset_index(drop=True)

    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]
