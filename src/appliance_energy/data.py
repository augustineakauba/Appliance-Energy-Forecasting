"""Load and prepare the UCI Appliances Energy Prediction dataset.

The raw data are sampled every 10 minutes (11 Jan - 27 May 2016,
a low-energy house in Stambruges, Belgium; Candanedo et al., 2017).
We parse the timestamp, run data-quality checks, and resample to
hourly resolution for the forecasting task.
"""

import pandas as pd

from .config import HOURLY_CSV, RAW_CSV, TARGET


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw 10-minute CSV and set a sorted DatetimeIndex."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # be safe: force every column numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Missing values, duplicate timestamps and gaps in the 10-min grid.

    Returned as a dict so the exact numbers can be quoted in the report.
    """
    deltas = df.index.to_series().diff().dropna()
    return {
        "n_rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "n_missing_values": int(df.isna().sum().sum()),
        "n_duplicate_timestamps": int(df.index.duplicated().sum()),
        "n_gaps_in_10min_grid": int((deltas != pd.Timedelta("10min")).sum()),
    }


def resample_hourly(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Aggregate the 10-minute data to hourly means.

    Averaging (rather than summing) keeps the target on the same scale
    as the raw data and matches the course demo pipeline. Because every
    hour contains exactly six 10-minute samples, hourly mean and hourly
    sum differ only by a constant factor of 6, so model rankings are
    unaffected by this choice.
    """
    hourly = df.resample("h").mean()
    # only keep complete hours (6 samples), then interpolate tiny gaps
    counts = df[TARGET].resample("h").count()
    hourly = hourly[counts > 0].interpolate("time").dropna()
    if save:
        hourly.to_csv(HOURLY_CSV)
    return hourly


def train_test_split(series_or_df, test_steps: int):
    """Chronological split: the last `test_steps` rows form the test set.

    A time-ordered split (never random!) prevents look-ahead leakage.
    """
    return series_or_df.iloc[:-test_steps], series_or_df.iloc[-test_steps:]
