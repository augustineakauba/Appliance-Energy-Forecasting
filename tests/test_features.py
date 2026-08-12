"""Tests that feature engineering does not leak future information."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from appliance_energy import features
from appliance_energy.config import TARGET


def _toy_frame(n=400):
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({TARGET: rng.uniform(30, 300, n)}, index=idx)
    for c in features.SENSOR_COLS + features.WEATHER_COLS:
        df[c] = rng.normal(size=n)
    return df


def test_lag_features_use_only_past():
    df = features.add_lag_features(_toy_frame())
    # lag_1 at time t must equal the target at t-1
    assert np.allclose(df["lag_1"].iloc[1:], df[TARGET].shift(1).iloc[1:],
                       equal_nan=True)


def test_rolling_features_exclude_current_hour():
    df = features.add_rolling_features(_toy_frame())
    t = df.index[100]
    window = df[TARGET].iloc[97:100]        # hours t-3..t-1 only
    assert np.isclose(df.loc[t, "roll_mean_3"], window.mean())


def test_feature_matrix_has_no_nans():
    assert not features.build_feature_matrix(_toy_frame()).isna().any().any()


def test_pastonly_columns_exclude_covariates():
    df = features.build_feature_matrix(_toy_frame())
    cols = features.feature_columns(df, include_covariates=False)
    assert not any(c in features.SENSOR_COLS + features.WEATHER_COLS
                   for c in cols)
