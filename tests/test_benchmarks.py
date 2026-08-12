"""Tests for the benchmark forecast functions."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from appliance_energy.models import benchmarks as bm

IDX = pd.date_range("2016-01-01", periods=400, freq="h")
Y = pd.Series(np.arange(400, dtype=float), index=IDX)
FUT = pd.date_range(IDX[-1] + pd.Timedelta("1h"), periods=24, freq="h")


def test_forecast_lengths_match_horizon():
    for name, fn in bm.BENCHMARKS.items():
        fc = fn(Y, 24, FUT)
        assert len(fc) == 24, f"{name} returned wrong length"


def test_naive_repeats_last_value():
    fc = bm.naive_forecast(Y, 24, FUT)
    assert (fc == Y.iloc[-1]).all()


def test_seasonal_naive_copies_one_day_back():
    fc = bm.seasonal_naive_forecast(Y, 24, FUT, seasonality=24)
    expected = Y.iloc[-24:].to_numpy()
    assert np.allclose(fc.to_numpy(), expected)


def test_drift_is_linear():
    fc = bm.drift_forecast(Y, 24, FUT)
    diffs = np.diff(fc.to_numpy())
    assert np.allclose(diffs, diffs[0])
