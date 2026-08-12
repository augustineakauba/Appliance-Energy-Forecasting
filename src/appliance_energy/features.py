"""Covariate engineering (assignment Part 5).

Feature groups
--------------
time    : hour (sin/cos), day-of-week (sin/cos), weekend flag
lags    : Appliances 1, 2, 3, 6, 12, 24, 48 and 168 hours ago
rolling : mean/std/max of Appliances over past 3, 24 and 168 h
sensors : indoor temperatures T1-T9 and humidities RH_1-RH_9, lights
weather : T_out, RH_out, pressure, windspeed, visibility, dew point

Leakage prevention
------------------
Every lag/rolling feature is built with .shift() so the feature at
hour t uses information strictly BEFORE t. Sensor/weather values AT
time t would not be known 24 h in advance in a real deployment -
using their realised test-set values makes the forecast CONDITIONAL
(discussed in report Q5); the recursive forecaster therefore also
supports a "past-only" feature set.
"""

import numpy as np
import pandas as pd

from .config import TARGET

SENSOR_COLS = ([f"T{i}" for i in range(1, 10)]
               + [f"RH_{i}" for i in range(1, 10)] + ["lights"])
WEATHER_COLS = ["T_out", "RH_out", "Press_mm_hg", "Windspeed",
                "Visibility", "Tdewpoint"]
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLL_WINDOWS = [3, 24, 168]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features; hour and day-of-week are sin/cos encoded so
    that 23:00 and 00:00 (or Sun and Mon) are numerically adjacent."""
    out = df.copy()
    hour = out.index.hour
    dow = out.index.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Lagged target: short lags capture persistence, lag 24 the daily
    cycle, lag 168 the weekly cycle."""
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame,
                         target: str = TARGET) -> pd.DataFrame:
    """Rolling statistics of the PAST target only: shift(1) guarantees
    the window ends before the current hour (no leakage)."""
    out = df.copy()
    past = out[target].shift(1)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = past.rolling(w).mean()
        out[f"roll_std_{w}"] = past.rolling(w).std()
    out["roll_max_24"] = past.rolling(24).max()
    return out


def build_feature_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    """Assemble the supervised-learning table and drop warm-up rows
    (first 168 h, where the weekly lag is undefined)."""
    df = hourly[[TARGET] + SENSOR_COLS + WEATHER_COLS].copy()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df.dropna()


def feature_columns(df: pd.DataFrame, include_covariates: bool = True):
    """Feature list for the ML model.

    include_covariates=True  -> everything (conditional forecast).
    include_covariates=False -> only past-target + time features
                                (a true operational forecast).
    """
    lag_roll = [c for c in df.columns
                if c.startswith("lag_") or c.startswith("roll_")]
    cols = TIME_COLS + lag_roll
    if include_covariates:
        cols += SENSOR_COLS + WEATHER_COLS
    return cols


def feature_group(col: str) -> str:
    """Map a feature name to its group (for grouped importances, Q3)."""
    if col.startswith("lag_"):
        return "lags"
    if col.startswith("roll_"):
        return "rolling"
    if col in SENSOR_COLS:
        return "indoor sensors"
    if col in WEATHER_COLS:
        return "weather"
    if col in TIME_COLS:
        return "time"
    return "other"
