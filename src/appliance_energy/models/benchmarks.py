"""Benchmark forecasting methods (assignment Part 3).

Mean, naive, daily/weekly seasonal naive, and drift - the reference
models every complex model must beat to justify its complexity.

All functions share the signature
    f(y_train: pd.Series, horizon: int, index) -> pd.Series
so the pipeline can loop over them.
"""

import numpy as np
import pandas as pd

from ..config import DAILY_PERIOD, WEEKLY_PERIOD


def mean_forecast(y_train, horizon, index):
    """Forecast = mean of the whole training series (flat line)."""
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train, horizon, index):
    """Forecast = last observed value repeated (random-walk forecast)."""
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train, horizon, index, seasonality):
    """Forecast = value one season ago (m=24: same hour yesterday;
    m=168: same hour last week), applied recursively beyond one season."""
    history = list(y_train.values)
    values = []
    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])
    return pd.Series(values, index=index)


def drift_forecast(y_train, horizon, index):
    """Last value plus the average historical slope - a straight line
    from first to last observation, extrapolated."""
    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)
    values = [y_train.iloc[-1] + slope * step
              for step in range(1, horizon + 1)]
    return pd.Series(values, index=index, name="drift")


# Registry the pipeline loops over
BENCHMARKS = {
    "mean": mean_forecast,
    "naive": naive_forecast,
    "seasonal_naive_daily":
        lambda tr, h, ix: seasonal_naive_forecast(tr, h, ix, DAILY_PERIOD),
    "seasonal_naive_weekly":
        lambda tr, h, ix: seasonal_naive_forecast(tr, h, ix, WEEKLY_PERIOD),
    "drift": drift_forecast,
}


def rolling_origin_forecast(y: pd.Series, test_index, model_fn,
                            horizon: int = 24) -> pd.Series:
    """Rolling-origin evaluation across the 14-day test period.

    The test period is split into consecutive 24 h windows; for each
    window the model sees all data strictly before the window start
    (the forecast origin) and predicts the next 24 h. This mimics a
    forecast issued once per day, so every prediction is a true
    <=24 h-ahead forecast.
    """
    preds = []
    for start in range(0, len(test_index), horizon):
        window = test_index[start:start + horizon]
        history = y.loc[:window[0]].iloc[:-1]
        preds.append(model_fn(history, len(window), window))
    return pd.concat(preds)
