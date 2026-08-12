"""Common accuracy metrics (assignment Parts 2 and 8).

Every model is scored with exactly the same functions on exactly the
same test period, so the comparison table is fair.
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    """Mean Absolute Error - average error size, in Wh."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error - penalises large errors (spikes) more."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%). Safe here because hourly
    appliance use never reaches 0 Wh in this dataset."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase(y_true, y_pred, y_train, m: int = 24):
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales the MAE by the in-sample MAE of the daily seasonal naive
    forecast. MASE < 1 -> the model beats the daily seasonal naive
    at that scale; MASE = 1 -> no better than it.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def bias(y_true, y_pred):
    """Mean error (pred - actual). Positive -> systematic over-forecast."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(y_pred - y_true))


def evaluate_forecast(name, y_true, y_pred, y_train, m: int = 24) -> dict:
    """All five metrics for one model, as a tidy row for the table."""
    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, m=m),
        "Bias": bias(y_true, y_pred),
    }


def metrics_table(rows) -> pd.DataFrame:
    """Collect rows into the model-comparison table, best RMSE first."""
    return (pd.DataFrame(rows).set_index("model")
            .sort_values("RMSE").round(2))
