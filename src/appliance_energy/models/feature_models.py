"""Feature-based ML model (assignment Part 6): XGBoost.

Includes a genuinely recursive 24 h forecaster: at prediction time,
lag/rolling features inside the 24 h window are rebuilt from the
model's OWN previous predictions, never from unseen actuals - so the
forecast really is 24 h ahead, unlike naive row-wise prediction which
would silently use lag_1 of the true test values (1 h ahead only).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import RANDOM_STATE, TARGET
from ..features import LAGS, ROLL_WINDOWS


def fit_xgb(X_train, y_train):
    """Gradient-boosted trees. Hyperparameters chosen small/regularised
    to limit overfitting on ~3000 rows; tuned via the last-14-days-of-
    train validation split in tune_xgb()."""
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", random_state=RANDOM_STATE,
        n_jobs=4)
    model.fit(X_train, y_train, verbose=False)
    return model


def tune_xgb(X_tr, y_tr, X_val, y_val, param_grid=None) -> pd.DataFrame:
    """Small validation-based hyperparameter search (hypertuning).

    Uses the final 14 days of the TRAINING period as validation, so
    the test set is never touched during model selection (avoids the
    'choosing the final model on test performance' leakage trap)."""
    if param_grid is None:
        param_grid = [
            {"max_depth": d, "learning_rate": lr, "n_estimators": n}
            for d in (3, 5, 7) for lr in (0.03, 0.1) for n in (300, 600)
        ]
    rows = []
    for params in param_grid:
        model = XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, random_state=RANDOM_STATE,
                             n_jobs=4, objective="reg:squarederror",
                             **params)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((pred - np.asarray(y_val)) ** 2)))
        rows.append({**params, "val_RMSE": round(rmse, 2)})
    return pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)


def _rebuild_lag_roll(history: pd.Series, ts) -> dict:
    """Recompute lag/rolling features for timestamp ts from `history`
    (actuals up to the forecast origin + own predictions after it)."""
    feats = {}
    vals = history.values
    for lag in LAGS:
        feats[f"lag_{lag}"] = vals[-lag]
    past = vals  # already ends at ts - 1h
    for w in ROLL_WINDOWS:
        feats[f"roll_mean_{w}"] = past[-w:].mean()
        feats[f"roll_std_{w}"] = past[-w:].std(ddof=1)
    feats["roll_max_24"] = past[-24:].max()
    return feats


def recursive_forecast(model, feature_df: pd.DataFrame, feature_cols,
                       y: pd.Series, test_index,
                       horizon: int = 24) -> pd.Series:
    """Rolling-origin recursive 24 h forecasts over the test period.

    For each day: lags/rollings are seeded with actual data up to the
    origin, then recursively filled with the model's own predictions
    within the 24 h window. Exogenous (sensor/weather/time) features
    are taken from feature_df at their realised values -> conditional
    forecast for the sensor/weather part (see report Q5).
    """
    preds = []
    for start in range(0, len(test_index), horizon):
        window = test_index[start:start + horizon]
        history = y.loc[:window[0]].iloc[:-1].copy()
        for ts in window:
            row = feature_df.loc[ts, feature_cols].to_dict()
            row.update({k: v for k, v in _rebuild_lag_roll(history, ts).items()
                        if k in feature_cols})
            X = pd.DataFrame([row], columns=feature_cols).astype(float)
            yhat = float(model.predict(X)[0])
            yhat = max(yhat, 0.0)          # energy cannot be negative
            preds.append((ts, yhat))
            history.loc[ts] = yhat          # feed prediction back in
    idx, vals = zip(*preds)
    return pd.Series(vals, index=list(idx), name="feature_model")


def grouped_importance(model, feature_cols, group_fn) -> pd.Series:
    """Sum XGBoost gain-importances within each feature group (Q3)."""
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    return imp.groupby(imp.index.map(group_fn)).sum().sort_values()
