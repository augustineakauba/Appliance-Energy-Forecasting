"""ARIMA/SARIMA(X) modelling (assignment Part 4).

Implements the optimize_ARIMA AIC grid search from the tutorials,
seasonal extension, residual diagnostics and forecasting with
confidence intervals via get_forecast().
"""

import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..config import FIGURE_DIR

warnings.filterwarnings("ignore")


def optimize_ARIMA(endog, order_list, d: int,
                   seasonal_order=(0, 0, 0, 0),
                   csv_path=None) -> pd.DataFrame:
    """Loop over every (p, q) candidate at fixed d, fit, record AIC.

    Follows the optimize_ARIMA pattern from tutorial 3: models that
    fail to converge are skipped; the returned table is sorted so the
    lowest-AIC (best) model comes first. Optionally checkpoints the
    partial table to CSV after every fit (the full grid is slow).
    """
    results = []
    for p, q in order_list:
        try:
            fit = SARIMAX(endog, order=(p, d, q),
                          seasonal_order=seasonal_order
                          ).fit(disp=False, maxiter=100)
            results.append([(p, d, q), seasonal_order, round(fit.aic, 1)])
        except Exception:
            continue
        if csv_path is not None:
            (pd.DataFrame(results, columns=["(p,d,q)", "(P,D,Q,m)", "AIC"])
             .sort_values("AIC").to_csv(csv_path, index=False))
    return (pd.DataFrame(results, columns=["(p,d,q)", "(P,D,Q,m)", "AIC"])
            .sort_values("AIC").reset_index(drop=True))


def grid_search_arima(y_train, p_range, d_range, q_range,
                      csv_path=None) -> pd.DataFrame:
    """Stage 1 - full non-seasonal grid required by the brief:
    p = 0..6, d = 0..2, q = 0..6 (7 x 3 x 7 = 147 fits)."""
    frames = []
    for d in d_range:
        frames.append(optimize_ARIMA(y_train, list(product(p_range, q_range)),
                                     d, csv_path=None))
        if csv_path is not None:
            pd.concat(frames).sort_values("AIC").to_csv(csv_path, index=False)
    return pd.concat(frames).sort_values("AIC").reset_index(drop=True)


def grid_search_seasonal(y_train, best_order, m: int = 24,
                         csv_path=None) -> pd.DataFrame:
    """Stage 2 - keep the best non-seasonal (p,d,q), grid the seasonal
    terms P, D, Q over {0,1} with period m = 24 (8 further fits).
    A full seasonal grid (0..6) would be computationally infeasible
    with m = 24; low seasonal orders are standard practice (FPP3)."""
    results = []
    for P, D, Q in product([0, 1], repeat=3):
        try:
            fit = SARIMAX(y_train, order=best_order,
                          seasonal_order=(P, D, Q, m)
                          ).fit(disp=False, maxiter=100)
            results.append([best_order, (P, D, Q, m), round(fit.aic, 1)])
        except Exception:
            continue
        if csv_path is not None:
            (pd.DataFrame(results, columns=["(p,d,q)", "(P,D,Q,m)", "AIC"])
             .sort_values("AIC").to_csv(csv_path, index=False))
    return (pd.DataFrame(results, columns=["(p,d,q)", "(P,D,Q,m)", "AIC"])
            .sort_values("AIC").reset_index(drop=True))


def fit_sarimax(y_train, order, seasonal_order, exog=None):
    """Fit the chosen SARIMA(X) specification."""
    return SARIMAX(y_train, exog=exog, order=order,
                   seasonal_order=seasonal_order
                   ).fit(disp=False, maxiter=200)


def residual_diagnostics(fit, tag: str) -> pd.DataFrame:
    """Residual checks: statsmodels plot_diagnostics (standardised
    residuals, histogram vs N(0,1), Q-Q plot, correlogram) plus a
    Ljung-Box test (H0: residuals are white noise)."""
    fig = fit.plot_diagnostics(figsize=(11, 8))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"fig_resid_{tag}.png", dpi=150)
    plt.close(fig)
    return acorr_ljungbox(fit.resid.dropna(), lags=[24, 48])


def forecast_with_ci(fit, horizon: int = 24, exog_future=None,
                     alpha: float = 0.05):
    """Forecast with (1-alpha) confidence intervals via get_forecast()."""
    fc = fit.get_forecast(steps=horizon, exog=exog_future)
    return fc.predicted_mean, fc.conf_int(alpha=alpha)


def rolling_sarimax_forecast(fit, y: pd.Series, test_index,
                             horizon: int = 24,
                             exog: pd.DataFrame = None) -> pd.Series:
    """Rolling-origin 24 h forecasts across the test period WITHOUT
    refitting: after each day the newly observed values are appended
    to the Kalman filter (parameters fixed), exactly as the model
    would be run operationally once per day."""
    preds = []
    current = fit
    for start in range(0, len(test_index), horizon):
        window = test_index[start:start + horizon]
        exog_fut = exog.loc[window] if exog is not None else None
        mean, _ = forecast_with_ci(current, len(window), exog_fut)
        preds.append(pd.Series(np.asarray(mean), index=window))
        current = current.append(y.loc[window], exog=exog_fut, refit=False)
    return pd.concat(preds)
