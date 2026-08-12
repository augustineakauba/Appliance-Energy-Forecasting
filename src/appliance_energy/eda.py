"""Exploratory analysis and stationarity testing (assignment Part 1).

Provides: overview plots, STL decomposition with seasonal/trend
strength, ADF and KPSS tests, ACF/PACF plots and differencing checks.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from .config import FIGURE_DIR


def plot_series_overview(hourly: pd.DataFrame, target: str):
    """Four-panel EDA figure: full series, one-week zoom,
    mean hour-of-day profile and mean day-of-week profile."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(hourly.index, hourly[target], lw=0.4, color="tab:blue")
    axes[0, 0].set_title("(a) Hourly appliance energy use, full series")
    axes[0, 0].set_ylabel("Energy (Wh)")

    week = hourly[target].iloc[24 * 28: 24 * 35]
    axes[0, 1].plot(week.index, week, color="tab:blue")
    axes[0, 1].set_title("(b) One-week zoom (daily cycle visible)")

    by_hour = hourly[target].groupby(hourly.index.hour).mean()
    axes[1, 0].bar(by_hour.index, by_hour.values, color="tab:orange")
    axes[1, 0].set_title("(c) Mean energy by hour of day")
    axes[1, 0].set_xlabel("Hour of day")
    axes[1, 0].set_ylabel("Mean energy (Wh)")

    by_dow = hourly[target].groupby(hourly.index.dayofweek).mean()
    axes[1, 1].bar(range(7), by_dow.values, color="tab:green")
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu",
                                "Fri", "Sat", "Sun"])
    axes[1, 1].set_title("(d) Mean energy by day of week")

    for ax in axes[0]:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig01_eda_overview.png", dpi=150)
    plt.close(fig)


def stl_decomposition(series: pd.Series, period: int = 24) -> dict:
    """STL decomposition (trend + seasonal + remainder) and the
    seasonal/trend strength statistics of Hyndman & Athanasopoulos:

        F_seasonal = max(0, 1 - Var(remainder)/Var(seasonal+remainder))

    F close to 1 means strong seasonality; close to 0 means none.
    """
    res = STL(series, period=period, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(11, 7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"fig02_stl_period{period}.png", dpi=150)
    plt.close(fig)

    f_seasonal = max(0.0, 1 - np.var(res.resid) /
                     np.var(res.seasonal + res.resid))
    f_trend = max(0.0, 1 - np.var(res.resid) /
                  np.var(res.trend + res.resid))
    return {"period": period,
            "seasonal_strength": round(float(f_seasonal), 3),
            "trend_strength": round(float(f_trend), 3)}


def adf_test(series: pd.Series, name: str) -> dict:
    """Augmented Dickey-Fuller test.
    H0: unit root (non-stationary). p < 0.05 -> stationary."""
    stat, pvalue, _, _, crit, _ = adfuller(series.dropna())
    return {"series": name, "test": "ADF", "statistic": round(stat, 3),
            "p_value": round(pvalue, 4), "crit_5%": round(crit["5%"], 3),
            "conclusion": "stationary" if pvalue < 0.05 else "non-stationary"}


def kpss_test(series: pd.Series, name: str) -> dict:
    """KPSS test - complements ADF because the hypotheses are reversed.
    H0: series IS (level-)stationary. p > 0.05 -> stationary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, _, crit = kpss(series.dropna(), regression="c",
                                     nlags="auto")
    return {"series": name, "test": "KPSS", "statistic": round(stat, 3),
            "p_value": round(pvalue, 4), "crit_5%": round(crit["5%"], 3),
            "conclusion": "stationary" if pvalue > 0.05 else "non-stationary"}


def acf_pacf_plots(series: pd.Series, fname: str, title: str,
                   lags: int = 72):
    """ACF and PACF out to 72 lags (3 days) so that the 24-hour
    seasonal spikes are clearly visible."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF - {title}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF - {title}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / fname, dpi=150)
    plt.close(fig)


def differencing_figure(series: pd.Series):
    """Raw vs first-differenced vs seasonally-differenced series."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(series, lw=0.4)
    axes[0].set_title("Raw hourly series")
    axes[1].plot(series.diff(), lw=0.4, color="tab:orange")
    axes[1].set_title("First difference (lag 1)")
    axes[2].plot(series.diff(24), lw=0.4, color="tab:green")
    axes[2].set_title("Seasonal difference (lag 24)")
    for ax in axes:
        ax.set_ylabel("Wh")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig04_differencing.png", dpi=150)
    plt.close(fig)
