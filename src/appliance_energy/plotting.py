"""Shared plotting utilities for forecasts and error diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

from .config import FIGURE_DIR


def plot_forecast_comparison(train, test, forecast_df, fname, title,
                             context_days: int = 7, ci=None):
    """All model forecasts over the test period, with training context.

    ci: optional (lower, upper, label) tuple for an uncertainty band.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ctx = train.tail(context_days * 24)
    ax.plot(ctx.index, ctx, color="grey", lw=1, label="Training data")
    ax.plot(test.index, test, color="black", lw=1.6, label="Actual (test)")
    for col in forecast_df.columns:
        if col != "actual":
            ax.plot(forecast_df.index, forecast_df[col], lw=1.1,
                    alpha=0.9, label=col)
    if ci is not None:
        lo, hi, label = ci
        ax.fill_between(lo.index, lo, hi, alpha=0.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / fname, dpi=150)
    plt.close(fig)


def plot_error_diagnostics(test, forecast_df, best_models, fname):
    """Error diagnostics for the top models: error-vs-time,
    error-by-hour-of-day, and residual ACF."""
    n = len(best_models)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.2 * n), squeeze=False)
    for i, name in enumerate(best_models):
        err = forecast_df[name] - test
        axes[i, 0].plot(err.index, err, lw=0.7)
        axes[i, 0].axhline(0, color="k", lw=0.5)
        axes[i, 0].set_title(f"{name}: error over time")
        axes[i, 0].set_ylabel("Error (Wh)")
        axes[i, 0].tick_params(axis="x", rotation=30)

        by_hour = err.groupby(err.index.hour).mean()
        axes[i, 1].bar(by_hour.index, by_hour.values, color="tab:orange")
        axes[i, 1].set_title(f"{name}: mean error by hour")
        axes[i, 1].set_xlabel("Hour of day")

        plot_acf(err.dropna(), lags=48, ax=axes[i, 2])
        axes[i, 2].set_title(f"{name}: error ACF")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / fname, dpi=150)
    plt.close(fig)


def plot_grouped_importance(grouped: pd.Series, fname):
    """Horizontal bar chart of XGBoost importance summed by group."""
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot.barh(ax=ax, color="tab:purple")
    ax.set_title("XGBoost feature importance by group (gain)")
    ax.set_xlabel("Total importance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / fname, dpi=150)
    plt.close(fig)
