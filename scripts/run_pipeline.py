"""Main forecasting pipeline (assignment Parts 2-8).

Runs every model on the same design:
  target      : hourly Appliances energy use (Wh)
  test period : final 14 days (336 h)
  horizon     : 24 h, rolling-origin (one forecast issued per day)
  metrics     : MAE, RMSE, MAPE, MASE, Bias

Usage (from the repository root):
    python scripts/run_pipeline.py                 # everything
    python scripts/run_pipeline.py --stage sarimax # one stage only

Stages cache their forecasts in outputs/forecasts/, so the final
'evaluate' stage can combine whatever has been run.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from appliance_energy.config import (DAILY_PERIOD, FIGURE_DIR, FORECAST_DIR,
                                     HORIZON, METRICS_DIR, SARIMAX_EXOG,
                                     TARGET, TEST_STEPS)
from appliance_energy import data, evaluation, features, plotting
from appliance_energy.models import benchmarks as bm
from appliance_energy.models import feature_models as fm
from appliance_energy.models import sarimax as sar

# chosen by the AIC grid search (scripts/run_sarima_grid.py)
BEST_ORDER = (1, 1, 3)
BEST_SEASONAL = (1, 1, 1, 24)


def get_data():
    hourly = data.resample_hourly(data.load_raw())
    y = hourly[TARGET]
    train, test = data.train_test_split(y, TEST_STEPS)
    return hourly, y, train, test


def save_fc(name: str, series: pd.Series):
    series.rename(name).to_csv(FORECAST_DIR / f"fc_{name}.csv")
    print(f"  saved forecast: {name}")


# ------------------------------------------------------------ stages
def stage_benchmarks():
    """Part 3: mean, naive, seasonal naives, drift (rolling origin)."""
    _, y, train, test = get_data()
    for name, fn in bm.BENCHMARKS.items():
        fc = bm.rolling_origin_forecast(y, test.index, fn, HORIZON)
        save_fc(name, fc)


def stage_sarima():
    """Part 4a: target-only SARIMA."""
    hourly, y, train, test = get_data()

    # ---- target-only SARIMA
    print("fitting SARIMA", BEST_ORDER, BEST_SEASONAL)
    fit = sar.fit_sarimax(train, BEST_ORDER, BEST_SEASONAL)
    print(f"  AIC = {fit.aic:.1f}")

    # residual diagnostics + Ljung-Box (Part 4 requirement)
    lb = sar.residual_diagnostics(fit, "sarima")
    lb.to_csv(METRICS_DIR / "sarima_ljungbox.csv")
    print("  Ljung-Box:\n", lb)

    # save the parameter summary for the report
    with open(METRICS_DIR / "sarima_summary.txt", "w") as f:
        f.write(str(fit.summary()))

    # next-24h forecast with 95% confidence intervals (first test day)
    mean24, ci24 = sar.forecast_with_ci(fit, HORIZON, alpha=0.05)
    ci24.index = test.index[:HORIZON]
    mean24.index = test.index[:HORIZON]
    pd.concat([mean24.rename("mean"), ci24], axis=1).to_csv(
        FORECAST_DIR / "sarima_next24_ci.csv")

    # rolling-origin forecasts over the full test period
    fc = sar.rolling_sarimax_forecast(fit, y, test.index, HORIZON)
    save_fc("sarima", fc)


def stage_sarimax():
    """Part 4b: SARIMAX with exogenous weather + calendar covariates."""
    hourly, y, train, test = get_data()
    exog_all = features.add_time_features(hourly)[SARIMAX_EXOG]
    print("fitting SARIMAX with exog:", SARIMAX_EXOG)
    fitx = sar.fit_sarimax(train, BEST_ORDER, BEST_SEASONAL,
                           exog=exog_all.loc[train.index])
    print(f"  AIC = {fitx.aic:.1f}")
    sar.residual_diagnostics(fitx, "sarimax")
    fcx = sar.rolling_sarimax_forecast(fitx, y, test.index, HORIZON,
                                       exog=exog_all)
    save_fc("sarimax", fcx)


def stage_ml():
    """Parts 5-6: covariates + XGBoost (tuned, recursive 24h)."""
    hourly, y, train, test = get_data()
    table = features.build_feature_matrix(hourly)

    # two variants: with covariates (conditional) and past-only (true)
    for variant, use_cov in [("feature_model", True),
                             ("feature_model_pastonly", False)]:
        cols = features.feature_columns(table, include_covariates=use_cov)
        tr = table.loc[table.index < test.index[0]]

        # hypertuning on the last 14 days of the TRAIN period
        val_split = tr.index[-TEST_STEPS]
        tr_in = tr[tr.index < val_split]
        val = tr[tr.index >= val_split]
        tune = fm.tune_xgb(tr_in[cols], tr_in[TARGET],
                           val[cols], val[TARGET])
        tune.to_csv(METRICS_DIR / f"xgb_tuning_{variant}.csv", index=False)
        best = tune.iloc[0]
        print(f"{variant}: best params", dict(best.drop('val_RMSE')))

        # refit on the full training period with the best params
        model = fm.fit_xgb(tr[cols], tr[TARGET])
        model.set_params(max_depth=int(best["max_depth"]),
                         learning_rate=float(best["learning_rate"]),
                         n_estimators=int(best["n_estimators"]))
        model.fit(tr[cols], tr[TARGET], verbose=False)

        fc = fm.recursive_forecast(model, table, cols, y, test.index,
                                   HORIZON)
        save_fc(variant, fc)

        if use_cov:
            grouped = fm.grouped_importance(model, cols,
                                            features.feature_group)
            grouped.to_csv(METRICS_DIR / "xgb_grouped_importance.csv")
            plotting.plot_grouped_importance(
                grouped, "fig09_feature_importance.png")


def stage_foundation():
    """Part 7: Chronos-Bolt zero-shot (target-only, rolling origin)."""
    _, y, train, test = get_data()
    try:
        from appliance_energy.models import foundation as fnd
        pipe = fnd.load_chronos()
    except Exception as e:
        print("Chronos unavailable here:", e)
        print("Run notebooks/06_foundation_model.ipynb (e.g. on Colab) "
              "to produce fc_foundation_model.csv")
        return
    med, lo, hi = fnd.rolling_chronos_forecast(pipe, y, test.index, HORIZON)
    save_fc("foundation_model", med)
    pd.DataFrame({"lower": lo, "upper": hi}).to_csv(
        FORECAST_DIR / "foundation_model_interval.csv")


def stage_evaluate():
    """Part 8: combine all cached forecasts, score, plot."""
    _, y, train, test = get_data()

    fc_df = pd.DataFrame({"actual": test})
    for f in sorted(FORECAST_DIR.glob("fc_*.csv")):
        name = f.stem[3:]
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        fc_df[name] = s

    fc_df.to_csv(FORECAST_DIR / "all_forecasts.csv")

    rows = [evaluation.evaluate_forecast(c, test, fc_df[c], train,
                                         m=DAILY_PERIOD)
            for c in fc_df.columns if c != "actual"]
    table = evaluation.metrics_table(rows)
    table.to_csv(METRICS_DIR / "model_comparison.csv")
    print(table.to_string())

    # main comparison figure: strongest models only, for readability
    main_models = [c for c in ["seasonal_naive_daily", "sarima", "sarimax",
                               "feature_model", "foundation_model"]
                   if c in fc_df.columns]
    plotting.plot_forecast_comparison(
        train, test, fc_df[main_models],
        "fig07_forecast_comparison.png",
        "24h rolling-origin forecasts over the 14-day test period")

    # all benchmarks figure
    bench = [c for c in ["mean", "naive", "seasonal_naive_daily",
                         "seasonal_naive_weekly", "drift"]
             if c in fc_df.columns]
    plotting.plot_forecast_comparison(
        train, test, fc_df[bench], "fig08_benchmarks.png",
        "Benchmark forecasts over the 14-day test period")

    # first-test-day zoom with SARIMA confidence intervals
    ci_path = FORECAST_DIR / "sarima_next24_ci.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path, index_col=0, parse_dates=True)
        day1 = test.index[:HORIZON]
        sub = fc_df.loc[day1, [c for c in main_models]]
        plotting.plot_forecast_comparison(
            train, test.loc[day1], sub,
            "fig10_next24h_ci.png",
            "Next-24-hour forecasts with SARIMA 95% CI",
            context_days=3,
            ci=(ci.iloc[:, 1], ci.iloc[:, 2], "SARIMA 95% CI"))

    # error diagnostics vs the strongest benchmark and the best models
    diag = [c for c in ["seasonal_naive_daily", "sarima", "feature_model",
                        "foundation_model"] if c in fc_df.columns]
    plotting.plot_error_diagnostics(test, fc_df, diag,
                                    "fig11_error_diagnostics.png")


STAGES = {"benchmarks": stage_benchmarks, "sarima": stage_sarima,
          "sarimax": stage_sarimax, "ml": stage_ml,
          "foundation": stage_foundation, "evaluate": stage_evaluate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES) + ["all"],
                        default="all")
    args = parser.parse_args()
    to_run = list(STAGES) if args.stage == "all" else [args.stage]
    for name in to_run:
        print(f"\n===== stage: {name} =====")
        STAGES[name]()


if __name__ == "__main__":
    main()
