"""Generate the analysis notebooks (run once; notebooks are committed).

Each notebook wraps the reusable functions in src/appliance_energy with
a short narrative, so exploration lives in notebooks and logic in src/
(as recommended in the assignment README).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

SETUP = """import sys, warnings
from pathlib import Path
sys.path.append(str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from appliance_energy.config import *
from appliance_energy import data, eda, features, evaluation, plotting"""


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3",
                                    "display_name": "Python 3",
                                    "language": "python"}},
        "cells": [
            {"cell_type": kind,
             "metadata": {},
             **({"source": src.splitlines(keepends=True)} if kind == "markdown"
                else {"source": src.splitlines(keepends=True),
                      "outputs": [], "execution_count": None})}
            for kind, src in cells
        ],
    }


def save(name, cells):
    path = NB_DIR / name
    path.write_text(json.dumps(nb(cells), indent=1))
    print("wrote", path.name)


# ------------------------------------------------ 01 download/cleaning
save("01_data_download_and_cleaning.ipynb", [
    ("markdown", "# 01 - Data download and cleaning\n\n"
     "Loads the UCI *Appliances Energy Prediction* dataset "
     "(10-minute sampling, Jan-May 2016), runs data-quality checks, "
     "and resamples to hourly resolution.\n\n"
     "If `data/raw/energydata_complete.csv` is missing, download it "
     "from the UCI repository first (see `scripts/download_data.py`)."),
    ("code", SETUP),
    ("code", "raw = data.load_raw()\nraw.head()"),
    ("markdown", "## Data quality\n\nNo missing values, duplicates or "
     "gaps: the 10-minute grid is complete."),
    ("code", "data.data_quality_report(raw)"),
    ("markdown", "## Resample to hourly\n\nEnergy readings are averaged "
     "over each hour (6 samples/hour; mean and sum differ only by a "
     "factor 6, so model rankings are unaffected). Hourly resolution "
     "makes SARIMA with daily seasonality (m=24) tractable."),
    ("code", "hourly = data.resample_hourly(raw)\n"
     "print(hourly.shape)\nhourly[[TARGET]].describe().round(1)"),
])

# ------------------------------------------------------------- 02 EDA
save("02_exploratory_analysis.ipynb", [
    ("markdown", "# 02 - Exploratory analysis and stationarity\n\n"
     "Time-series components, STL decomposition, ADF/KPSS tests and "
     "ACF/PACF (assignment Part 1)."),
    ("code", SETUP),
    ("code", "hourly = data.resample_hourly(data.load_raw())\n"
     "y = hourly[TARGET]\n"
     "eda.plot_series_overview(hourly, TARGET)\n"
     "display(Image(str(FIGURE_DIR / 'fig01_eda_overview.png')))"),
    ("markdown", "The series has a clear **daily cycle** (low overnight, "
     "peaks around midday and 18:00) and a mild weekly effect "
     "(weekends higher), with no visible long-term trend."),
    ("code", "print(eda.stl_decomposition(y, period=24))\n"
     "print(eda.stl_decomposition(y, period=168))\n"
     "display(Image(str(FIGURE_DIR / 'fig02_stl_period24.png')))"),
    ("markdown", "Seasonal strength is ~0.32 (daily) and ~0.37 (weekly): "
     "seasonality is present but the remainder dominates - the series "
     "is noisy and spiky."),
    ("markdown", "## Stationarity tests (train period only)"),
    ("code", "train = y.iloc[:-TEST_STEPS]\n"
     "tests = pd.DataFrame([\n"
     "    eda.adf_test(train, 'raw'), eda.kpss_test(train, 'raw'),\n"
     "    eda.adf_test(train.diff(), 'diff(1)'),\n"
     "    eda.kpss_test(train.diff(), 'diff(1)'),\n"
     "    eda.adf_test(train.diff(24), 'sdiff(24)'),\n"
     "    eda.kpss_test(train.diff(24), 'sdiff(24)'),\n"
     "])\ntests"),
    ("markdown", "ADF rejects a unit root (p < 0.05) and KPSS does not "
     "reject stationarity: the hourly series is already "
     "**level-stationary**. Differencing is therefore not strictly "
     "required, although the AIC grid search later marginally prefers "
     "d = 1 (differences of near-stationary series remain stationary)."),
    ("code", "eda.acf_pacf_plots(train, 'fig03_acf_pacf_raw.png', 'raw')\n"
     "eda.differencing_figure(y)\n"
     "display(Image(str(FIGURE_DIR / 'fig03_acf_pacf_raw.png')))"),
    ("markdown", "The ACF shows clear spikes at lags 24, 48, 72 - the "
     "daily seasonal component that motivates seasonal models."),
])

# ------------------------------------------------------ 03 benchmarks
save("03_benchmark_models.ipynb", [
    ("markdown", "# 03 - Benchmark models\n\n"
     "Mean, naive, daily/weekly seasonal naive and drift (Part 3), "
     "evaluated with rolling-origin 24 h forecasts over the final "
     "14 days."),
    ("code", SETUP + "\nfrom appliance_energy.models import benchmarks as bm"),
    ("code", "hourly = data.resample_hourly(data.load_raw())\n"
     "y = hourly[TARGET]\n"
     "train, test = data.train_test_split(y, TEST_STEPS)\n"
     "rows = []\n"
     "fc_df = pd.DataFrame({'actual': test})\n"
     "for name, fn in bm.BENCHMARKS.items():\n"
     "    fc = bm.rolling_origin_forecast(y, test.index, fn, HORIZON)\n"
     "    fc_df[name] = fc\n"
     "    rows.append(evaluation.evaluate_forecast(name, test, fc, train))\n"
     "evaluation.metrics_table(rows)"),
    ("code", "plotting.plot_forecast_comparison(\n"
     "    train, test, fc_df.drop(columns='actual'),\n"
     "    'fig08_benchmarks.png', 'Benchmark forecasts')\n"
     "display(Image(str(FIGURE_DIR / 'fig08_benchmarks.png')))"),
    ("markdown", "The weekly seasonal naive has the best benchmark MAE "
     "(it captures both the daily shape and weekend effects), while "
     "the flat mean wins on RMSE because copying last week also "
     "copies its unrepeatable spikes. Naive and drift are far worse - "
     "appliance use mean-reverts within hours, so the last observed "
     "value carries little information about tomorrow."),
])

# --------------------------------------------------------- 04 sarimax
save("04_sarimax_models.ipynb", [
    ("markdown", "# 04 - SARIMA / SARIMAX\n\n"
     "AIC grid search (p=0..6, d=0..2, q=0..6, then seasonal terms), "
     "residual diagnostics, and 24 h forecasts with confidence "
     "intervals (Part 4).\n\n"
     "The full 147-model grid takes ~20 min; run "
     "`python scripts/run_sarima_grid.py --d 0/1/2` then "
     "`--seasonal` to reproduce. Results are read from "
     "`outputs/metrics/`."),
    ("code", SETUP + "\nfrom appliance_energy.models import sarimax as sar"),
    ("code", "grid = pd.read_csv(METRICS_DIR / 'arima_grid_full.csv')\n"
     "print('models fitted:', len(grid))\ngrid.head(10)"),
    ("code", "pd.read_csv(METRICS_DIR / 'sarima_seasonal_grid.csv')"),
    ("markdown", "Adding the seasonal component (1,1,1,24) improves AIC "
     "by ~545 - overwhelming evidence for daily seasonality. Final "
     "model: **SARIMA(1,1,3)(1,1,1,24)**, chosen by AIC with a "
     "parsimony rule (fewest parameters within 2 AIC of the "
     "minimum)."),
    ("code", "hourly = data.resample_hourly(data.load_raw())\n"
     "y = hourly[TARGET]\n"
     "train, test = data.train_test_split(y, TEST_STEPS)\n"
     "fit = sar.fit_sarimax(train, (1, 1, 3), (1, 1, 1, 24))\n"
     "print(fit.summary())"),
    ("markdown", "## Residual diagnostics\n\nLjung-Box p > 0.05 at lags "
     "24 and 48: residuals are consistent with white noise, i.e. the "
     "model has captured the autocorrelation structure. The histogram/"
     "Q-Q plots show heavy tails - the Gaussian intervals will be "
     "approximate."),
    ("code", "print(sar.residual_diagnostics(fit, 'sarima'))\n"
     "display(Image(str(FIGURE_DIR / 'fig_resid_sarima.png')))"),
    ("code", "mean24, ci24 = sar.forecast_with_ci(fit, HORIZON)\n"
     "fc = sar.rolling_sarimax_forecast(fit, y, test.index, HORIZON)\n"
     "evaluation.metrics_table([\n"
     "    evaluation.evaluate_forecast('sarima', test, fc, train)])"),
])

# -------------------------------------------------------------- 05 ml
save("05_feature_based_models.ipynb", [
    ("markdown", "# 05 - Feature-based model (XGBoost)\n\n"
     "Covariate engineering (Part 5) + tuned XGBoost with genuinely "
     "recursive 24 h forecasts (Part 6). Two variants:\n"
     "- `feature_model`: all covariates (conditional forecast)\n"
     "- `feature_model_pastonly`: lags/rolling/time only "
     "(true operational forecast)"),
    ("code", SETUP + "\nfrom appliance_energy.models import feature_models as fm"),
    ("code", "hourly = data.resample_hourly(data.load_raw())\n"
     "y = hourly[TARGET]\n"
     "train, test = data.train_test_split(y, TEST_STEPS)\n"
     "table = features.build_feature_matrix(hourly)\n"
     "print(table.shape)\nlist(table.columns)"),
    ("markdown", "Hyperparameters were tuned on the last 14 days of the "
     "*training* period (never the test set):"),
    ("code", "pd.read_csv(METRICS_DIR / 'xgb_tuning_feature_model.csv').head()"),
    ("code", "cols = features.feature_columns(table, include_covariates=True)\n"
     "tr = table.loc[table.index < test.index[0]]\n"
     "model = fm.fit_xgb(tr[cols], tr[TARGET])\n"
     "fc = fm.recursive_forecast(model, table, cols, y, test.index, HORIZON)\n"
     "evaluation.metrics_table([\n"
     "    evaluation.evaluate_forecast('feature_model', test, fc, train)])"),
    ("code", "grouped = fm.grouped_importance(model, cols, features.feature_group)\n"
     "plotting.plot_grouped_importance(grouped, 'fig09_feature_importance.png')\n"
     "display(Image(str(FIGURE_DIR / 'fig09_feature_importance.png')))"),
    ("markdown", "Careful with the grouped importances: the indoor-sensor "
     "group sums highest only because it has 19 features. Per feature, "
     "lags dominate (lag_1 alone is the top feature), then time and "
     "rolling features; outdoor weather is last. The past-only variant "
     "generalises *better* (see notebook 07) - the sensor covariates "
     "mostly add noise at a 24 h recursive horizon."),
])

# ------------------------------------------- 06 foundation (Colab-ready)
save("06_foundation_model.ipynb", [
    ("markdown", "# 06 - Foundation model: Chronos-2 zero-shot\n\n"
     "**This notebook is self-contained** so it can run on Google "
     "Colab (recommended: GPU runtime, but CPU works too, ~5 min).\n\n"
     "Chronos-2 (Ansari et al., 2024) is a pretrained time-series "
     "foundation model used here **zero-shot and target-only**: it "
     "sees only the history of `Appliances`, no covariates, and is "
     "never trained on our data. Forecasts are issued rolling-origin, "
     "24 h at a time, over the final 14 days - the same design as "
     "every other model.\n\n"
     "After running, download `fc_foundation_model.csv` and "
     "`foundation_model_interval.csv` into `outputs/forecasts/` of "
     "the repository, then run "
     "`python scripts/run_pipeline.py --stage evaluate`."),
    ("code", "%pip install -q chronos-forecasting torch pandas matplotlib"),
    ("code", "import numpy as np\nimport pandas as pd\n"
     "import matplotlib.pyplot as plt\nimport torch\n"
     "from chronos import Chronos2Pipeline"),
    ("markdown", "## Load and prepare the data (same as the pipeline)"),
    ("code", "URL = ('https://archive.ics.uci.edu/ml/machine-learning-"
     "databases/00374/energydata_complete.csv')\n"
     "df = pd.read_csv(URL)\n"
     "df['date'] = pd.to_datetime(df['date'])\n"
     "df = df.set_index('date').sort_index()\n"
     "y = df['Appliances'].resample('h').mean().interpolate('time').dropna()\n"
     "TEST_STEPS, HORIZON = 14 * 24, 24\n"
     "train, test = y.iloc[:-TEST_STEPS], y.iloc[-TEST_STEPS:]\n"
     "print(train.index.min(), '->', test.index.max(), len(y))"),
    ("markdown", "## Load Chronos-2"),
    ("code", "device_map = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
     "pipeline = Chronos2Pipeline.from_pretrained('amazon/chronos-2',\n"
     "                                            device_map=device_map)"),
    ("markdown", "## Rolling-origin 24 h forecasts over the test period\n\n"
     "One forecast per day; after each day the context is extended "
     "with the actual observations (as an operational system would)."),
    ("code", "def get_quantile_column(frame, q):\n"
     "    for col in [q, str(q), f'{q:.1f}', f'{q:.2f}']:\n"
     "        if col in frame.columns:\n"
     "            return col\n"
     "    raise KeyError(f'quantile {q} not in {list(frame.columns)}')\n"
     "\n"
     "med_all, lo_all, hi_all = [], [], []\n"
     "for start in range(0, TEST_STEPS, HORIZON):\n"
     "    window = test.index[start:start + HORIZON]\n"
     "    history = y.loc[:window[0]].iloc[:-1]\n"
     "    context_df = pd.DataFrame({'id': 'appliances',\n"
     "                               'timestamp': history.index,\n"
     "                               'target': history.to_numpy()})\n"
     "    pred = pipeline.predict_df(context_df,\n"
     "                               prediction_length=HORIZON,\n"
     "                               quantile_levels=[0.1, 0.5, 0.9],\n"
     "                               id_column='id',\n"
     "                               timestamp_column='timestamp',\n"
     "                               target='target')\n"
     "    pred = pred.sort_values('timestamp').tail(HORIZON)\n"
     "    med_all.append(pd.Series(\n"
     "        pred[get_quantile_column(pred, 0.5)].to_numpy(), index=window))\n"
     "    lo_all.append(pd.Series(\n"
     "        pred[get_quantile_column(pred, 0.1)].to_numpy(), index=window))\n"
     "    hi_all.append(pd.Series(\n"
     "        pred[get_quantile_column(pred, 0.9)].to_numpy(), index=window))\n"
     "    print('forecast day', start // 24 + 1, 'done')\n"
     "\n"
     "median = pd.concat(med_all).rename('foundation_model')\n"
     "lower = pd.concat(lo_all)\n"
     "upper = pd.concat(hi_all)"),
    ("markdown", "## Evaluate and save"),
    ("code", "def mae(a, b):\n"
     "    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))\n"
     "def rmse(a, b):\n"
     "    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))\n"
     "scale = np.mean(np.abs(train.values[24:] - train.values[:-24]))\n"
     "coverage = float(np.mean((test >= lower) & (test <= upper)))\n"
     "print(f'MAE  {mae(test, median):.2f}  RMSE {rmse(test, median):.2f}')\n"
     "print(f'MASE {mae(test, median)/scale:.2f}')\n"
     "print(f'80% interval coverage: {coverage:.2%}, mean width: '\n"
     "      f'{float(np.mean(upper - lower)):.1f} Wh')"),
    ("code", "median.to_csv('fc_foundation_model.csv')\n"
     "pd.DataFrame({'lower': lower, 'upper': upper}).to_csv(\n"
     "    'foundation_model_interval.csv')\n"
     "print('saved - move both CSVs into outputs/forecasts/')"),
    ("code", "fig, ax = plt.subplots(figsize=(14, 5))\n"
     "ax.plot(train.tail(72).index, train.tail(72), color='grey',\n"
     "        label='Training data')\n"
     "ax.plot(test.index, test, color='black', lw=1.4, label='Actual')\n"
     "ax.plot(median.index, median, color='tab:red', label='Chronos-2 median')\n"
     "ax.fill_between(test.index, lower, upper, alpha=0.2,\n"
     "                color='tab:red', label='Chronos-2 10-90%')\n"
     "ax.set_ylabel('Appliance energy use (Wh)')\n"
     "ax.legend()\n"
     "plt.tight_layout()\n"
     "plt.savefig('fig12_chronos.png', dpi=150)\n"
     "plt.show()"),
])

# ------------------------------------------------------ 07 comparison
save("07_model_comparison.ipynb", [
    ("markdown", "# 07 - Model comparison and error diagnostics\n\n"
     "Combines every cached forecast (Part 8). Run "
     "`python scripts/run_pipeline.py --stage evaluate` after all "
     "model stages (and after adding the Chronos forecast from "
     "notebook 06)."),
    ("code", SETUP),
    ("code", "fc_df = pd.read_csv(FORECAST_DIR / 'all_forecasts.csv',\n"
     "                    index_col=0, parse_dates=True)\n"
     "comparison = pd.read_csv(METRICS_DIR / 'model_comparison.csv',\n"
     "                         index_col=0)\ncomparison"),
    ("code", "display(Image(str(FIGURE_DIR / 'fig07_forecast_comparison.png')))"),
    ("code", "display(Image(str(FIGURE_DIR / 'fig10_next24h_ci.png')))"),
    ("code", "display(Image(str(FIGURE_DIR / 'fig11_error_diagnostics.png')))"),
    ("markdown", "See the report (Section 9) for the full discussion: "
     "which models beat the strongest benchmark, error structure by "
     "hour of day, and whether the extra complexity is justified."),
])

print("done")
