# Appliance Energy Forecasting

A reproducible time-series forecasting pipeline for modelling and forecasting household appliance energy use, built for the *Appliances Energy Prediction* dataset (UCI ML Repository; Candanedo et al., 2017).

The project compares simple benchmark models, a SARIMA/SARIMAX model selected by a full AIC grid search, a feature-based XGBoost model, and the Chronos-2 time-series foundation model, all under an identical evaluation design.

## Project aim

Forecast short-term (24 h) household appliance energy use and evaluate whether increasingly complex models improve on simple benchmarks.

Main questions:

1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model such as XGBoost improve performance?
5. Does a time-series foundation model such as Chronos provide any additional benefit?
6. Which model would be most suitable for a practical smart-home energy forecasting system?

## Dataset

`energydata_complete.csv` — 19,735 rows at 10-minute resolution (11 Jan – 27 May 2016, a low-energy house in Stambruges, Belgium). Target variable: `Appliances` (appliance energy use, Wh). Covariates: `lights`, indoor temperature/humidity sensors `T1–T9` / `RH_1–RH_9`, and outdoor weather from Chievres airport (`T_out`, `RH_out`, `Press_mm_hg`, `Windspeed`, `Visibility`, `Tdewpoint`).

The 10-minute data are resampled to **hourly** resolution (complete hours only), which makes SARIMA with daily seasonality (m = 24) tractable.

## Forecasting design

| Setting | Value |
|---|---|
| Target | Hourly `Appliances` (Wh) |
| Horizon | 24 hours |
| Test period | Final 14 days (336 h) |
| Scheme | Rolling origin: one 24 h forecast issued per day; models see only data before each origin |
| Metrics | MAE, RMSE, MAPE, MASE (scale: in-sample daily seasonal naive), Bias |

## Models

1. **Benchmarks** — mean, naive, daily seasonal naive (lag 24), weekly seasonal naive (lag 168), drift.
2. **SARIMA/SARIMAX** — full AIC grid over p = 0..6, d = 0..2, q = 0..6 (147 fits), then seasonal terms P, D, Q ∈ {0,1} with m = 24. Selected model: SARIMA(1,1,3)(1,1,1,24); SARIMAX adds outdoor weather + calendar exogenous variables.
3. **Feature-based** — XGBoost on lag, rolling, time, indoor-sensor and weather features, with genuinely recursive 24 h forecasting (its own predictions feed the lags inside each window). A past-only variant (no sensor/weather covariates) is also fitted to quantify covariate value and forecast realism.
4. **Foundation model** — Chronos-2, zero-shot, target-only (notebook `06_foundation_model.ipynb`, self-contained / Colab-ready).

## Repository structure

```text
energy-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # energydata_complete.csv (downloaded)
│   └── processed/            # appliance_hourly.csv
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
├── src/
│   └── appliance_energy/
│       ├── config.py         # paths, constants, grid ranges
│       ├── data.py           # loading, quality checks, resampling, split
│       ├── eda.py            # EDA plots, STL, ADF/KPSS, ACF/PACF
│       ├── features.py       # time/lag/rolling/sensor/weather features
│       ├── evaluation.py     # MAE, RMSE, MAPE, MASE, Bias
│       ├── plotting.py       # forecast + diagnostic figures
│       └── models/
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           └── foundation.py
├── scripts/
│   ├── download_data.py
│   ├── run_part1_eda.py
│   ├── run_sarima_grid.py
│   └── run_pipeline.py
├── outputs/
│   ├── figures/
│   ├── forecasts/            # per-model CSVs + all_forecasts.csv
│   └── metrics/              # model_comparison.csv, grids, tests
├── reports/                  # final report
└── tests/
    ├── test_benchmarks.py
    ├── test_evaluation.py
    └── test_features.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
# 1. data
python scripts/download_data.py

# 2. EDA + stationarity tests (Part 1)
python scripts/run_part1_eda.py

# 3. SARIMA AIC grid search (Part 4; ~20 min total, checkpointed)
python scripts/run_sarima_grid.py --d 0
python scripts/run_sarima_grid.py --d 1
python scripts/run_sarima_grid.py --d 2
python scripts/run_sarima_grid.py --seasonal

# 4. all models + evaluation (Parts 3-8)
python scripts/run_pipeline.py                    # or --stage <name>
```

Stages (`benchmarks`, `sarima`, `sarimax`, `ml`, `foundation`, `evaluate`) cache forecasts in `outputs/forecasts/`, so they can be run independently and re-evaluated at any time.

**Foundation model:** if Chronos cannot run locally, execute `notebooks/06_foundation_model.ipynb` (e.g. on Google Colab), copy `fc_foundation_model.csv` and `foundation_model_interval.csv` into `outputs/forecasts/`, and re-run `python scripts/run_pipeline.py --stage evaluate`.

## Outputs

- `outputs/forecasts/all_forecasts.csv` — actual values + every model's rolling-origin forecast over the test period
- `outputs/metrics/model_comparison.csv` — MAE / RMSE / MAPE / MASE / Bias per model
- `outputs/metrics/arima_grid_full.csv`, `sarima_seasonal_grid.csv` — AIC grid results
- `outputs/metrics/stationarity_tests.csv`, `stl_strengths.csv` — Part 1 statistics
- `outputs/figures/` — EDA, ACF/PACF, residual diagnostics, forecast comparison, feature importance, error diagnostics

## Data leakage safeguards

- Chronological train/test split; test period never used for model or hyperparameter selection (XGBoost is tuned on the last 14 days of the *training* period).
- All lag/rolling features are `shift()`-ed so features at hour *t* use only data before *t*; enforced by `tests/test_features.py`.
- The XGBoost forecaster is genuinely recursive: inside each 24 h window, lags come from its own predictions, never from unseen actuals.
- Sensor/weather covariates at forecast time are **not** known 24 h ahead; models that use their realised test-set values (SARIMAX, `feature_model`) are reported as *conditional* forecasts, and past-only variants are provided for comparison.

## Tests

```bash
pytest
```

Covers forecast lengths, metric correctness (perfect forecast ⇒ 0; MASE = 1 at seasonal-naive scale), and leakage checks on lag/rolling features.

## References

- Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Ansari, A. F., et al. (2024). Chronos: Learning the language of time series. *TMLR*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
