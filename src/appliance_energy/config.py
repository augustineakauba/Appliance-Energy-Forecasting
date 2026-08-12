"""Central configuration for the appliance energy forecasting project.

Keeping every 'magic number' in one place makes the pipeline
reproducible and easy to modify.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"

for _p in [RAW_DIR, PROCESSED_DIR, FIGURE_DIR, FORECAST_DIR, METRICS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "energydata_complete.csv"
HOURLY_CSV = PROCESSED_DIR / "appliance_hourly.csv"

DATA_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00374/energydata_complete.csv")

# ------------------------------------------------------------ modelling
RANDOM_STATE = 0

TARGET = "Appliances"          # target variable (Wh per 10 min, averaged)

DAILY_PERIOD = 24              # hourly data: 24 obs = 1 day
WEEKLY_PERIOD = 168            # hourly data: 168 obs = 1 week

HORIZON = 24                   # forecast horizon: next 24 hours
TEST_STEPS = 14 * 24           # test period: final 14 days (336 hours)

# SARIMA grid-search ranges required by the assignment brief
P_RANGE = range(0, 7)          # p = 0..6
D_RANGE = range(0, 3)          # d = 0..2
Q_RANGE = range(0, 7)          # q = 0..6

# Exogenous variables considered for SARIMAX (weather + calendar,
# see report Q5 for the discussion of availability at forecast origin)
SARIMAX_EXOG = ["T_out", "RH_out", "Windspeed",
                "hour_sin", "hour_cos", "is_weekend"]
