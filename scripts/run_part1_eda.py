"""Part 1: data preparation, EDA and stationarity testing.

Run from the repository root:  python scripts/run_part1_eda.py
Saves figures to outputs/figures/ and tables to outputs/metrics/.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import json

import pandas as pd

from appliance_energy.config import METRICS_DIR, TARGET, TEST_STEPS
from appliance_energy import data, eda


def main():
    # ---------------------------------------------------- load + clean
    raw = data.load_raw()
    quality = data.data_quality_report(raw)
    print("Data quality:", quality)

    hourly = data.resample_hourly(raw)
    print("Hourly shape:", hourly.shape)
    y = hourly[TARGET]

    with open(METRICS_DIR / "data_quality.json", "w") as f:
        json.dump({**quality, "hourly_rows": len(hourly)}, f, indent=2,
                  default=str)

    # ------------------------------------------------------- EDA plots
    eda.plot_series_overview(hourly, TARGET)

    # ------------------------------------- decomposition + seasonality
    strength_24 = eda.stl_decomposition(y, period=24)
    strength_168 = eda.stl_decomposition(y, period=168)
    print("STL strengths:", strength_24, strength_168)

    # ------------------------------- stationarity tests (train only!)
    train = y.iloc[:-TEST_STEPS]
    tests = [
        eda.adf_test(train, "raw"),
        eda.kpss_test(train, "raw"),
        eda.adf_test(train.diff(), "diff(1)"),
        eda.kpss_test(train.diff(), "diff(1)"),
        eda.adf_test(train.diff(24), "seasonal diff(24)"),
        eda.kpss_test(train.diff(24), "seasonal diff(24)"),
    ]
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(METRICS_DIR / "stationarity_tests.csv", index=False)
    print(tests_df.to_string(index=False))

    pd.DataFrame([strength_24, strength_168]).to_csv(
        METRICS_DIR / "stl_strengths.csv", index=False)

    # ----------------------------------------------- ACF/PACF figures
    eda.acf_pacf_plots(train, "fig03_acf_pacf_raw.png", "raw series")
    eda.acf_pacf_plots(train.diff(), "fig05_acf_pacf_diff1.png",
                       "first difference")
    eda.acf_pacf_plots(train.diff(24), "fig06_acf_pacf_sdiff24.png",
                       "seasonal difference (24)")
    eda.differencing_figure(y)

    # summary stats for the report
    print("\nTarget summary:\n", y.describe().round(1).to_string())


if __name__ == "__main__":
    main()
