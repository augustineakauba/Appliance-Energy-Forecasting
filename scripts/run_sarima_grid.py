"""Part 4 stage 1: the full ARIMA AIC grid required by the brief.

Loops over ALL combinations of p = 0..6, d = 0..2, q = 0..6
(147 fits). Slow, so it can be run one d at a time and checkpoints
results after every fit:

    python scripts/run_sarima_grid.py --d 0
    python scripts/run_sarima_grid.py --d 1
    python scripts/run_sarima_grid.py --d 2

Then stage 2 (seasonal terms on the best order):

    python scripts/run_sarima_grid.py --seasonal
"""

import argparse
import sys
from ast import literal_eval
from itertools import product
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from appliance_energy.config import (METRICS_DIR, P_RANGE, Q_RANGE, TARGET,
                                     TEST_STEPS)
from appliance_energy import data
from appliance_energy.models import sarimax as sar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=None,
                        help="run the (p,q) grid for this d only")
    parser.add_argument("--seasonal", action="store_true",
                        help="run the seasonal stage on the best order")
    args = parser.parse_args()

    hourly = data.resample_hourly(data.load_raw(), save=False)
    train, _ = data.train_test_split(hourly[TARGET], TEST_STEPS)

    if args.seasonal:
        # stage 2: read stage-1 results, grid P,D,Q in {0,1}, m=24
        frames = [pd.read_csv(METRICS_DIR / f"arima_grid_d{d}.csv")
                  for d in range(3)]
        full = pd.concat(frames).sort_values("AIC").reset_index(drop=True)
        full.to_csv(METRICS_DIR / "arima_grid_full.csv", index=False)
        # Parsimony rule: among models within 2 AIC of the minimum
        # (statistically indistinguishable, Burnham & Anderson 2004),
        # choose the one with the fewest parameters.
        cand = full[full["AIC"] <= full["AIC"].min() + 2].copy()
        cand["n_params"] = cand["(p,d,q)"].apply(
            lambda s: sum(literal_eval(str(s))))
        best_order = literal_eval(str(
            cand.sort_values(["n_params", "AIC"]).iloc[0]["(p,d,q)"]))
        print("Best non-seasonal order (parsimony rule):", best_order)
        res = sar.grid_search_seasonal(
            train, best_order, m=24,
            csv_path=METRICS_DIR / "sarima_seasonal_grid.csv")
        print(res.to_string(index=False))
    else:
        d = args.d
        res = sar.optimize_ARIMA(
            train, list(product(P_RANGE, Q_RANGE)), d,
            csv_path=METRICS_DIR / f"arima_grid_d{d}.csv")
        print(f"d={d} grid done ({len(res)} models). Top 5:")
        print(res.head().to_string(index=False))


if __name__ == "__main__":
    main()
