"""Tests for the evaluation metrics."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from appliance_energy import evaluation as ev

RNG = np.random.default_rng(0)
Y_TRAIN = RNG.uniform(50, 150, 500)
Y_TRUE = RNG.uniform(50, 150, 48)


def test_perfect_forecast_scores_zero():
    assert ev.mae(Y_TRUE, Y_TRUE) == 0
    assert ev.rmse(Y_TRUE, Y_TRUE) == 0
    assert ev.mase(Y_TRUE, Y_TRUE, Y_TRAIN) == 0
    assert ev.bias(Y_TRUE, Y_TRUE) == 0


def test_rmse_at_least_mae():
    pred = Y_TRUE + RNG.normal(0, 20, 48)
    assert ev.rmse(Y_TRUE, pred) >= ev.mae(Y_TRUE, pred)


def test_bias_sign():
    assert ev.bias(Y_TRUE, Y_TRUE + 10) > 0   # over-forecast
    assert ev.bias(Y_TRUE, Y_TRUE - 10) < 0   # under-forecast


def test_mase_equals_one_for_snaive_scale():
    # forecast whose MAE equals the seasonal-naive in-sample MAE -> 1
    m = 24
    scale = np.mean(np.abs(Y_TRAIN[m:] - Y_TRAIN[:-m]))
    pred = Y_TRUE + scale
    assert abs(ev.mase(Y_TRUE, pred, Y_TRAIN, m=m) - 1) < 1e-9
