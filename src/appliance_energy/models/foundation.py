"""Foundation-model forecasting with Chronos (assignment Part 7).

Chronos (Ansari et al., 2024) is a pretrained time-series foundation
model: it tokenises a scaled context window and autoregressively
samples future tokens, producing a predictive DISTRIBUTION (quantiles)
zero-shot, with no training on our data at all.

Used here as a target-only, zero-shot model: it sees only the past of
the Appliances series (no covariates), so unlike SARIMAX/XGBoost its
forecast is a true operational forecast.
"""

import numpy as np
import pandas as pd

# 512 h context (~3 weeks) is plenty for daily/weekly structure and
# keeps inference fast on CPU.
CONTEXT_HOURS = 512


def load_chronos(model_name: str = "amazon/chronos-bolt-small",
                 device: str = "cpu"):
    """Load a Chronos-Bolt pipeline (downloads weights on first call)."""
    from chronos import BaseChronosPipeline
    import torch
    return BaseChronosPipeline.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.float32)


def chronos_forecast(pipeline, history: pd.Series, index,
                     horizon: int = 24):
    """One 24 h zero-shot forecast: returns median and 10-90 % band."""
    import torch
    context = torch.tensor(history.values[-CONTEXT_HOURS:],
                           dtype=torch.float32)
    quantiles, _ = pipeline.predict_quantiles(
        context=context, prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9])
    q = quantiles[0].numpy()
    median = pd.Series(q[:, 1], index=index, name="foundation_model")
    lo = pd.Series(q[:, 0], index=index)
    hi = pd.Series(q[:, 2], index=index)
    return median, lo, hi


def rolling_chronos_forecast(pipeline, y: pd.Series, test_index,
                             horizon: int = 24):
    """Rolling-origin 24 h Chronos forecasts across the test period
    (forecast issued once per 'day', context updated with actuals)."""
    med_all, lo_all, hi_all = [], [], []
    for start in range(0, len(test_index), horizon):
        window = test_index[start:start + horizon]
        history = y.loc[:window[0]].iloc[:-1]
        med, lo, hi = chronos_forecast(pipeline, history, window,
                                       len(window))
        med_all.append(med)
        lo_all.append(lo)
        hi_all.append(hi)
    return (pd.concat(med_all), pd.concat(lo_all), pd.concat(hi_all))
