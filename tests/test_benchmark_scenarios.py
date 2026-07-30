from __future__ import annotations

import numpy as np
import pandas as pd

from gas_power.models.baselines import LastValueModel, LinearTrendModel
from gas_power.scoring import ConfigurableScorer
from gas_power.validation import TimeSeriesRollingSplitter, run_rolling_validation


def _validate(frame: pd.DataFrame, model_factory):
    return run_rolling_validation(
        frame=frame,
        model_factory=model_factory,
        splitter=TimeSeriesRollingSplitter("expanding", 1, 120, 24, 24),
        target_columns=["generator_1", "generator_all"],
        horizons=[1, 2, 4, 8],
        interval_minutes=15,
        near_zero_threshold=1.0e-6,
        worst_error_count=5,
    )


def test_last_value_is_perfect_on_constant_scenario() -> None:
    index = pd.date_range("2025-01-01", periods=200, freq="15min")
    frame = pd.DataFrame(
        {"generator_1": 50.0, "generator_all": 200.0}, index=index
    )
    artifacts = _validate(frame, LastValueModel)
    overall = artifacts.metrics.loc[artifacts.metrics["scope"] == "overall"].iloc[0]
    assert overall["mape"] == 0.0
    assert overall["score_1_mape"] == 1.0


def test_linear_trend_beats_last_value_on_linear_ramp() -> None:
    index = pd.date_range("2025-01-01", periods=200, freq="15min")
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {"generator_1": 20.0 + x, "generator_all": 100.0 + 2.0 * x}, index=index
    )
    last = _validate(frame, LastValueModel)
    trend = _validate(frame, lambda: LinearTrendModel(window=5))
    last_mape = float(last.metrics.loc[last.metrics["scope"] == "overall", "mape"].iloc[0])
    trend_mape = float(trend.metrics.loc[trend.metrics["scope"] == "overall", "mape"].iloc[0])
    assert trend_mape < last_mape
    assert trend_mape < 1.0e-12


def test_configurable_scorer_supports_raw_percent_and_weights() -> None:
    tidy = pd.DataFrame(
        {
            "target": ["generator_1", "generator_all"],
            "horizon_steps": [1, 1],
            "y_true": [100.0, 200.0],
            "y_pred": [99.0, 198.0],
        }
    )
    raw = ConfigurableScorer(
        {
            "formula": "one_minus_mape",
            "target_weights": {"generator_1": 0.5, "generator_all": 0.5},
        }
    ).score(tidy)
    percent = ConfigurableScorer(
        {"formula": "one_minus_mape_percent", "target_weights": {}}
    ).score(tidy)
    assert np.isclose(raw.final_score, 0.99)
    assert np.isclose(percent.final_score, 99.0)
