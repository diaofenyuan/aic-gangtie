from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_power.metrics import official_mape, summarize_predictions
from gas_power.models.base import prediction_columns
from gas_power.outputs import (
    OutputValidationError,
    validate_forecast_frame,
    write_forecast_csv,
    write_optimization_csv,
)


def test_official_mape_does_not_smooth_zero_denominator() -> None:
    with pytest.warns(RuntimeWarning, match="接近零"):
        metric = official_mape([0.0, 2.0], [1.0, 1.0])
    assert np.isinf(metric.mape)
    assert metric.zero_count == 1
    assert np.isneginf(metric.score_1_mape)


def test_metric_summary_has_all_required_scopes() -> None:
    tidy = pd.DataFrame(
        {
            "target": ["generator_1", "generator_1", "generator_all", "generator_all"],
            "horizon_steps": [1, 2, 1, 2],
            "y_true": [10.0, 12.0, 20.0, 24.0],
            "y_pred": [9.0, 13.0, 18.0, 25.0],
        }
    )
    summary = summarize_predictions(tidy, near_zero_threshold=1.0e-6)
    assert {"target_horizon", "target", "horizon", "overall"}.issubset(
        set(summary["scope"])
    )


def test_forecast_and_optimization_outputs_are_revalidated(tmp_path) -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="15min")
    targets = ["generator_1", "generator_all"]
    horizons = [1, 2]
    columns = prediction_columns(targets, horizons)
    forecast = pd.DataFrame(np.ones((2, len(columns))), index=index, columns=columns)
    checked = write_forecast_csv(
        forecast, tmp_path / "s_result.csv", targets, horizons
    )
    assert list(checked.columns) == ["datetime", *columns]
    assert (tmp_path / "s_result.csv").read_bytes().decode("utf-8")

    bad = forecast.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(OutputValidationError, match="缺失值"):
        validate_forecast_frame(bad, targets, horizons)

    gas_columns = [
        "opt_generator_use_blast_furnace_gas",
        "opt_generator_use_coke_gas",
        "opt_generator_use_converter_gas",
    ]
    plan = pd.DataFrame(np.ones((2, 3)), index=index, columns=gas_columns)
    checked_plan = write_optimization_csv(
        plan, tmp_path / "opt_result.csv", gas_columns
    )
    assert list(checked_plan.columns) == ["datetime", *gas_columns]

