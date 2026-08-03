from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gas_power.features import CausalFeatureBuilder
from gas_power.gpu_gate import evaluate_residual_gate
from gas_power.models.base import prediction_columns
from gas_power.models.baselines import LastValueModel
from gas_power.models.boosting import BoostingMultiHorizonModel
from gas_power.postprocessing import PhysicalForecastPostprocessor


class _ResidualModel(BoostingMultiHorizonModel):
    def _new_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_iter=10, random_state=2026)


def test_residual_route_and_physical_postprocessing_run_on_cpu() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="15min")
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 30.0 + 0.1 * x + np.sin(x / 10.0),
            "generator_all": 150.0 + 0.3 * x + np.sin(x / 8.0),
        },
        index=index,
    )
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 4],
            "rolling_windows": [4],
            "rolling_statistics": ["mean", "std"],
        },
        roles={"targets": ["generator_1", "generator_all"]},
    )
    model = _ResidualModel(
        backend="lightgbm",
        strategy="global",
        target_mode="residual",
        feature_builder=builder,
        baseline_model=LastValueModel(),
    )
    model.fit(frame.iloc[:220], ["generator_1"], [1, 2], train_end=index[219])
    origins = pd.DatetimeIndex(index[220:222])
    prediction = model.predict(frame, origins, ["generator_1"], [1, 2])
    assert np.isfinite(prediction.to_numpy()).all()

    columns = prediction_columns(["generator_1", "generator_all"], [1, 2])
    unsafe = pd.DataFrame(
        [[-1.0, 90.0, 70.0, 200.0], [80.0, 150.0, 60.0, 500.0]],
        index=origins,
        columns=columns,
    )
    result = PhysicalForecastPostprocessor(
        {
            "non_negative": True,
            "target_capacity_mw": {"generator_1": 50.0, "generator_all": 440.0},
            "ramp_limit_mw_per_step": None,
            "enforce_target_consistency": True,
        }
    ).apply(unsafe, frame, origins, ["generator_1", "generator_all"], [1, 2])
    assert result.adjusted_cells > 0
    assert result.predictions.to_numpy().min() >= 0.0
    assert result.predictions[columns[:2]].to_numpy().max() <= 50.0


def test_residual_gate_requires_all_fold_evidence() -> None:
    residual = evaluate_residual_gate(
        {"require_all_folds_improved": True, "min_fold_mape_gain": 0.001},
        [0.002, -0.001],
    )
    assert residual["allowed"] is False
