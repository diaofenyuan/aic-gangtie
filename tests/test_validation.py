from __future__ import annotations

import numpy as np
import pandas as pd

from gas_power.features import CausalFeatureBuilder
from gas_power.models.baselines import LastValueModel
from gas_power.validation import TimeSeriesRollingSplitter, run_rolling_validation


def test_expanding_and_rolling_splits_are_strictly_ordered() -> None:
    index = pd.date_range("2025-01-01", periods=400, freq="15min")
    for mode in ("expanding", "rolling"):
        splitter = TimeSeriesRollingSplitter(
            mode=mode,
            folds=2,
            initial_train_points=200,
            validation_points=20,
            step_points=40,
            rolling_train_points=120,
        )
        splits = splitter.split(index, max_horizon=8)
        assert len(splits) == 2
        assert all(split.train_end < split.validation_origins.min() for split in splits)


def test_rolling_validation_reports_target_horizon_and_worst_conditions() -> None:
    index = pd.date_range("2025-01-01", periods=400, freq="15min")
    x = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 50.0 + np.sin(x / 10.0),
            "generator_all": 200.0 + 2.0 * np.sin(x / 10.0),
            "production": 100.0 + np.cos(x / 15.0),
            "demand": 70.0 + np.cos(x / 15.0),
            "holder": 100_000.0 + 100.0 * np.sin(x / 12.0),
        },
        index=index,
    )
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1],
            "rolling_windows": [4],
            "rolling_statistics": ["mean", "std", "min", "max", "slope"],
        },
        roles={
            "targets": ["generator_1", "generator_all"],
            "gas_production": {"blast_furnace": "production"},
            "gas_user_demand": {"blast_furnace": "demand"},
            "gas_holder": {"blast_furnace": "holder"},
        },
    )
    artifacts = run_rolling_validation(
        frame=frame,
        model_factory=LastValueModel,
        splitter=TimeSeriesRollingSplitter("expanding", 2, 200, 20, 40),
        target_columns=["generator_1", "generator_all"],
        horizons=[1, 2, 8],
        interval_minutes=15,
        near_zero_threshold=1.0e-6,
        worst_error_count=10,
        feature_builder=builder,
    )
    assert len(artifacts.predictions) == 2 * 20 * 2 * 3
    assert "target_horizon" in set(artifacts.metrics["scope"])
    assert "feat_gas_balance_blast_furnace" in artifacts.worst_errors.columns

