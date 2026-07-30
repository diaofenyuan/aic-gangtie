from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gas_power.availability import (
    FeatureAvailabilityRegistry,
    FieldAvailability,
)
from gas_power.features import CausalFeatureBuilder
from gas_power.models.base import prediction_columns
from gas_power.outputs import OutputValidationError, validate_forecast_frame
from gas_power.time_semantics import validate_target_times


def test_conservative_registry_excludes_future_leak_from_formal_features() -> None:
    index = pd.date_range("2025-01-01", periods=32, freq="15min")
    frame = pd.DataFrame(
        {
            "generator_all": np.arange(32, dtype=float),
            "future_generator_all_leak": np.arange(32, dtype=float) + 1.0,
        },
        index=index,
    )
    registry = FeatureAvailabilityRegistry(
        fields={
            "generator_all": FieldAvailability(
                name="generator_all",
                available_at_origin=True,
                min_lag_steps=0,
                is_label=True,
                allow_short=True,
                allow_long=True,
            ),
            "future_generator_all_leak": FieldAvailability(
                name="future_generator_all_leak",
                available_at_origin=False,
                is_label=True,
                allow_short=False,
                allow_long=False,
            ),
        }
    )
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 2],
            "rolling_windows": [4],
            "rolling_statistics": ["mean"],
        },
        roles={
            "targets": ["generator_all"],
            "candidate_features": ["future_generator_all_leak"],
        },
        availability=registry,
    )
    features = builder.transform(frame)
    assert any("generator_all" in column for column in features)
    assert not any("future_generator_all_leak" in column for column in features)


def test_wrong_target_time_and_shifted_submission_columns_are_rejected() -> None:
    origins = pd.date_range("2025-01-01", periods=2, freq="15min")
    with pytest.raises(ValueError, match="target_time"):
        validate_target_times(origins, origins, [1, 1], interval_minutes=15)

    targets = ["generator_1", "generator_all"]
    horizons = [1, 2]
    columns = prediction_columns(targets, horizons)
    frame = pd.DataFrame(np.ones((2, 4)), index=origins, columns=columns)
    shifted = frame[[columns[1], columns[0], columns[2], columns[3]]]
    with pytest.raises(OutputValidationError, match="列名或顺序"):
        validate_forecast_frame(shifted, targets, horizons)

    with pytest.raises(OutputValidationError, match="预测起点"):
        validate_forecast_frame(
            frame,
            targets,
            horizons,
            expected_origins=origins + pd.Timedelta(minutes=15),
        )


def test_generator_target_swap_is_caught_by_consistency_check() -> None:
    index = pd.date_range("2025-01-01", periods=1, freq="15min")
    columns = prediction_columns(["generator_1", "generator_all"], [1])
    frame = pd.DataFrame([[200.0, 50.0]], index=index, columns=columns)
    with pytest.raises(OutputValidationError, match="列互换"):
        validate_forecast_frame(
            frame,
            ["generator_1", "generator_all"],
            [1],
            enforce_target_consistency=True,
        )
