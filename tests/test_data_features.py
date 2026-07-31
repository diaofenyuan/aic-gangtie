from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gas_power.data import (
    DataQualityReport,
    _combine_sources,
    _resolve_sources,
    clean_aligned_frame,
)
from gas_power.features import (
    CausalFeatureBuilder,
    assert_feature_causality,
    assert_shift_before_rolling,
)


def _frame(periods: int = 160) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="15min")
    x = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 20.0 + x,
            "generator_all": 40.0 + 2.0 * x,
            "production": 100.0 + x,
            "demand": 70.0 + x * 0.5,
            "process_demand": 5.0 + x * 0.1,
            "holder": 90_000.0 + 100.0 * np.sin(x / 8.0),
        },
        index=index,
    )
    frame.index.name = "datetime"
    return frame


def _builder() -> CausalFeatureBuilder:
    return CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 4],
            "rolling_windows": [4, 8],
            "rolling_statistics": ["mean", "std", "min", "max", "slope"],
            "unit_online_threshold_mw": 1.0,
            "stable_delta_threshold_mw": 2.0,
        },
        roles={
            "targets": ["generator_1", "generator_all"],
            "gas_production": {"blast_furnace": "production"},
            "gas_user_demand": {"blast_furnace": "demand"},
            "gas_process_demand": {"blast_furnace": ["process_demand"]},
            "gas_holder": {"blast_furnace": "holder"},
        },
    )


def test_cleaning_is_causal_and_keeps_missing_indicators() -> None:
    frame = _frame()
    frame.iloc[30, frame.columns.get_loc("production")] = np.nan
    frame.iloc[70, frame.columns.get_loc("production")] *= 20.0
    settings = {
        "imputation": {"method": "ffill", "limit": 4},
        "outliers": {
            "enabled": True,
            "window": 24,
            "min_periods": 12,
            "iqr_multiplier": 3.0,
            "replace_with": "median",
        },
    }
    cutoff = frame.index[90]
    baseline = clean_aligned_frame(frame, settings, DataQualityReport()).loc[:cutoff]
    perturbed = frame.copy()
    perturbed.loc[perturbed.index > cutoff, "production"] += 1_000_000.0
    candidate = clean_aligned_frame(perturbed, settings, DataQualityReport()).loc[:cutoff]
    assert_frame_equal(baseline, candidate)
    assert baseline.loc[frame.index[30], "feat_missing__production"] == 1
    assert np.isfinite(baseline.loc[frame.index[30], "production"])


def test_features_shift_before_rolling_and_pass_future_perturbation() -> None:
    frame = _frame()
    builder = _builder()
    features = builder.transform(frame)
    timestamp = frame.index[20]
    expected = frame["generator_1"].iloc[16:20].mean()
    assert features.at[timestamp, "feat_roll_generator_1_4_mean"] == expected
    assert_shift_before_rolling(builder, frame, "generator_1", 4)
    assert_feature_causality(builder, frame, frame.index[100])
    assert "feat_gas_balance_blast_furnace" in features
    expected_balance = (
        frame.at[frame.index[19], "production"]
        - frame.at[frame.index[19], "demand"]
        - frame.at[frame.index[19], "process_demand"]
    )
    assert np.isclose(
        features.at[timestamp, "feat_gas_balance_blast_furnace"],
        expected_balance,
    )
    assert "feat_holder_change_rate_blast_furnace" in features
    expected_holder_change = (
        frame.at[frame.index[19], "holder"] - frame.at[frame.index[18], "holder"]
    ) / 15.0
    assert np.isclose(
        features.at[timestamp, "feat_holder_change_rate_blast_furnace"],
        expected_holder_change,
    )
    assert "feat_state_generator_all_ramp_up" in features


def test_feature_progress_reports_each_configured_source_group() -> None:
    builder = _builder()
    completed: list[str] = []

    builder.transform(_frame(), progress_callback=completed.append)

    assert len(completed) == builder.progress_steps()
    assert "构建字段特征 generator_1" in completed
    assert completed[-1] == "组装特征表"


def test_numbered_industrial_sources_are_aggregated_by_pattern() -> None:
    frame = pd.DataFrame(
        {
            "blast_furnace_1": [100.0, 110.0],
            "blast_furnace_2": [120.0, 130.0],
            "air_heater_1": [20.0, 21.0],
        }
    )
    mapping = {
        "sources": ["blast_furnace_1"],
        "patterns": [r"^blast_furnace_\d+$"],
        "combine": "sum",
    }
    sources = _resolve_sources(frame.columns, mapping)
    combined = _combine_sources(frame, sources, str(mapping["combine"]))

    assert sources == ["blast_furnace_1", "blast_furnace_2"]
    assert combined.tolist() == [220.0, 240.0]
