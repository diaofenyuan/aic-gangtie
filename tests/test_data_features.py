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
            "generator_use": 50_000.0 + 80.0 * x,
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
            "holder_capacity": 200_000.0,
            "holder_safe_min_fraction": 0.15,
            "holder_safe_max_fraction": 0.90,
            "holder_momentum_enabled": True,
            "holder_momentum_steps": [1, 4, 8, 16],
            "gas_load_mismatch": {
                "enabled": True,
                "windows": [4, 8],
                "minimum_load_mw": 1.0,
            },
        },
        roles={
            "targets": ["generator_1", "generator_all"],
            "gas_production": {"blast_furnace": "production"},
            "gas_user_demand": {"blast_furnace": "demand"},
            "gas_process_demand": {"blast_furnace": ["process_demand"]},
            "gas_holder": {"blast_furnace": "holder"},
            "generator_gas_use": {"blast_furnace": "generator_use"},
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
    candidate = clean_aligned_frame(perturbed, settings, DataQualityReport()).loc[
        :cutoff
    ]
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
    assert features.at[timestamp, "feat_time_minute"] == timestamp.minute
    assert features.at[timestamp, "feat_time_slot"] == 20
    assert features.at[timestamp, "feat_time_month"] == timestamp.month
    assert np.isclose(
        features.at[timestamp, "feat_state_generator_1_4_mean_gap"],
        frame.at[timestamp, "generator_1"] - expected,
    )
    assert np.isclose(
        features.at[timestamp, "feat_state_generator_1_4_relative_mean_gap"],
        frame.at[timestamp, "generator_1"] / expected - 1.0,
    )
    long_timestamp = frame.index[120]
    expected_96 = frame["generator_1"].iloc[24:120].mean()
    assert np.isclose(
        features.at[long_timestamp, "feat_state_generator_1_96_mean_gap"],
        frame.at[long_timestamp, "generator_1"] - expected_96,
    )
    assert np.isclose(
        features.at[
            long_timestamp,
            "feat_state_generator_1_96_relative_mean_gap",
        ],
        frame.at[long_timestamp, "generator_1"] / expected_96 - 1.0,
    )
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
    expected_holder_change_4 = (
        frame.at[frame.index[19], "holder"] - frame.at[frame.index[15], "holder"]
    ) / 60.0
    assert np.isclose(
        features.at[timestamp, "feat_holder_change_rate_blast_furnace_4"],
        expected_holder_change_4,
    )
    expected_holder_fraction = frame.at[frame.index[19], "holder"] / 200_000.0
    assert np.isclose(
        features.at[timestamp, "feat_holder_level_fraction_blast_furnace"],
        expected_holder_fraction,
    )
    assert np.isclose(
        features.at[timestamp, "feat_holder_low_margin_blast_furnace"],
        expected_holder_fraction - 0.15,
    )
    assert np.isclose(
        features.at[timestamp, "feat_holder_high_margin_blast_furnace"],
        0.90 - expected_holder_fraction,
    )
    expected_use_per_mw = (
        frame.at[frame.index[19], "generator_use"]
        / frame.at[timestamp, "generator_all"]
    )
    assert np.isclose(
        features.at[
            timestamp,
            "feat_gas_load_blast_furnace_use_per_mw_generator_all",
        ],
        expected_use_per_mw,
    )
    ratio = frame["generator_use"].shift(1) / frame["generator_all"]
    expected_reference = ratio.iloc[16:20].median()
    assert np.isclose(
        features.at[
            timestamp,
            "feat_gas_load_blast_furnace_use_per_mw_generator_all_4_relative_gap",
        ],
        expected_use_per_mw / expected_reference - 1.0,
    )
    assert "feat_state_generator_all_ramp_up" in features


def test_operating_features_compare_same_daily_and_weekly_slots() -> None:
    frame = _frame(periods=800)
    features = _builder().transform(frame)
    timestamp = frame.index[700]

    assert np.isclose(
        features.at[timestamp, "feat_state_generator_1_same_slot_day_delta"],
        frame.at[timestamp, "generator_1"]
        - frame.at[frame.index[700 - 96], "generator_1"],
    )
    assert np.isclose(
        features.at[timestamp, "feat_state_generator_1_same_slot_week_delta"],
        frame.at[timestamp, "generator_1"]
        - frame.at[frame.index[700 - 7 * 96], "generator_1"],
    )


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
