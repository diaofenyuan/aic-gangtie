from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import joblib
from pandas.testing import assert_frame_equal

from gas_power.config import load_config
from gas_power.data import (
    DataError,
    DataQualityReport,
    PreparedForecastData,
    clean_aligned_frame,
    inspect_submission_input_quality,
    normalize_submission_input_frame,
    prepare_submission_sources,
    prepare_scoring_with_history,
    sanitize_submission_features,
)
from gas_power.ensemble_selection import (
    apply_oof_weights,
    cross_fit_hourly_mape_calibration,
    evaluate_oof_column_gate,
    fit_oof_weights,
    fit_hourly_mape_calibration,
    mape_sample_weights,
    project_forecasts,
)
from gas_power.features import CausalFeatureBuilder
from gas_power.models.baselines import LastValueModel
from gas_power.models.boosting import BoostingMultiHorizonModel
from gas_power.models.deep import NeuralResidualMultiHorizonModel
from gas_power.models.factory import build_model
from gas_power.models.parameterization import (
    ComponentReconstructionModel,
    GasAvailabilityForecastModel,
)
from gas_power.pipeline import (
    _assert_selection_fold_coverage,
    _combine_selection_predictions,
    _selection_oof_local_entries,
)
from gas_power.scoring import ConfigurableScorer
from gas_power.tuning import candidate_config_from_settings, select_diverse_trials_for_review
from gas_power.validation import RecentWindowSplitter, run_rolling_validation


def _prepared(
    frame: pd.DataFrame,
    *,
    source: str,
    targets: tuple[str, ...] = ("generator_1", "generator_all"),
) -> PreparedForecastData:
    raw_targets = frame[list(targets)].copy(deep=True)
    flags = pd.DataFrame(index=frame.index)
    return PreparedForecastData(
        model_input=frame.copy(deep=True),
        raw_targets=raw_targets,
        raw_observations=frame.copy(deep=True),
        missing_flags=flags.copy(),
        anomaly_flags=flags.copy(),
        label_valid_mask=raw_targets.notna(),
        source=source,
    )


def test_tuned_candidate_can_use_global_multi_horizon_strategy() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "official_preliminary.yaml")
    candidate = candidate_config_from_settings(
        config,
        backend="lightgbm",
        training_window_days=60,
        half_life_days=30,
        parameters={"n_estimators": 150},
        parameterization="direct",
        strategy="global",
    )

    assert candidate.section("forecast")["machine_learning"]["strategy"] == "global"
    assert candidate.section("residual_model")["strategy"] == "global"
    assert build_model(candidate).fit_progress_steps(
        ["generator_1", "generator_all"], list(range(1, 9))
    ) == 2


def test_submission_sources_drop_training_empty_columns_and_repair_missing() -> None:
    train_index = pd.date_range("2025-01-01", periods=120, freq="15min")
    score_index = pd.date_range(train_index[-1] + pd.Timedelta(minutes=15), periods=8, freq="15min")
    training = pd.DataFrame(
        {
            "valid": np.linspace(10.0, 20.0, len(train_index)),
            "invalid": np.nan,
        },
        index=train_index,
    )
    scoring = pd.DataFrame(
        {"valid": np.linspace(21.0, 28.0, len(score_index)), "invalid": np.nan},
        index=score_index,
    )
    scoring.loc[score_index[0], "valid"] = np.nan

    repaired, diagnostics = prepare_submission_sources(
        training,
        scoring,
        {"imputation": {"method": "ffill", "limit": None}, "outliers": {"enabled": False}},
        score_index,
        history_points=96,
    )

    assert list(repaired.columns) == ["valid"]
    assert repaired.at[score_index[0], "valid"] == training.iloc[-1]["valid"]
    assert np.isfinite(repaired.to_numpy(dtype=float)).all()
    assert diagnostics["invalid_columns"] == ["invalid"]


def test_submission_features_use_training_only_pruning_and_finite_fallback() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    training = pd.DataFrame(
        {
            "feat_keep": np.arange(8.0),
            "feat_duplicate": np.arange(8.0),
            "feat_constant": 1.0,
            "feat_empty": np.nan,
        },
        index=index,
    )
    scoring = training.iloc[-3:].copy()
    scoring.iloc[0, scoring.columns.get_loc("feat_keep")] = np.nan

    cleaned, diagnostics = sanitize_submission_features(training, scoring)

    assert list(cleaned.columns) == ["feat_keep"]
    assert np.isfinite(cleaned.to_numpy(dtype=float)).all()
    assert diagnostics["duplicate"] == ["feat_duplicate"]
    assert diagnostics["constant"] == ["feat_constant"]
    assert diagnostics["all_nonfinite"] == ["feat_empty"]


def test_submission_matrix_normalization_passes_strict_quality_gate() -> None:
    index = pd.date_range("2025-05-01", periods=12, freq="15min")
    feature_keep = np.array([*range(11), 100.0], dtype=float)
    frame = pd.DataFrame(
        {
            "feature_keep": feature_keep,
            "feature_duplicate": feature_keep,
            "feature_constant": 5.0,
            "feature_zero_iqr": [0.0] * 11 + [1.0],
        },
        index=index,
    )

    cleaned, diagnostics = normalize_submission_input_frame(
        frame,
        {
            "drop_constant_columns": True,
            "drop_duplicate_columns": True,
            "iqr_multiplier": 1.5,
        },
    )

    assert list(cleaned.columns) == ["feature_keep"]
    assert cleaned["feature_keep"].max() < 100.0
    assert diagnostics["winsorized_cells"] == 2
    assert diagnostics["dropped_constant_columns_before_winsor"] == [
        "feature_constant"
    ]
    assert diagnostics["dropped_duplicate_columns_before_winsor"] == [
        "feature_duplicate"
    ]
    assert diagnostics["dropped_constant_columns_after_winsor"] == [
        "feature_zero_iqr"
    ]
    assert diagnostics["passed"]
    assert diagnostics["clip_iqr_multiplier"] == 1.0
    assert diagnostics["final_quality"]["constant_columns"] == []
    assert diagnostics["final_quality"]["duplicate_columns"] == []
    assert diagnostics["final_quality"]["iqr_outlier_cells"] == 0
    assert diagnostics["final_quality"]["iqr_outlier_cells_all_methods"] == 0
    assert diagnostics["final_quality"]["zscore_outlier_cells"] == 0
    serialized_quality = inspect_submission_input_quality(cleaned)
    assert serialized_quality["iqr_outlier_cells_all_methods"] == 0
    assert serialized_quality["zscore_outlier_cells"] == 0


def test_outlier_detection_never_replaces_finite_labels() -> None:
    index = pd.date_range("2025-01-01", periods=80, freq="15min")
    frame = pd.DataFrame(
        {
            "generator_1": 100.0 + np.sin(np.arange(len(index))),
            "generator_all": 200.0 + np.cos(np.arange(len(index))),
        },
        index=index,
    )
    frame.loc[index[60], "generator_1"] = 900.0
    original = frame.copy(deep=True)
    cleaned = clean_aligned_frame(
        frame,
        {
            "imputation": {"method": "ffill", "limit": 8},
            "outliers": {
                "enabled": True,
                "window": 24,
                "min_periods": 12,
                "iqr_multiplier": 3.0,
            },
        },
        DataQualityReport(),
    )
    assert_frame_equal(cleaned[list(original.columns)], original)
    assert cleaned.at[index[60], "feat_outlier__generator_1"] == 1
    assert "feat_robust__generator_1" in cleaned


def test_validation_always_scores_against_explicit_raw_targets() -> None:
    index = pd.date_range("2025-01-01", periods=260, freq="15min")
    model_frame = pd.DataFrame(
        {"generator_1": 100.0, "generator_all": 200.0}, index=index
    )
    raw_targets = model_frame.copy()
    raw_targets.loc[index[225], "generator_1"] = 50.0
    artifacts = run_rolling_validation(
        frame=model_frame,
        raw_targets=raw_targets,
        model_factory=LastValueModel,
        splitter=RecentWindowSplitter(folds=1, validation_points=40, step_points=40),
        target_columns=["generator_1", "generator_all"],
        horizons=[1],
        interval_minutes=15,
        near_zero_threshold=1.0e-6,
        worst_error_count=10,
    )
    row = artifacts.predictions.loc[
        (artifacts.predictions["target_datetime"] == index[225])
        & (artifacts.predictions["target"] == "generator_1")
    ].iloc[0]
    assert row["y_true"] == 50.0
    assert row["y_pred"] == 100.0


def test_scoring_context_fill_and_future_perturbation_are_causal() -> None:
    train_index = pd.date_range("2025-01-01", periods=20, freq="15min")
    scoring_index = pd.date_range(train_index[-1] + pd.Timedelta(minutes=15), periods=8, freq="15min")
    training_frame = pd.DataFrame(
        {"generator_1": np.arange(20.0), "generator_all": np.arange(20.0) + 100.0},
        index=train_index,
    )
    scoring_frame = pd.DataFrame(
        {"generator_1": np.arange(8.0) + 20.0, "generator_all": np.arange(8.0) + 120.0},
        index=scoring_index,
    )
    scoring_frame.loc[scoring_index[0], "generator_1"] = np.nan
    settings = {"imputation": {"method": "ffill", "limit": 8}, "outliers": {"enabled": False}}
    training = _prepared(training_frame, source="training")
    scoring = _prepared(scoring_frame, source="scoring")
    baseline = prepare_scoring_with_history(training, scoring, settings, history_points=8)
    assert baseline.model_input.at[scoring_index[0], "generator_1"] == training_frame.iloc[-1]["generator_1"]

    perturbed_frame = scoring_frame.copy()
    perturbed_frame.loc[scoring_index[4]:, "generator_all"] += 10_000.0
    candidate = prepare_scoring_with_history(
        training, _prepared(perturbed_frame, source="scoring"), settings, history_points=8
    )
    assert_frame_equal(
        baseline.model_input.loc[:scoring_index[3]],
        candidate.model_input.loc[:scoring_index[3]],
    )


def test_scoring_context_allows_shared_official_reference_timestamp() -> None:
    train_index = pd.date_range("2025-04-30 23:15:00", periods=4, freq="15min")
    scoring_index = pd.date_range(train_index[-1], periods=3, freq="15min")
    training_frame = pd.DataFrame(
        {
            "generator_1": [10.0, 11.0, 12.0, 13.0],
            "generator_all": [20.0, 21.0, 22.0, 23.0],
        },
        index=train_index,
    )
    scoring_frame = pd.DataFrame(
        {
            "generator_1": [13.0, 14.0, 15.0],
            "generator_all": [23.0, 24.0, 25.0],
        },
        index=scoring_index,
    )
    settings = {
        "imputation": {"method": "ffill", "limit": 8},
        "outliers": {"enabled": False},
    }

    prepared = prepare_scoring_with_history(
        _prepared(training_frame, source="training"),
        _prepared(scoring_frame, source="scoring"),
        settings,
        history_points=3,
    )

    assert prepared.model_input.index.equals(scoring_index)
    assert prepared.model_input.at[scoring_index[0], "generator_1"] == 13.0


def test_scoring_context_rejects_training_data_after_scoring_start() -> None:
    train_index = pd.date_range("2025-05-01 00:00:00", periods=3, freq="15min")
    scoring_index = pd.date_range("2025-05-01 00:15:00", periods=3, freq="15min")
    training_frame = pd.DataFrame(
        {"generator_1": 10.0, "generator_all": 20.0}, index=train_index
    )
    scoring_frame = pd.DataFrame(
        {"generator_1": 11.0, "generator_all": 21.0}, index=scoring_index
    )
    settings = {
        "imputation": {"method": "ffill", "limit": 8},
        "outliers": {"enabled": False},
    }

    with pytest.raises(DataError, match="训练期结束时间晚于评分期起点"):
        prepare_scoring_with_history(
            _prepared(training_frame, source="training"),
            _prepared(scoring_frame, source="scoring"),
            settings,
            history_points=3,
        )


def test_scoring_source_is_rejected_by_training_interfaces() -> None:
    index = pd.date_range("2025-01-01", periods=80, freq="15min")
    frame = pd.DataFrame(
        {"generator_1": np.arange(80.0), "generator_all": np.arange(80.0) + 100.0},
        index=index,
    )
    with pytest.raises(ValueError, match="scoring"):
        LastValueModel().fit(
            frame,
            ["generator_1", "generator_all"],
            [1],
            data_source="scoring",
        )
    with pytest.raises(AssertionError, match="scoring"):
        run_rolling_validation(
            frame=frame,
            model_factory=LastValueModel,
            splitter=RecentWindowSplitter(folds=1, validation_points=10),
            target_columns=["generator_1", "generator_all"],
            horizons=[1],
            interval_minutes=15,
            near_zero_threshold=1.0e-6,
            worst_error_count=5,
            data_source="scoring",
        )


def test_mape_weights_and_fusion_constraints() -> None:
    weights = mape_sample_weights([10.0, 20.0, 40.0], floor_quantile=0.0)
    assert weights[0] > weights[1] > weights[2]
    assert np.isclose(weights.mean(), 1.0)
    combined = apply_oof_weights(
        {"first": np.array([10.0, 20.0]), "second": np.array([20.0, 40.0])},
        [0.75, 0.25],
    )
    assert np.allclose(combined, [12.5, 25.0])

    prediction = pd.DataFrame(
        {
            "generator_1_t+15_pred": [-1.0, 220.0],
            "generator_all_t+15_pred": [400.0, 100.0],
        }
    )
    projected = project_forecasts(
        prediction,
        target_columns=["generator_1", "generator_all"],
        horizons=[1],
        capacities={"generator_1": 200.0, "generator_all": 440.0},
    )
    assert projected.iloc[0].tolist() == [0.0, 400.0]
    assert projected.iloc[1].tolist() == [100.0, 100.0]


def test_boosting_uses_raw_labels_and_records_training_source() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="15min")
    frame = pd.DataFrame(
        {"generator_1": np.arange(100.0), "generator_all": np.arange(100.0) + 100.0},
        index=index,
    )
    builder = CausalFeatureBuilder(
        feature_config={"lag_steps": [1], "rolling_windows": [4], "rolling_statistics": ["mean"]},
        roles={"targets": ["generator_1", "generator_all"]},
    )
    model = BoostingMultiHorizonModel(
        backend="lightgbm",
        strategy="direct",
        target_mode="delta",
        feature_builder=builder,
        parameters={"n_estimators": 5, "n_jobs": 1},
    )
    model.fit(
        frame,
        ["generator_1"],
        [1],
        raw_targets=frame[["generator_1"]].copy(),
        data_source="training",
    )
    assert model.training_metadata_["raw_labels"] is True
    assert model.training_metadata_["data_source"] == "training"


def test_component_parameterization_reconstructs_total_target() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="15min")
    frame = pd.DataFrame(
        {"generator_1": np.arange(100.0), "generator_all": np.arange(100.0) + 100.0},
        index=index,
    )
    builder = CausalFeatureBuilder(
        feature_config={"lag_steps": [1], "rolling_windows": [4], "rolling_statistics": ["mean"]},
        roles={"targets": ["generator_1", "generator_all"]},
    )
    base = BoostingMultiHorizonModel(
        backend="lightgbm",
        strategy="direct",
        target_mode="delta",
        feature_builder=builder,
        parameters={"n_estimators": 5, "n_jobs": 1},
    )
    model = ComponentReconstructionModel(base, builder)
    model.fit(
        frame.iloc[:90],
        ["generator_1", "generator_all"],
        [1],
        raw_targets=frame.iloc[:90][["generator_1", "generator_all"]],
        feature_matrix=builder.transform(frame.iloc[:90]),
    )
    prediction = model.predict(
        frame,
        pd.DatetimeIndex([index[90]]),
        ["generator_1", "generator_all"],
        [1],
    )
    assert np.isfinite(prediction.to_numpy()).all()
    assert prediction.iloc[0]["generator_all_t+15_pred"] >= prediction.iloc[0]["generator_1_t+15_pred"]


def test_gas_availability_model_predicts_resource_then_consistent_generation() -> None:
    index = pd.date_range("2025-01-01", periods=240, freq="15min")
    x = np.arange(len(index), dtype=float)
    production = 120.0 + 5.0 * np.sin(x / 12.0)
    demand = 70.0 + 2.0 * np.cos(x / 8.0)
    holder = 50_000.0 + 20.0 * np.sin(x / 20.0)
    available = production - demand - np.r_[np.nan, np.diff(holder)]
    available = np.nan_to_num(available, nan=available[1])
    one = 80.0 + 0.3 * available
    total = one + 40.0 + 0.1 * available
    frame = pd.DataFrame(
        {
            "generator_1": one,
            "generator_all": total,
            "blast_furnace_1": production,
            "blast_furnace_user1": demand,
            "blast_furnace_gas_holder_1": holder,
        },
        index=index,
    )
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 4],
            "rolling_windows": [4],
            "rolling_statistics": ["mean", "std"],
        },
        roles={
            "targets": ["generator_1", "generator_all"],
            "gas_production": {"blast_furnace": "blast_furnace_1"},
            "gas_user_demand": {"blast_furnace": "blast_furnace_user1"},
            "gas_process_demand": {},
            "gas_holder": {"blast_furnace": "blast_furnace_gas_holder_1"},
        },
    )
    stage1 = BoostingMultiHorizonModel(
        backend="lightgbm",
        strategy="global",
        target_mode="residual",
        feature_builder=builder,
        parameters={"n_estimators": 5, "n_jobs": 1},
        baseline_model=LastValueModel(),
    )
    model = GasAvailabilityForecastModel(stage1, builder)
    model.fit(
        frame,
        ["generator_1", "generator_all"],
        [1, 2],
        raw_targets=frame[["generator_1", "generator_all"]],
    )

    origins = pd.DatetimeIndex(index[-4:])
    prediction = model.predict(
        frame, origins, ["generator_1", "generator_all"], [1, 2]
    )

    assert prediction.shape == (4, 4)
    assert np.isfinite(prediction.to_numpy(dtype=float)).all()
    for horizon in (15, 30):
        assert (
            prediction[f"generator_all_t+{horizon}_pred"]
            >= prediction[f"generator_1_t+{horizon}_pred"]
        ).all()


def test_column_gate_accepts_stable_per_horizon_gain_without_whole_model_gate() -> None:
    folds = np.repeat(np.arange(8), 4)
    truth = np.full(len(folds), 100.0)
    oof = pd.DataFrame(
        {
            "fold": folds,
            "target": "generator_all",
            "horizon_steps": 8,
            "y_true": truth,
            "last_value": truth + 10.0,
            "candidate": truth + np.where(folds < 6, 6.0, 11.0),
        }
    )

    gate = evaluate_oof_column_gate(
        oof,
        target_column="generator_all",
        horizon=8,
        candidate_column="candidate",
        minimum_non_degraded_folds=5,
        maximum_worst_degradation=0.02,
    )

    assert gate.passed
    assert gate.non_degraded_folds == 6


def test_hourly_calibration_is_cross_fitted_and_reduces_stable_bias() -> None:
    rows: list[dict[str, object]] = []
    for fold in range(6):
        for hour in range(0, 24, 2):
            rows.append(
                {
                    "fold": fold,
                    "target": "generator_all",
                    "horizon_steps": 4,
                    "target_datetime": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(days=fold, hours=hour),
                    "y_true": 105.0,
                    "y_pred": 100.0,
                }
            )
    predictions = pd.DataFrame(rows)

    calibrated = cross_fit_hourly_mape_calibration(
        predictions,
        hour_bin_size=4,
        minimum_samples=2,
        shrinkage=0.0,
    )
    factors = fit_hourly_mape_calibration(
        predictions,
        hour_bin_size=4,
        minimum_samples=2,
        shrinkage=0.0,
    )

    assert np.allclose(calibrated.to_numpy(dtype=float), 105.0)
    assert set(factors) == {"generator_all_t+60_pred"}
    assert all(np.isclose(value, 1.05) for value in factors["generator_all_t+60_pred"].values())


def test_selection_fold_combination_requires_aligned_four_plus_four_folds() -> None:
    def predictions(folds: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fold": np.arange(folds),
                "origin": pd.date_range("2025-01-01", periods=folds, freq="D"),
            }
        )

    combined = _combine_selection_predictions(
        predictions(4),
        predictions(4),
        recent_folds=4,
        cross_month_folds=4,
    )
    _assert_selection_fold_coverage(
        combined,
        recent_folds=4,
        cross_month_folds=4,
        context="测试候选",
    )

    with pytest.raises(ValueError, match="缺少"):
        _assert_selection_fold_coverage(
            combined.loc[combined["fold"] != "recent_3"],
            recent_folds=4,
            cross_month_folds=4,
            context="测试候选",
        )


def test_review_selection_keeps_best_trial_from_each_parameterization() -> None:
    trials = [
        SimpleNamespace(
            number=number,
            value=value,
            user_attrs={"candidate_config": {"parameterization": family}},
        )
        for number, value, family in (
            (0, 0.050, "component"),
            (1, 0.051, "component"),
            (2, 0.052, "direct"),
            (3, 0.053, "direct"),
            (4, 0.060, "gas_availability"),
            (5, 0.054, "component"),
        )
    ]

    selected = select_diverse_trials_for_review(trials, top_k=5)

    assert [trial.number for trial in selected] == [0, 1, 2, 3, 4]


def test_review_selection_keeps_target_and_column_specialists() -> None:
    trials = [
        SimpleNamespace(
            number=0,
            value=0.040,
            user_attrs={
                "candidate_config": {"parameterization": "component", "strategy": "direct"},
                "target_mape": {"generator_1": 0.060, "generator_all": 0.020},
                "column_mape": {"generator_1_h1": 0.060, "generator_all_h1": 0.020},
            },
        ),
        SimpleNamespace(
            number=1,
            value=0.041,
            user_attrs={
                "candidate_config": {"parameterization": "direct", "strategy": "direct"},
                "target_mape": {"generator_1": 0.030, "generator_all": 0.052},
                "column_mape": {"generator_1_h1": 0.030, "generator_all_h1": 0.052},
            },
        ),
        SimpleNamespace(
            number=2,
            value=0.042,
            user_attrs={
                "candidate_config": {"parameterization": "direct", "strategy": "global"},
                "target_mape": {"generator_1": 0.045, "generator_all": 0.018},
                "column_mape": {"generator_1_h1": 0.045, "generator_all_h1": 0.018},
            },
        ),
    ]

    selected = select_diverse_trials_for_review(trials, top_k=3)

    assert {trial.number for trial in selected} == {0, 1, 2}


def test_recent_fold_weight_moves_fusion_toward_recent_specialist() -> None:
    oof = pd.DataFrame(
        {
            "fold": ["recent_0"] * 4 + ["cross_month_0"] * 4,
            "target": ["generator_1"] * 8,
            "horizon_steps": [1] * 8,
            "y_true": [100.0] * 8,
            "baseline": [110.0] * 4 + [100.0] * 4,
            "recent_specialist": [100.0] * 4 + [110.0] * 4,
        }
    )

    equal, _ = fit_oof_weights(
        oof,
        ["baseline", "recent_specialist"],
        target_column="generator_1",
        horizon=1,
    )
    aggressive, _ = fit_oof_weights(
        oof,
        ["baseline", "recent_specialist"],
        target_column="generator_1",
        horizon=1,
        fold_group_weights={"recent_": 3.0, "cross_month_": 1.0},
    )

    assert aggressive[1] > equal[1]


def test_selection_oof_entries_replace_baseline_score_source(tmp_path: Path) -> None:
    path = tmp_path / "selection.csv"
    pd.DataFrame(
        {
            "fold": ["recent_0", "cross_month_0"],
            "target": ["generator_1", "generator_1"],
            "horizon_steps": [1, 1],
            "y_true": [100.0, 100.0],
            "y_pred": [98.0, 96.0],
        }
    ).to_csv(path, index=False)

    entries = _selection_oof_local_entries(
        path,
        ConfigurableScorer({"formula": "one_minus_mape"}),
    )

    assert entries["recent"]["folds"] == 1
    assert entries["recent"]["score"]["score_percent"] == pytest.approx(98.0)
    assert entries["cross_month"]["score"]["score_percent"] == pytest.approx(96.0)


@pytest.mark.parametrize("architecture", ["tcn", "patchtst"])
def test_deep_model_round_trip_preserves_predictions(
    architecture: str, tmp_path
) -> None:
    index = pd.date_range("2025-01-01", periods=96, freq="15min")
    phase = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "generator_1": 80.0 + np.sin(phase / 5.0),
            "generator_all": 190.0 + np.cos(phase / 7.0),
        },
        index=index,
    )
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 2],
            "rolling_windows": [4],
            "rolling_statistics": ["mean"],
        },
        roles={"targets": ["generator_1", "generator_all"]},
    )
    model = NeuralResidualMultiHorizonModel(
        architecture=architecture,
        feature_builder=builder,
        context_steps=8,
        epochs=2,
        patience=1,
        batch_size=16,
        hidden_size=8,
        patch_size=4,
        seeds=[7],
        device="cpu",
    )
    model.fit(
        frame.iloc[:80],
        ["generator_1", "generator_all"],
        [1, 2],
        raw_targets=frame.iloc[:80][["generator_1", "generator_all"]],
    )
    origins = pd.DatetimeIndex([index[80], index[81]])
    expected = model.predict(
        frame, origins, ["generator_1", "generator_all"], [1, 2]
    )
    model_path = tmp_path / f"{architecture}.joblib"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    actual = restored.predict(
        frame, origins, ["generator_1", "generator_all"], [1, 2]
    )
    assert_frame_equal(actual, expected, check_exact=True)
    assert np.isfinite(actual.to_numpy(dtype=float)).all()
