from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import joblib
from pandas.testing import assert_frame_equal

from gas_power.data import (
    DataQualityReport,
    PreparedForecastData,
    clean_aligned_frame,
    prepare_scoring_with_history,
)
from gas_power.ensemble_selection import (
    apply_oof_weights,
    mape_sample_weights,
    project_forecasts,
)
from gas_power.features import CausalFeatureBuilder
from gas_power.models.baselines import LastValueModel
from gas_power.models.boosting import BoostingMultiHorizonModel
from gas_power.models.deep import NeuralResidualMultiHorizonModel
from gas_power.models.parameterization import ComponentReconstructionModel
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
