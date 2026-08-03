from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gas_power.features import CausalFeatureBuilder
from gas_power.models.base import prediction_column
from gas_power.models.baselines import (
    LastValueModel,
    LinearTrendModel,
    SeasonalMedianModel,
    SeasonalNaiveModel,
    WindowMeanModel,
)
from gas_power.models.ensemble import (
    HorizonWeightedEnsembleModel,
    HourlyCalibratedModel,
    WeightedEnsembleModel,
)
from gas_power.models.boosting import BoostingMultiHorizonModel
from gas_power.validation import assert_model_causality


def _history() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=8 * 96, freq="15min")
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {"generator_1": values, "generator_all": 2.0 * values}, index=index
    )


def test_all_required_baselines() -> None:
    frame = _history()
    origin = pd.DatetimeIndex([frame.index[-1]])
    targets = ["generator_1"]
    horizons = [1, 4]

    last = LastValueModel().fit(frame, targets, horizons)
    mean = WindowMeanModel(window=4).fit(frame, targets, horizons)
    trend = LinearTrendModel(window=8).fit(frame, targets, horizons)
    daily = SeasonalNaiveModel(period_steps=96).fit(frame, targets, horizons)
    weekly = SeasonalNaiveModel(period_steps=672).fit(frame, targets, horizons)
    daily_median = SeasonalMedianModel(period_steps=96, seasons=3).fit(
        frame, targets, horizons
    )

    assert (
        last.predict(frame, origin, targets, horizons).iloc[0, 0] == frame.iloc[-1, 0]
    )
    assert (
        mean.predict(frame, origin, targets, horizons).iloc[0, 0]
        == frame.iloc[-4:, 0].mean()
    )
    assert np.isclose(
        trend.predict(frame, origin, targets, horizons).iloc[0, 0],
        frame.iloc[-1, 0] + 1,
    )
    daily_time = origin[0] + pd.Timedelta(minutes=15) - pd.Timedelta(days=1)
    weekly_time = origin[0] + pd.Timedelta(minutes=15) - pd.Timedelta(days=7)
    assert (
        daily.predict(frame, origin, targets, horizons).iloc[0, 0]
        == frame.at[daily_time, "generator_1"]
    )
    assert (
        weekly.predict(frame, origin, targets, horizons).iloc[0, 0]
        == frame.at[weekly_time, "generator_1"]
    )
    expected_daily_median = np.median(
        [
            frame.at[daily_time - pd.Timedelta(days=day), "generator_1"]
            for day in range(3)
        ]
    )
    assert (
        daily_median.predict(frame, origin, targets, horizons).iloc[0, 0]
        == expected_daily_median
    )


def test_weighted_ensemble_and_model_leakage_check() -> None:
    frame = _history()
    targets = ["generator_1", "generator_all"]
    horizons = [1, 8]
    model = WeightedEnsembleModel(
        [(LastValueModel(), 0.75), (WindowMeanModel(window=4), 0.25)], clip_min=0.0
    )
    model.fit(frame.iloc[:600], targets, horizons, train_end=frame.index[599])
    origin = frame.index[620]
    prediction = model.predict(frame, pd.DatetimeIndex([origin]), targets, horizons)
    assert list(prediction.columns) == [
        prediction_column(target, horizon) for target in targets for horizon in horizons
    ]
    assert_model_causality(model, frame, origin, targets, horizons)


def test_hourly_calibrated_model_applies_target_time_factor() -> None:
    frame = _history()
    origin = pd.DatetimeIndex([pd.Timestamp("2025-01-08 23:45:00")])
    model = HourlyCalibratedModel(
        LastValueModel(),
        {"generator_1_t+15_pred": {"0": 1.05}},
        hour_bin_size=4,
    ).fit(frame, ["generator_1"], [1])

    prediction = model.predict(frame, origin, ["generator_1"], [1])

    assert np.isclose(
        prediction.iloc[0]["generator_1_t+15_pred"],
        frame.loc[origin[0], "generator_1"] * 1.05,
    )


def test_horizon_ensemble_applies_causal_regime_weights() -> None:
    frame = _history()
    stable_origin = frame.index[620]
    ramp_origin = frame.index[621]
    frame.loc[ramp_origin, "generator_1"] = (
        frame.loc[stable_origin, "generator_1"] + 10.0
    )
    column = prediction_column("generator_1", 1)
    model = HorizonWeightedEnsembleModel(
        {
            "last": LastValueModel(),
            "mean": WindowMeanModel(window=4),
        },
        {column: {"last": 1.0}},
        regime_column_weights={column: {"ramp_up": {"mean": 1.0}}},
        stable_delta_threshold_mw=2.0,
    ).fit(frame.iloc[:600], ["generator_1"], [1], train_end=frame.index[599])

    origins = pd.DatetimeIndex([stable_origin, ramp_origin])
    prediction = model.predict(frame, origins, ["generator_1"], [1])

    assert (
        prediction.at[stable_origin, column] == frame.at[stable_origin, "generator_1"]
    )
    assert np.isclose(
        prediction.at[ramp_origin, column],
        frame.loc[:ramp_origin, "generator_1"].iloc[-4:].mean(),
    )
    assert_model_causality(model, frame, ramp_origin, ["generator_1"], [1])


class _DependencyFreeBoostingModel(BoostingMultiHorizonModel):
    """用 sklearn 后端验证 direct/global 组样本逻辑，不强制可选库。"""

    def _new_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_iter=10, random_state=2026)


def test_boosting_target_time_features_handle_month_boundary() -> None:
    builder = CausalFeatureBuilder(
        feature_config={},
        roles={"targets": ["generator_1"]},
    )
    model = _DependencyFreeBoostingModel(
        backend="lightgbm",
        strategy="global",
        target_mode="absolute",
        feature_builder=builder,
    )
    origin = pd.Timestamp("2025-01-31 23:45:00")
    values = model._add_target_time_features(
        pd.DataFrame({"known": [1.0]}, index=pd.DatetimeIndex([origin])),
        horizon=1,
    )

    assert values.at[origin, "feat_target_time_minute"] == 0
    assert values.at[origin, "feat_target_time_slot"] == 0
    assert values.at[origin, "feat_target_time_month"] == 2


def test_boosting_supports_direct_and_global_multistep_strategies() -> None:
    frame = _history().iloc[:400]
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 4],
            "rolling_windows": [4],
            "rolling_statistics": ["mean", "std", "min", "max", "slope"],
        },
        roles={"targets": ["generator_1", "generator_all"]},
    )
    origins = pd.DatetimeIndex(frame.index[-2:])
    for strategy, target_mode in (("direct", "delta"), ("global", "absolute")):
        model = _DependencyFreeBoostingModel(
            backend="lightgbm",
            strategy=strategy,
            target_mode=target_mode,
            feature_builder=builder,
        )
        progress_updates: list[str] = []
        model.set_fit_progress_callback(progress_updates.append)
        model.fit(
            frame.iloc[:350],
            ["generator_1"],
            [1, 2],
            train_end=frame.index[349],
        )
        expected_steps = 2 if strategy == "direct" else 1
        assert model.fit_progress_steps(["generator_1"], [1, 2]) == expected_steps
        assert len(progress_updates) == expected_steps
        prediction = model.predict(frame, origins, ["generator_1"], [1, 2])
        assert prediction.shape == (2, 2)
        assert np.isfinite(prediction.to_numpy()).all()
        estimator = next(iter(model.models_.values()))
        assert "feat_target_time_day_sin" in estimator.feature_names_in_
        assert "feat_target_time_hour" in estimator.feature_names_in_
        assert "feat_target_time_minute" in estimator.feature_names_in_
        assert "feat_target_time_slot" in estimator.feature_names_in_
        assert "feat_target_time_month" in estimator.feature_names_in_


def test_boosting_temporal_mixup_adds_only_training_rows_deterministically() -> None:
    frame = _history().iloc[:400]
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1, 4],
            "rolling_windows": [4],
            "rolling_statistics": ["mean", "std"],
        },
        roles={"targets": ["generator_1"]},
    )

    def fit_model() -> _DependencyFreeBoostingModel:
        model = _DependencyFreeBoostingModel(
            backend="lightgbm",
            strategy="direct",
            target_mode="delta",
            feature_builder=builder,
            sample_weighting={
                "mape": True,
                "temporal_mixup_ratio": 0.25,
                "temporal_mixup_min_lambda": 0.70,
                "temporal_mixup_max_lambda": 0.90,
                "regime_stable_delta_threshold_mw": 2.0,
            },
        )
        return model.fit(
            frame.iloc[:350],
            ["generator_1"],
            [1],
            train_end=frame.index[349],
        )

    first = fit_model()
    second = fit_model()
    original = int(first.training_metadata_["training_rows_before_augmentation"])
    augmented = int(first.training_metadata_["augmented_training_rows"])

    assert original > 0
    assert augmented == round(original * 0.25)
    assert second.training_metadata_["augmented_training_rows"] == augmented
    origins = pd.DatetimeIndex(frame.index[-2:])
    assert np.allclose(
        first.predict(frame, origins, ["generator_1"], [1]),
        second.predict(frame, origins, ["generator_1"], [1]),
    )


def test_boosting_uses_constant_regressor_for_constant_residuals() -> None:
    index = pd.date_range("2025-01-01", periods=100, freq="15min")
    frame = pd.DataFrame({"generator_1": 42.0}, index=index)
    builder = CausalFeatureBuilder(
        feature_config={
            "lag_steps": [1],
            "rolling_windows": [4],
            "rolling_statistics": ["mean"],
        },
        roles={"targets": ["generator_1"]},
    )
    model = BoostingMultiHorizonModel(
        backend="catboost",
        strategy="global",
        target_mode="residual",
        feature_builder=builder,
        baseline_model=LastValueModel(),
    )

    model.fit(frame, ["generator_1"], [1, 2])
    prediction = model.predict(
        frame,
        pd.DatetimeIndex([index[-1]]),
        ["generator_1"],
        [1, 2],
    )

    assert model.training_metadata_["constant_response_models"] == ["generator_1"]
    assert np.allclose(prediction.to_numpy(dtype=float), 42.0)
