"""根据 YAML 构建模型，避免 CLI 写死实现。"""

from __future__ import annotations

from typing import Any, Mapping

from gas_power.availability import FeatureAvailabilityRegistry
from gas_power.config import ProjectConfig
from gas_power.features import CausalFeatureBuilder
from gas_power.models.base import ForecastModel
from gas_power.models.baselines import (
    DampedTrendModel,
    LastValueModel,
    LinearTrendModel,
    SeasonalNaiveModel,
    WindowMeanModel,
    WindowMedianModel,
)
from gas_power.models.boosting import BoostingMultiHorizonModel
from gas_power.models.ensemble import WeightedEnsembleModel
from gas_power.models.parameterization import ComponentReconstructionModel
from gas_power.models.deep import NeuralResidualMultiHorizonModel


def baseline_from_spec(spec: Mapping[str, Any], interval_minutes: int) -> ForecastModel:
    model_type = str(spec.get("type", ""))
    if model_type == "last_value":
        return LastValueModel(interval_minutes=interval_minutes)
    if model_type == "window_mean":
        return WindowMeanModel(
            window=int(spec.get("window", 8)), interval_minutes=interval_minutes
        )
    if model_type == "window_median":
        return WindowMedianModel(
            window=int(spec.get("window", 8)), interval_minutes=interval_minutes
        )
    if model_type == "linear_trend":
        return LinearTrendModel(
            window=int(spec.get("window", 8)), interval_minutes=interval_minutes
        )
    if model_type == "damped_trend":
        return DampedTrendModel(
            window=int(spec.get("window", 5)),
            damping=float(spec.get("damping", 0.85)),
            interval_minutes=interval_minutes,
        )
    if model_type == "seasonal_naive":
        return SeasonalNaiveModel(
            period_steps=int(spec.get("period_steps", 96)),
            interval_minutes=interval_minutes,
        )
    if model_type == "weighted_ensemble":
        component_specs = spec.get("components", [])
        if not isinstance(component_specs, list) or not component_specs:
            raise ValueError("融合基线 components 必须是非空列表")
        components = [
            (
                baseline_from_spec(component, interval_minutes),
                float(component.get("weight", 1.0)),
            )
            for component in component_specs
            if isinstance(component, Mapping)
        ]
        return WeightedEnsembleModel(components, clip_min=spec.get("clip_min"))
    raise ValueError(f"未知基线模型类型: {model_type}")


_baseline_from_spec = baseline_from_spec


def build_model(config: ProjectConfig) -> ForecastModel:
    forecast = config.section("forecast")
    model_spec = forecast.get("model", {})
    if not isinstance(model_spec, Mapping):
        raise ValueError("forecast.model 必须是字典")
    model_type = str(model_spec.get("type", "weighted_ensemble"))
    interval_minutes = int(config.section("optimization").get("interval_minutes", 15))

    if model_type == "weighted_ensemble":
        component_specs = model_spec.get("components", [])
        if not isinstance(component_specs, list):
            raise ValueError("融合模型 components 必须是列表")
        components: list[tuple[ForecastModel, float]] = []
        for component_spec in component_specs:
            if not isinstance(component_spec, Mapping):
                raise ValueError("融合子模型配置必须是字典")
            components.append(
                (
                    baseline_from_spec(component_spec, interval_minutes),
                    float(component_spec.get("weight", 1.0)),
                )
            )
        clip_value = model_spec.get("clip_min")
        return WeightedEnsembleModel(
            components,
            clip_min=float(clip_value) if clip_value is not None else None,
        )

    if model_type in {
        "last_value",
        "window_mean",
        "window_median",
        "linear_trend",
        "damped_trend",
        "seasonal_naive",
    }:
        return baseline_from_spec(model_spec, interval_minutes)

    if model_type in {
        "lightgbm",
        "catboost",
        "residual_lightgbm",
        "residual_catboost",
        "component_reconstruction",
    }:
        machine_learning = forecast.get("machine_learning", {})
        if not isinstance(machine_learning, Mapping):
            raise ValueError("forecast.machine_learning 必须是字典")
        roles = config.section("data")["roles"]
        builder = CausalFeatureBuilder(
            feature_config=config.section("features"),
            roles=roles,
            interval_minutes=interval_minutes,
            availability=FeatureAvailabilityRegistry.from_config(config),
            model_scope="long",
        )
        residual = model_type.startswith("residual_")
        parameterized = model_type == "component_reconstruction"
        backend = (
            str(machine_learning.get("backend", "lightgbm"))
            if parameterized
            else model_type.removeprefix("residual_")
        )
        residual_config = config.raw.get("residual_model", {})
        effective_ml = residual_config if residual else machine_learning
        baseline_spec = residual_config.get("baseline", {"type": "last_value"})
        if residual and not isinstance(baseline_spec, Mapping):
            raise ValueError("residual_model.baseline 必须是字典")
        base_model = BoostingMultiHorizonModel(
            backend=backend,
            strategy=str(effective_ml.get("strategy", "direct")),
            target_mode="residual" if residual else str(machine_learning.get("target_mode", "delta")),
            feature_builder=builder,
            parameters=effective_ml.get("parameters", machine_learning.get("parameters", {})),
            seed=config.seed,
            interval_minutes=interval_minutes,
            baseline_model=(
                baseline_from_spec(baseline_spec, interval_minutes) if residual else None
            ),
            sample_weighting=(
                effective_ml.get("sample_weighting", {})
                if isinstance(effective_ml.get("sample_weighting", {}), Mapping)
                else {}
            ),
            training_window_days=(
                int(effective_ml["training_window_days"])
                if effective_ml.get("training_window_days") is not None
                else None
            ),
        )
        if parameterized:
            return ComponentReconstructionModel(base_model, builder)
        return base_model
    if model_type in {"tcn", "patchtst"}:
        deep = forecast.get("deep_learning", {})
        if not isinstance(deep, Mapping):
            raise ValueError("forecast.deep_learning 必须是字典")
        roles = config.section("data")["roles"]
        builder = CausalFeatureBuilder(
            feature_config=config.section("features"),
            roles=roles,
            interval_minutes=interval_minutes,
            availability=FeatureAvailabilityRegistry.from_config(config),
            model_scope="long",
        )
        return NeuralResidualMultiHorizonModel(
            architecture=model_type,
            feature_builder=builder,
            context_steps=int(deep.get("context_steps", 192)),
            epochs=int(deep.get("epochs", 200)),
            patience=int(deep.get("patience", 20)),
            batch_size=int(deep.get("batch_size", 128)),
            learning_rate=float(deep.get("learning_rate", 1.0e-3)),
            hidden_size=int(deep.get("hidden_size", 64)),
            patch_size=int(deep.get("patch_size", 16)),
            dropout=float(deep.get("dropout", 0.1)),
            seeds=[int(value) for value in deep.get("seeds", [2026, 2027, 2028])],
            device=str(deep.get("device", "auto")),
            interval_minutes=interval_minutes,
        )
    raise ValueError(f"未知模型类型: {model_type}")
