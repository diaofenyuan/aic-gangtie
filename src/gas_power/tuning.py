"""训练期 Optuna 搜索与完整时间折复核。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gas_power.config import ProjectConfig
from gas_power.data import PreparedForecastData
from gas_power.ensemble_selection import CandidateGate, evaluate_candidate_gate
from gas_power.features import CausalFeatureBuilder
from gas_power.models.baselines import LastValueModel
from gas_power.models.factory import build_model
from gas_power.validation import RecentWindowSplitter, run_rolling_validation, splitter_from_config


@dataclass
class TunedCandidate:
    trial_number: int
    parameters: dict[str, Any]
    recent_mape: float
    recent_worst_mape: float
    cross_month_mape: float
    selection_metric: float
    gate: CandidateGate
    recent_predictions: pd.DataFrame
    cross_month_predictions: pd.DataFrame


def _overall_mape_by_fold(predictions: pd.DataFrame) -> pd.Series:
    values = predictions.copy()
    denominator = np.maximum(np.abs(values["y_true"].to_numpy(dtype=float)), 1.0e-6)
    values["ape"] = np.abs(values["y_pred"] - values["y_true"]) / denominator
    return values.groupby("fold", sort=True)["ape"].mean()


def candidate_config_from_settings(
    config: ProjectConfig,
    *,
    backend: str,
    training_window_days: int | None,
    half_life_days: int | None,
    parameters: Mapping[str, Any],
    parameterization: str,
) -> ProjectConfig:
    raw = deepcopy(config.raw)
    raw["forecast"]["model"] = {
        "type": (
            "component_reconstruction"
            if parameterization == "component"
            else f"residual_{backend}"
        )
    }
    raw["forecast"]["machine_learning"]["backend"] = backend
    raw["forecast"]["machine_learning"]["strategy"] = "direct"
    raw["forecast"]["machine_learning"]["target_mode"] = "residual"
    effective = raw["residual_model"]
    effective.update(
        {
            "enabled": True,
            "backend": backend,
            "strategy": "direct",
            "baseline": {"type": "last_value"},
            "parameters": dict(parameters),
            "training_window_days": training_window_days,
            "sample_weighting": {
                "mape": True,
                "floor_quantile": 0.01,
                "minimum_floor": 1.0e-6,
                "recency_half_life_days": half_life_days,
            },
        }
    )
    if parameterization == "component":
        raw["forecast"]["machine_learning"].update(
            {
                "parameters": dict(parameters),
                "training_window_days": training_window_days,
                "sample_weighting": effective["sample_weighting"],
                "target_mode": "delta",
            }
        )
    return ProjectConfig(raw=raw, source=config.source, root=config.root)


def deep_config_from_settings(
    config: ProjectConfig,
    *,
    architecture: str,
    context_steps: int,
    epochs: int | None = None,
) -> ProjectConfig:
    """构造仅使用官方训练数据、从零训练的深度候选配置。"""

    if architecture not in {"tcn", "patchtst"}:
        raise ValueError("深度候选架构必须是 tcn 或 patchtst")
    raw = deepcopy(config.raw)
    raw["forecast"]["model"] = {"type": architecture}
    deep = raw["forecast"].setdefault("deep_learning", {})
    deep["context_steps"] = int(context_steps)
    if epochs is not None:
        deep["epochs"] = int(epochs)
    return ProjectConfig(raw=raw, source=config.source, root=config.root)


def candidate_config_from_descriptor(
    config: ProjectConfig, descriptor: Mapping[str, Any]
) -> ProjectConfig:
    """将可持久化候选描述还原为最终模型配置。"""

    settings = dict(descriptor)
    kind = str(settings.pop("kind", "tree"))
    if kind == "tree":
        return candidate_config_from_settings(config, **settings)
    if kind == "deep":
        return deep_config_from_settings(config, **settings)
    raise ValueError(f"未知候选类型: {kind}")


def run_tree_search(
    config: ProjectConfig,
    prepared: PreparedForecastData,
    feature_builder: CausalFeatureBuilder,
    feature_matrix: pd.DataFrame | None = None,
    *,
    n_trials: int = 30,
    top_k: int = 5,
    show_progress: bool = False,
) -> tuple[list[TunedCandidate], Any]:
    """先在最近四折粗筛，再对前五组参数执行十折和跨月份复核。"""

    prepared.assert_training_allowed()
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("运行高精度参数搜索需要安装 optuna 可选依赖") from exc

    frame = prepared.model_input
    cached_features = (
        feature_builder.transform(frame)
        if feature_matrix is None
        else feature_matrix.reindex(frame.index)
    )
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    horizons = list(range(1, int(config.section("forecast")["short_steps"]) + 1))
    validation = config.section("validation")
    recent_config = config.raw.get("recent_validation", {})
    recent_config = recent_config if isinstance(recent_config, Mapping) else {}
    coarse_splitter = RecentWindowSplitter(
        folds=4,
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )

    def objective(trial: Any) -> float:
        backend = trial.suggest_categorical("backend", ["lightgbm", "catboost"])
        window = trial.suggest_categorical("training_window_days", [30, 60, 90, None])
        half_life = trial.suggest_categorical("half_life_days", [14, 30, 60, None])
        parameterization = trial.suggest_categorical("parameterization", ["direct", "component"])
        optuna_settings = config.raw.get("optuna", {})
        optuna_settings = optuna_settings if isinstance(optuna_settings, Mapping) else {}
        estimator_min = int(optuna_settings.get("n_estimators_min", 300))
        estimator_max = int(optuna_settings.get("n_estimators_max", 1200))
        estimator_step = int(optuna_settings.get("n_estimators_step", 150))
        parameters = _trial_parameters_with_backend(
            trial,
            backend,
            estimator_min=estimator_min,
            estimator_max=estimator_max,
            estimator_step=estimator_step,
        )
        if show_progress:
            tqdm.write(
                f"开始调参 trial {trial.number + 1}/{int(n_trials)}："
                f"{backend}，{parameterization}，{parameters['n_estimators' if backend == 'lightgbm' else 'iterations']} 棵树"
            )
        candidate_config = candidate_config_from_settings(
            config,
            backend=backend,
            training_window_days=window,
            half_life_days=half_life,
            parameters=parameters,
            parameterization=parameterization,
        )
        artifacts = run_rolling_validation(
            frame=frame,
            model_factory=lambda item=candidate_config: build_model(item),
            splitter=coarse_splitter,
            target_columns=targets,
            horizons=horizons,
            interval_minutes=15,
            near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
            worst_error_count=20,
            raw_targets=prepared.raw_targets,
            feature_matrix=cached_features,
            data_source=prepared.source,
            show_progress=show_progress,
            progress_description=f"调参 trial {trial.number + 1}/{int(n_trials)} 粗筛",
        )
        scores = _overall_mape_by_fold(artifacts.predictions)
        trial.set_user_attr("candidate_config", {
            "backend": backend,
            "training_window_days": window,
            "half_life_days": half_life,
            "parameterization": parameterization,
            "parameters": parameters,
        })
        return float(0.8 * scores.mean() + 0.2 * scores.max())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=show_progress)
    best_trials = sorted(study.trials, key=lambda item: float(item.value))[: int(top_k)]

    recent_splitter = RecentWindowSplitter(
        folds=int(recent_config.get("folds", 10)),
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )
    baseline_recent = run_rolling_validation(
        frame=frame,
        model_factory=LastValueModel,
        splitter=recent_splitter,
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=20,
        raw_targets=prepared.raw_targets,
        feature_matrix=cached_features,
        data_source=prepared.source,
        show_progress=show_progress,
        progress_description="最近窗口基线验证",
    )
    completed: list[TunedCandidate] = []
    for candidate_index, trial in enumerate(best_trials, start=1):
        settings = dict(trial.user_attrs["candidate_config"])
        candidate_config = candidate_config_from_descriptor(config, settings)
        recent = run_rolling_validation(
            frame=frame,
            model_factory=lambda item=candidate_config: build_model(item),
            splitter=recent_splitter,
            target_columns=targets,
            horizons=horizons,
            interval_minutes=15,
            near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
            worst_error_count=50,
            raw_targets=prepared.raw_targets,
            feature_matrix=cached_features,
            data_source=prepared.source,
            show_progress=show_progress,
            progress_description=(
                f"树模型候选 {candidate_index}/{len(best_trials)} 最近窗口复核"
            ),
        )
        cross = run_rolling_validation(
            frame=frame,
            model_factory=lambda: build_model(candidate_config),
            splitter=splitter_from_config(validation),
            target_columns=targets,
            horizons=horizons,
            interval_minutes=15,
            near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
            worst_error_count=50,
            raw_targets=prepared.raw_targets,
            feature_matrix=cached_features,
            data_source=prepared.source,
            show_progress=show_progress,
            progress_description=(
                f"树模型候选 {candidate_index}/{len(best_trials)} 跨月份复核"
            ),
        )
        recent_scores = _overall_mape_by_fold(recent.predictions)
        cross_scores = _overall_mape_by_fold(cross.predictions)
        gate = evaluate_candidate_gate(baseline_recent.predictions, recent.predictions)
        selection = float(
            0.7 * recent_scores.mean()
            + 0.2 * recent_scores.max()
            + 0.1 * cross_scores.mean()
        )
        completed.append(
            TunedCandidate(
                trial_number=int(trial.number),
                parameters=settings,
                recent_mape=float(recent_scores.mean()),
                recent_worst_mape=float(recent_scores.max()),
                cross_month_mape=float(cross_scores.mean()),
                selection_metric=selection,
                gate=gate,
                recent_predictions=recent.predictions,
                cross_month_predictions=cross.predictions,
            )
        )
    return sorted(completed, key=lambda item: item.selection_metric), study


def run_deep_search(
    config: ProjectConfig,
    prepared: PreparedForecastData,
    feature_builder: CausalFeatureBuilder,
    feature_matrix: pd.DataFrame | None = None,
    *,
    coarse_folds: int = 4,
    top_k: int = 2,
    coarse_epochs: int = 60,
    show_progress: bool = False,
) -> list[TunedCandidate]:
    """粗筛 TCN/PatchTST 上下文，并对优胜候选执行完整稳定性复核。"""

    prepared.assert_training_allowed()
    try:
        __import__("torch")
    except ImportError as exc:
        raise ImportError("运行深度候选搜索需要安装 PyTorch 可选依赖") from exc
    frame = prepared.model_input
    cached_features = (
        feature_builder.transform(frame)
        if feature_matrix is None
        else feature_matrix.reindex(frame.index)
    )
    forecast = config.section("forecast")
    deep = forecast.get("deep_learning", {})
    deep = deep if isinstance(deep, Mapping) else {}
    contexts = [int(value) for value in deep.get("context_candidates", [192, 384, 672])]
    architectures = [str(value).lower() for value in deep.get("architectures", ["tcn", "patchtst"])]
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    horizons = list(range(1, int(forecast["short_steps"]) + 1))
    validation = config.section("validation")
    recent_config = config.raw.get("recent_validation", {})
    recent_config = recent_config if isinstance(recent_config, Mapping) else {}
    coarse_splitter = RecentWindowSplitter(
        folds=int(coarse_folds),
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )
    coarse_results: list[tuple[float, int, dict[str, Any]]] = []
    candidate_index = 0
    candidate_total = len(architectures) * len(contexts)
    for architecture in architectures:
        for context_steps in contexts:
            candidate_index += 1
            if show_progress:
                tqdm.write(
                    f"开始深度候选 {candidate_index}/{candidate_total}："
                    f"{architecture}，上下文 {context_steps} 步"
                )
            descriptor = {
                "kind": "deep",
                "architecture": architecture,
                "context_steps": context_steps,
            }
            coarse_config = deep_config_from_settings(
                config,
                architecture=architecture,
                context_steps=context_steps,
                epochs=min(int(deep.get("epochs", 200)), int(coarse_epochs)),
            )
            artifacts = run_rolling_validation(
                frame=frame,
                model_factory=lambda item=coarse_config: build_model(item),
                splitter=coarse_splitter,
                target_columns=targets,
                horizons=horizons,
                interval_minutes=15,
                near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
                worst_error_count=20,
                raw_targets=prepared.raw_targets,
                feature_matrix=cached_features,
                data_source=prepared.source,
                show_progress=show_progress,
                progress_description=(
                    f"深度候选 {candidate_index}/{candidate_total} 粗筛"
                ),
            )
            scores = _overall_mape_by_fold(artifacts.predictions)
            coarse_results.append(
                (
                    float(0.8 * scores.mean() + 0.2 * scores.max()),
                    -candidate_index,
                    descriptor,
                )
            )

    recent_splitter = RecentWindowSplitter(
        folds=int(recent_config.get("folds", 10)),
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )
    baseline_recent = run_rolling_validation(
        frame=frame,
        model_factory=LastValueModel,
        splitter=recent_splitter,
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=20,
        raw_targets=prepared.raw_targets,
        feature_matrix=cached_features,
        data_source=prepared.source,
        show_progress=show_progress,
        progress_description="深度候选最近窗口基线验证",
    )
    completed: list[TunedCandidate] = []
    selected_deep = sorted(coarse_results)[: int(top_k)]
    for candidate_index, (_, trial_number, descriptor) in enumerate(
        selected_deep, start=1
    ):
        candidate_config = candidate_config_from_descriptor(config, descriptor)
        recent = run_rolling_validation(
            frame=frame,
            model_factory=lambda item=candidate_config: build_model(item),
            splitter=recent_splitter,
            target_columns=targets,
            horizons=horizons,
            interval_minutes=15,
            near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
            worst_error_count=50,
            raw_targets=prepared.raw_targets,
            feature_matrix=cached_features,
            data_source=prepared.source,
            show_progress=show_progress,
            progress_description=(
                f"深度候选 {candidate_index}/{len(selected_deep)} 最近窗口复核"
            ),
        )
        cross = run_rolling_validation(
            frame=frame,
            model_factory=lambda item=candidate_config: build_model(item),
            splitter=splitter_from_config(validation),
            target_columns=targets,
            horizons=horizons,
            interval_minutes=15,
            near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
            worst_error_count=50,
            raw_targets=prepared.raw_targets,
            feature_matrix=cached_features,
            data_source=prepared.source,
            show_progress=show_progress,
            progress_description=(
                f"深度候选 {candidate_index}/{len(selected_deep)} 跨月份复核"
            ),
        )
        recent_scores = _overall_mape_by_fold(recent.predictions)
        cross_scores = _overall_mape_by_fold(cross.predictions)
        gate = evaluate_candidate_gate(baseline_recent.predictions, recent.predictions)
        completed.append(
            TunedCandidate(
                trial_number=trial_number,
                parameters=descriptor,
                recent_mape=float(recent_scores.mean()),
                recent_worst_mape=float(recent_scores.max()),
                cross_month_mape=float(cross_scores.mean()),
                selection_metric=float(
                    0.7 * recent_scores.mean()
                    + 0.2 * recent_scores.max()
                    + 0.1 * cross_scores.mean()
                ),
                gate=gate,
                recent_predictions=recent.predictions,
                cross_month_predictions=cross.predictions,
            )
        )
    return sorted(completed, key=lambda item: item.selection_metric)


def _trial_parameters_with_backend(
    trial: Any,
    backend: str,
    *,
    estimator_min: int = 300,
    estimator_max: int = 1200,
    estimator_step: int = 150,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
    }
    if estimator_min <= 0 or estimator_max < estimator_min or estimator_step <= 0:
        raise ValueError("Optuna 树数范围必须满足 min>0、max>=min、step>0")
    estimators = trial.suggest_int(
        "n_estimators", estimator_min, estimator_max, step=estimator_step
    )
    if backend == "lightgbm":
        common.update(
            {
                "n_estimators": estimators,
                "objective": "regression_l1",
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0e-3, 10.0, log=True),
                "n_jobs": -1,
            }
        )
    else:
        common.update(
            {
                "iterations": estimators,
                "loss_function": trial.suggest_categorical("loss_function", ["MAE", "Huber:delta=1.0"]),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0e-3, 10.0, log=True),
                "thread_count": -1,
                "allow_writing_files": False,
            }
        )
    return common
