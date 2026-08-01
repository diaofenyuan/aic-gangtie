"""训练期 Optuna 搜索与完整时间折复核。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_power.config import ProjectConfig
from gas_power.data import PreparedForecastData
from gas_power.ensemble_selection import CandidateGate, evaluate_candidate_gate
from gas_power.features import CausalFeatureBuilder
from gas_power.models.baselines import LastValueModel
from gas_power.models.factory import build_model
from gas_power.runtime import progress_bar
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


def select_diverse_trials_for_review(
    trials: Sequence[Any],
    top_k: int,
) -> list[Any]:
    """优先保留每种参数化的最佳试验，再按总排名补足复核名额。"""

    ranked = sorted(
        (trial for trial in trials if trial.value is not None),
        key=lambda item: float(item.value),
    )
    family_best: dict[str, Any] = {}
    for trial in ranked:
        descriptor = trial.user_attrs.get("candidate_config", {})
        family = str(descriptor.get("parameterization", "unknown"))
        family_best.setdefault(family, trial)

    selected = sorted(family_best.values(), key=lambda item: float(item.value))[:top_k]
    selected_numbers = {int(trial.number) for trial in selected}
    for trial in ranked:
        if len(selected) >= top_k:
            break
        if int(trial.number) not in selected_numbers:
            selected.append(trial)
            selected_numbers.add(int(trial.number))
    return sorted(selected, key=lambda item: float(item.value))


def candidate_config_from_settings(
    config: ProjectConfig,
    *,
    backend: str,
    training_window_days: int | None,
    half_life_days: int | None,
    parameters: Mapping[str, Any],
    parameterization: str,
    strategy: str = "direct",
) -> ProjectConfig:
    if strategy not in {"direct", "global"}:
        raise ValueError("树模型多步策略必须是 direct 或 global")
    if parameterization not in {"direct", "component", "gas_availability"}:
        raise ValueError("树模型参数化必须是 direct、component 或 gas_availability")
    raw = deepcopy(config.raw)
    raw["forecast"]["model"] = {
        "type": (
            "component_reconstruction"
            if parameterization == "component"
            else "gas_availability"
            if parameterization == "gas_availability"
            else f"residual_{backend}"
        )
    }
    raw["forecast"]["machine_learning"]["backend"] = backend
    raw["forecast"]["machine_learning"]["strategy"] = strategy
    raw["forecast"]["machine_learning"]["target_mode"] = "residual"
    effective = raw["residual_model"]
    effective.update(
        {
            "enabled": True,
            "backend": backend,
            "strategy": strategy,
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
    coarse_folds: int = 4,
    show_progress: bool = False,
) -> tuple[list[TunedCandidate], Any]:
    """按配置预算粗筛树模型，再对排名靠前的候选执行完整复核。"""

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
    selection_config = config.raw.get("model_selection", {})
    selection_config = selection_config if isinstance(selection_config, Mapping) else {}
    optuna_settings = config.raw.get("optuna", {})
    optuna_settings = optuna_settings if isinstance(optuna_settings, Mapping) else {}
    strategy = str(optuna_settings.get("strategy", "direct")).lower()
    if strategy not in {"direct", "global"}:
        raise ValueError("optuna.strategy 必须是 direct 或 global")
    if min(int(n_trials), int(top_k), int(coarse_folds)) <= 0:
        raise ValueError("Optuna 的 n_trials、top_k 和 coarse_folds 必须大于 0")
    configured_startup_trials = int(
        optuna_settings.get("n_startup_trials", min(10, int(n_trials)))
    )
    if configured_startup_trials < 0:
        raise ValueError("optuna.n_startup_trials 不得小于 0")
    startup_trials = min(configured_startup_trials, int(n_trials))
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

    search_progress = None

    def update_search_status(status: str) -> None:
        if search_progress is not None:
            search_progress.set_postfix_str(status, refresh=True)

    def objective(trial: Any) -> float:
        backend = trial.suggest_categorical("backend", ["lightgbm", "catboost"])
        window = trial.suggest_categorical("training_window_days", [30, 60, 90, None])
        half_life = trial.suggest_categorical("half_life_days", [14, 30, 60, None])
        parameterization = trial.suggest_categorical(
            "parameterization", ["direct", "component", "gas_availability"]
        )
        parameter_prefix = "catboost_iterations" if backend == "catboost" else "n_estimators"
        estimator_min = int(optuna_settings.get(f"{parameter_prefix}_min", 300))
        estimator_max = int(optuna_settings.get(f"{parameter_prefix}_max", 1200))
        estimator_step = int(optuna_settings.get(f"{parameter_prefix}_step", 150))
        parameters = _trial_parameters_with_backend(
            trial,
            backend,
            estimator_min=estimator_min,
            estimator_max=estimator_max,
            estimator_step=estimator_step,
        )
        estimator_count = parameters[
            "n_estimators" if backend == "lightgbm" else "iterations"
        ]
        trial_label = (
            f"trial {trial.number + 1}/{int(n_trials)} | {backend} | "
            f"{parameterization} | {strategy} 多步 | {estimator_count} 棵树"
        )
        update_search_status(trial_label)
        candidate_config = candidate_config_from_settings(
            config,
            backend=backend,
            training_window_days=window,
            half_life_days=half_life,
            parameters=parameters,
            parameterization=parameterization,
            strategy=strategy,
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
            show_progress=False,
            progress_description=f"调参 trial {trial.number + 1}/{int(n_trials)} 粗筛",
            progress_callback=lambda status: update_search_status(
                f"{trial_label} | {status}"
            ),
        )
        scores = _overall_mape_by_fold(artifacts.predictions)
        trial.set_user_attr("candidate_config", {
            "backend": backend,
            "training_window_days": window,
            "half_life_days": half_life,
            "parameterization": parameterization,
            "strategy": strategy,
            "parameters": parameters,
        })
        objective_value = float(0.8 * scores.mean() + 0.2 * scores.max())
        update_search_status(f"{trial_label} | MAPE {float(scores.mean()):.6f}")
        if search_progress is not None:
            search_progress.update(1)
        return objective_value

    study_signature = {
        "source": config.source.name,
        "seed": config.seed,
        "targets": targets,
        "horizons": horizons,
        "features": config.section("features"),
        "validation": validation,
        "recent_validation": recent_config,
        "optuna": optuna_settings,
        "residual_model": config.raw.get("residual_model", {}),
        "target_hash": hashlib.sha256(
            pd.util.hash_pandas_object(frame[targets], index=True).values.tobytes()
        ).hexdigest(),
    }
    study_digest = hashlib.sha256(
        json.dumps(
            study_signature,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    study_name = f"{config.source.stem}-{study_digest}"
    storage_path = config.path("cache") / "optuna" / f"{config.source.stem}.sqlite3"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=config.seed,
            n_startup_trials=startup_trials,
        ),
        storage=f"sqlite:///{storage_path.as_posix()}",
        study_name=study_name,
        load_if_exists=True,
    )
    completed_count = sum(trial.value is not None for trial in study.trials)
    remaining_trials = max(int(n_trials) - completed_count, 0)
    search_progress = progress_bar(
        total=int(n_trials),
        initial=completed_count,
        desc="树模型搜索",
        unit="轮",
        leave=True,
        disable=not show_progress,
    )
    search_progress.set_postfix_str(
        f"已恢复 {completed_count} 轮 | 待运行 {remaining_trials} 轮",
        refresh=True,
    )
    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=False,
            gc_after_trial=True,
        )
    finally:
        search_progress.close()
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if len(completed_trials) < int(top_k):
        raise ValueError(
            f"Optuna 完成试验数 {len(completed_trials)} 少于复核候选数 {int(top_k)}"
        )
    best_trials = select_diverse_trials_for_review(completed_trials, int(top_k))

    recent_splitter = RecentWindowSplitter(
        folds=int(selection_config.get("recent_folds", recent_config.get("folds", 10))),
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )
    cross_validation = dict(validation)
    cross_validation["folds"] = int(
        selection_config.get("cross_month_folds", validation.get("folds", 8))
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
            splitter=splitter_from_config(cross_validation),
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
    depth_max = 6 if backend == "catboost" else 10
    depth_parameter = "catboost_max_depth" if backend == "catboost" else "lightgbm_max_depth"
    common: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": trial.suggest_int(depth_parameter, 4, depth_max),
    }
    if estimator_min <= 0 or estimator_max < estimator_min or estimator_step <= 0:
        raise ValueError("Optuna 树数范围必须满足 min>0、max>=min、step>0")
    estimator_parameter = (
        "catboost_iterations" if backend == "catboost" else "lightgbm_n_estimators"
    )
    estimators = trial.suggest_int(
        estimator_parameter, estimator_min, estimator_max, step=estimator_step
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
