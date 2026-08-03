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
from gas_power.validation import (
    RecentWindowSplitter,
    run_rolling_validation,
    splitter_from_config,
)


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
    name: str = ""
    validation_role: str = "challenger"


def _overall_mape_by_fold(predictions: pd.DataFrame) -> pd.Series:
    values = predictions.copy()
    denominator = np.maximum(np.abs(values["y_true"].to_numpy(dtype=float)), 1.0e-6)
    values["ape"] = np.abs(values["y_pred"] - values["y_true"]) / denominator
    return values.groupby("fold", sort=True)["ape"].mean()


def _evaluate_selection_candidate_gate(
    baseline_recent: pd.DataFrame,
    candidate_recent: pd.DataFrame,
    baseline_cross_month: pd.DataFrame,
    candidate_cross_month: pd.DataFrame,
    settings: Mapping[str, Any],
) -> CandidateGate:
    """要求候选在近期和跨月留出折上分别通过门槛。"""

    recent = evaluate_candidate_gate(
        baseline_recent,
        candidate_recent,
        minimum_mean_gain=float(settings.get("recent_minimum_mean_gain", 0.001)),
        maximum_worst_degradation=float(
            settings.get("recent_maximum_worst_degradation", 0.015)
        ),
        minimum_improved_folds=int(settings.get("recent_minimum_improved_folds", 3)),
    )
    cross_month = evaluate_candidate_gate(
        baseline_cross_month,
        candidate_cross_month,
        minimum_mean_gain=float(
            settings.get("cross_month_minimum_mean_gain", 0.0)
        ),
        maximum_worst_degradation=float(
            settings.get("cross_month_maximum_worst_degradation", 0.015)
        ),
        minimum_improved_folds=int(
            settings.get("cross_month_minimum_improved_folds", 2)
        ),
    )
    target_gains = {
        **{f"recent:{target}": gain for target, gain in recent.target_gains.items()},
        **{
            f"cross_month:{target}": gain
            for target, gain in cross_month.target_gains.items()
        },
    }
    reasons = (
        *(f"近期留出: {reason}" for reason in recent.reasons),
        *(f"跨月留出: {reason}" for reason in cross_month.reasons),
    )
    return CandidateGate(
        passed=recent.passed and cross_month.passed,
        recent_mean_gain=float(
            0.5 * (recent.recent_mean_gain + cross_month.recent_mean_gain)
        ),
        recent_worst_gain=min(
            recent.recent_worst_gain, cross_month.recent_worst_gain
        ),
        improved_folds=recent.improved_folds + cross_month.improved_folds,
        target_gains=target_gains,
        reasons=tuple(reasons),
    )


def _mape_diagnostics(
    predictions: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """记录粗筛候选的逐目标、逐预测列 MAPE，供专家候选选择使用。"""

    values = predictions.copy()
    denominator = np.maximum(np.abs(values["y_true"].to_numpy(dtype=float)), 1.0e-6)
    values["ape"] = np.abs(values["y_pred"] - values["y_true"]) / denominator
    target_scores = {
        str(target): float(score)
        for target, score in values.groupby("target", sort=True)["ape"].mean().items()
    }
    column_scores = {
        f"{target}_h{int(horizon)}": float(score)
        for (target, horizon), score in values.groupby(
            ["target", "horizon_steps"], sort=True
        )["ape"]
        .mean()
        .items()
    }
    return target_scores, column_scores


def select_diverse_trials_for_review(
    trials: Sequence[Any],
    top_k: int,
    preferred_parameterizations: Sequence[str] = (),
) -> list[Any]:
    """兼顾整体最优、目标专家、预测列专家和模型族多样性。"""

    ranked = sorted(
        (trial for trial in trials if trial.value is not None),
        key=lambda item: float(item.value),
    )
    if not ranked or top_k <= 0:
        return []

    selected: list[Any] = []
    selected_numbers: set[int] = set()

    def add(trial: Any) -> None:
        number = int(trial.number)
        if len(selected) < int(top_k) and number not in selected_numbers:
            selected.append(trial)
            selected_numbers.add(number)

    # 整体最优必须保留，避免专家选择牺牲全部列的共同收益。
    add(ranked[0])

    target_names = sorted(
        {
            str(target)
            for trial in ranked
            for target in trial.user_attrs.get("target_mape", {})
        }
    )
    # 先处理当前更难的目标，使有限复核名额优先覆盖短板。
    target_names.sort(
        key=lambda target: float(
            ranked[0].user_attrs.get("target_mape", {}).get(target, 0.0)
        ),
        reverse=True,
    )
    for target in target_names:
        eligible = [
            trial
            for trial in ranked
            if target in trial.user_attrs.get("target_mape", {})
        ]
        if eligible:
            add(
                min(
                    eligible,
                    key=lambda trial: float(trial.user_attrs["target_mape"][target]),
                )
            )

    # 官方思路中的煤气可用量参数化可预留一个复核名额，但仍需通过完整 OOF 门控。
    for parameterization in preferred_parameterizations:
        eligible = [
            trial
            for trial in ranked
            if str(
                trial.user_attrs.get("candidate_config", {}).get("parameterization", "")
            )
            == str(parameterization)
        ]
        if eligible:
            add(min(eligible, key=lambda trial: float(trial.value)))

    column_wins: dict[int, int] = {}
    for column in sorted(
        {
            str(column)
            for trial in ranked
            for column in trial.user_attrs.get("column_mape", {})
        }
    ):
        eligible = [
            trial
            for trial in ranked
            if column in trial.user_attrs.get("column_mape", {})
        ]
        if not eligible:
            continue
        winner = min(
            eligible,
            key=lambda trial: float(trial.user_attrs["column_mape"][column]),
        )
        column_wins[int(winner.number)] = column_wins.get(int(winner.number), 0) + 1
    for trial in sorted(
        ranked,
        key=lambda item: (-column_wins.get(int(item.number), 0), float(item.value)),
    ):
        if column_wins.get(int(trial.number), 0) > 0:
            add(trial)

    family_best: dict[str, Any] = {}
    for trial in ranked:
        descriptor = trial.user_attrs.get("candidate_config", {})
        family = "|".join(
            (
                str(descriptor.get("parameterization", "unknown")),
                str(descriptor.get("strategy", "unknown")),
            )
        )
        family_best.setdefault(family, trial)

    for trial in sorted(family_best.values(), key=lambda item: float(item.value)):
        add(trial)
    for trial in ranked:
        add(trial)
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
    temporal_mixup_ratio: float = 0.0,
    temporal_mixup_min_lambda: float = 0.70,
    temporal_mixup_max_lambda: float = 0.90,
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
                "temporal_mixup_ratio": float(temporal_mixup_ratio),
                "temporal_mixup_min_lambda": float(temporal_mixup_min_lambda),
                "temporal_mixup_max_lambda": float(temporal_mixup_max_lambda),
                "regime_online_threshold_mw": float(
                    config.section("features").get("unit_online_threshold_mw", 1.0)
                ),
                "regime_stable_delta_threshold_mw": float(
                    config.section("features").get(
                        "stable_delta_threshold_mw", 2.0
                    )
                ),
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


def candidate_config_from_descriptor(
    config: ProjectConfig, descriptor: Mapping[str, Any]
) -> ProjectConfig:
    """将可持久化候选描述还原为最终模型配置。"""

    settings = dict(descriptor)
    kind = str(settings.pop("kind", "tree"))
    if kind == "tree":
        return candidate_config_from_settings(config, **settings)
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
    configured_strategy = str(optuna_settings.get("strategy", "direct")).lower()
    if configured_strategy not in {"direct", "global", "mixed"}:
        raise ValueError("optuna.strategy 必须是 direct、global 或 mixed")
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

    def objective(trial: Any) -> float:
        backend = trial.suggest_categorical("backend", ["lightgbm", "catboost"])
        strategy = (
            trial.suggest_categorical("strategy", ["global", "direct"])
            if configured_strategy == "mixed"
            else configured_strategy
        )
        window = trial.suggest_categorical("training_window_days", [30, 60, 90, None])
        half_life = trial.suggest_categorical("half_life_days", [14, 30, 60, None])
        parameterization = trial.suggest_categorical(
            "parameterization", ["direct", "component", "gas_availability"]
        )
        parameter_prefix = (
            "catboost_iterations" if backend == "catboost" else "n_estimators"
        )
        estimator_min = int(optuna_settings.get(f"{parameter_prefix}_min", 300))
        estimator_max = int(optuna_settings.get(f"{parameter_prefix}_max", 1200))
        estimator_step = int(optuna_settings.get(f"{parameter_prefix}_step", 150))
        if strategy == "direct":
            estimator_max = min(
                estimator_max,
                int(
                    optuna_settings.get(f"direct_{parameter_prefix}_max", estimator_max)
                ),
            )
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
            show_progress=show_progress,
            progress_description=f"{trial_label} 粗筛",
        )
        scores = _overall_mape_by_fold(artifacts.predictions)
        target_mape, column_mape = _mape_diagnostics(artifacts.predictions)
        trial.set_user_attr(
            "candidate_config",
            {
                "backend": backend,
                "training_window_days": window,
                "half_life_days": half_life,
                "parameterization": parameterization,
                "strategy": strategy,
                "parameters": parameters,
            },
        )
        trial.set_user_attr("target_mape", target_mape)
        trial.set_user_attr("column_mape", column_mape)
        objective_value = float(0.8 * scores.mean() + 0.2 * scores.max())
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
        "trial_diagnostics_version": 2,
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
    preferred_parameterizations = [
        str(value) for value in optuna_settings.get("preferred_parameterizations", [])
    ]
    for parameterization in preferred_parameterizations:
        already_scheduled = any(
            str(
                trial.params.get(
                    "parameterization",
                    trial.system_attrs.get("fixed_params", {}).get(
                        "parameterization", ""
                    ),
                )
            )
            == parameterization
            for trial in study.trials
        )
        if not already_scheduled:
            study.enqueue_trial({"parameterization": parameterization})
    completed_count = sum(trial.value is not None for trial in study.trials)
    remaining_trials = max(int(n_trials) - completed_count, 0)
    study.optimize(
        objective,
        n_trials=remaining_trials,
        timeout=(
            float(optuna_settings["timeout_seconds"])
            if optuna_settings.get("timeout_seconds") is not None
            else None
        ),
        show_progress_bar=False,
        gc_after_trial=True,
    )
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if len(completed_trials) < int(top_k):
        raise ValueError(
            f"Optuna 完成试验数 {len(completed_trials)} 少于复核候选数 {int(top_k)}"
        )
    best_trials = select_diverse_trials_for_review(
        completed_trials,
        int(top_k),
        preferred_parameterizations=preferred_parameterizations,
    )
    incumbent_trial_numbers = [
        int(value) for value in selection_config.get("incumbent_trial_numbers", [])
    ]
    if incumbent_trial_numbers:
        trials_by_number = {int(trial.number): trial for trial in completed_trials}
        missing_incumbents = [
            number for number in incumbent_trial_numbers if number not in trials_by_number
        ]
        if missing_incumbents:
            raise ValueError(f"固定基线试验不存在: {missing_incumbents}")
        required = [trials_by_number[number] for number in incumbent_trial_numbers]
        selected_numbers = {int(trial.number) for trial in required}
        best_trials = [
            *required,
            *(
                trial
                for trial in best_trials
                if int(trial.number) not in selected_numbers
            ),
        ][: max(int(top_k), len(required))]

    review_specs: list[tuple[int, str, str, dict[str, Any]]] = [
        (
            int(trial.number),
            f"trial_{int(trial.number)}",
            (
                "incumbent"
                if int(trial.number) in set(incumbent_trial_numbers)
                else "challenger"
            ),
            dict(trial.user_attrs["candidate_config"]),
        )
        for trial in best_trials
    ]
    trials_by_number = {int(trial.number): trial for trial in completed_trials}
    fixed_challengers = selection_config.get("fixed_challengers", [])
    if fixed_challengers and not isinstance(fixed_challengers, list):
        raise ValueError("model_selection.fixed_challengers 必须是列表")
    for challenger_index, raw_challenger in enumerate(fixed_challengers, start=1):
        if not isinstance(raw_challenger, Mapping):
            raise ValueError("固定增强候选必须使用字典配置")
        challenger = dict(raw_challenger)
        name = str(challenger.pop("name", f"fixed_{challenger_index}"))
        base_trial_number = int(challenger.pop("base_trial"))
        base_trial = trials_by_number.get(base_trial_number)
        if base_trial is None:
            raise ValueError(f"增强候选基准试验不存在: {base_trial_number}")
        settings = dict(base_trial.user_attrs["candidate_config"])
        settings.update(challenger)
        review_specs.append((base_trial_number, name, "challenger", settings))

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
    baseline_cross = run_rolling_validation(
        frame=frame,
        model_factory=LastValueModel,
        splitter=splitter_from_config(cross_validation),
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=20,
        raw_targets=prepared.raw_targets,
        feature_matrix=cached_features,
        data_source=prepared.source,
        show_progress=show_progress,
        progress_description="跨月份基线验证",
    )
    gate_settings = selection_config.get("candidate_gate", {})
    gate_settings = gate_settings if isinstance(gate_settings, Mapping) else {}
    completed: list[TunedCandidate] = []
    for candidate_index, (trial_number, name, role, settings) in enumerate(
        review_specs, start=1
    ):
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
                f"树模型候选 {candidate_index}/{len(review_specs)} {name} 最近窗口复核"
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
                f"树模型候选 {candidate_index}/{len(review_specs)} {name} 跨月份复核"
            ),
        )
        recent_scores = _overall_mape_by_fold(recent.predictions)
        cross_scores = _overall_mape_by_fold(cross.predictions)
        gate = _evaluate_selection_candidate_gate(
            baseline_recent.predictions,
            recent.predictions,
            baseline_cross.predictions,
            cross.predictions,
            gate_settings,
        )
        # 近期仍占主要权重，但跨月留出不再只是象征性参与排序。
        selection = float(
            0.55 * recent_scores.mean()
            + 0.10 * recent_scores.max()
            + 0.30 * cross_scores.mean()
            + 0.05 * cross_scores.max()
        )
        completed.append(
            TunedCandidate(
                trial_number=trial_number,
                parameters=settings,
                recent_mape=float(recent_scores.mean()),
                recent_worst_mape=float(recent_scores.max()),
                cross_month_mape=float(cross_scores.mean()),
                selection_metric=selection,
                gate=gate,
                recent_predictions=recent.predictions,
                cross_month_predictions=cross.predictions,
                name=name,
                validation_role=role,
            )
        )
    return sorted(completed, key=lambda item: item.selection_metric), study


def _trial_parameters_with_backend(
    trial: Any,
    backend: str,
    *,
    estimator_min: int = 300,
    estimator_max: int = 1200,
    estimator_step: int = 150,
) -> dict[str, Any]:
    depth_max = 6 if backend == "catboost" else 10
    depth_parameter = (
        "catboost_max_depth" if backend == "catboost" else "lightgbm_max_depth"
    )
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
                "loss_function": trial.suggest_categorical(
                    "loss_function", ["MAE", "Huber:delta=1.0"]
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1.0e-3, 10.0, log=True
                ),
                "thread_count": -1,
                "allow_writing_files": False,
            }
        )
    return common
