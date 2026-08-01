"""数据、预测、验证与优化的可复用命令流水线。"""

from __future__ import annotations

import json
import hashlib
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from gas_power.audit import calculate_lag_correlations, run_data_audit
from gas_power.availability import FeatureAvailabilityRegistry
from gas_power.benchmark import run_baseline_benchmark
from gas_power.config import ProjectConfig, configured_value
from gas_power.data import (
    ConfiguredDataLoader,
    PreparedForecastData,
    inspect_submission_input_quality,
    normalize_submission_input_frame,
    load_original_input_frame,
    prepare_submission_sources,
    prepare_scoring_with_history,
    sanitize_submission_features,
    validate_preliminary_input_frame,
    write_time_frame,
)
from gas_power.features import (
    CausalFeatureBuilder,
    assert_feature_causality,
    assert_shift_before_rolling,
)
from gas_power.environment import check_high_accuracy_environment
from gas_power.gpu_gate import evaluate_gpu_gate, evaluate_residual_gate
from gas_power.models import build_model
from gas_power.models.base import prediction_column, prediction_columns
from gas_power.models.baselines import DampedTrendModel, LastValueModel
from gas_power.models.factory import baseline_from_spec
from gas_power.models.ensemble import HorizonWeightedEnsembleModel
from gas_power.ensemble_selection import (
    apply_oof_weights,
    evaluate_oof_column_gate,
    fit_oof_weights,
)
from gas_power.optimization import (
    DispatchInput,
    HighsDispatchOptimizer,
    OptimizationError,
)
from gas_power.outputs import write_forecast_csv, write_optimization_csv
from gas_power.postprocessing import (
    PhysicalForecastPostprocessor,
    compare_postprocessing_metrics,
)
from gas_power.relations import discover_relations
from gas_power.runtime import progress_bar, progress_enabled, track_progress
from gas_power.scoring import ConfigurableScorer
from gas_power.submission import validate_submission_bundle
from gas_power.submission_freeze import freeze_submission
from gas_power.synthetic import SYNTHETIC_WARNING, generate_synthetic_dataset
from gas_power.validation import (
    RecentWindowSplitter,
    run_rolling_validation,
    splitter_from_config,
    summarize_detailed_validation,
)
from gas_power.tuning import (
    candidate_config_from_descriptor,
    run_deep_search,
    run_tree_search,
)

OFFICIAL_PRELIMINARY_DATA_NOTICE = (
    "当前仅使用赛事方提供的初赛参赛数据；评分集不得进入训练、调参、"
    "特征构造或模型选择，赛事数据不得传播或公开展示。"
)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)


def _selection_fold_subset(
    predictions: pd.DataFrame,
    *,
    validation_type: str,
    fold_count: int,
) -> pd.DataFrame:
    """均匀抽取指定数量时间折，并为近期/跨月折建立唯一标识。"""

    folds = sorted(predictions["fold"].dropna().unique().tolist())
    if not folds:
        raise ValueError(f"{validation_type} 验证没有可用时间折")
    count = min(int(fold_count), len(folds))
    positions = np.linspace(0, len(folds) - 1, num=count, dtype=int)
    selected = [folds[int(position)] for position in positions]
    output = predictions.loc[predictions["fold"].isin(selected)].copy()
    output["validation_type"] = str(validation_type)
    output["fold"] = output["fold"].map(lambda value: f"{validation_type}_{value}")
    return output


def _combine_selection_predictions(
    recent: pd.DataFrame,
    cross_month: pd.DataFrame,
    *,
    recent_folds: int,
    cross_month_folds: int,
) -> pd.DataFrame:
    return pd.concat(
        [
            _selection_fold_subset(
                recent,
                validation_type="recent",
                fold_count=recent_folds,
            ),
            _selection_fold_subset(
                cross_month,
                validation_type="cross_month",
                fold_count=cross_month_folds,
            ),
        ],
        ignore_index=True,
    )


def _assert_selection_fold_coverage(
    predictions: pd.DataFrame,
    *,
    recent_folds: int,
    cross_month_folds: int,
    context: str,
) -> None:
    """确保逐列门控使用完整且对齐的近期折和跨月折。"""

    actual = set(predictions["fold"].dropna().astype(str).unique())
    expected = {
        *(f"recent_{fold}" for fold in range(int(recent_folds))),
        *(f"cross_month_{fold}" for fold in range(int(cross_month_folds))),
    }
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            f"{context} 的模型选择折未完整对齐："
            f"缺少 {missing}，多出 {unexpected}"
        )


def _large_error_reports(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """输出前 5% 大误差样本及按工况聚合的诊断表。"""

    values = predictions.copy()
    denominator = np.maximum(np.abs(values["y_true"].to_numpy(dtype=float)), 1.0e-6)
    values["ape"] = np.abs(values["y_pred"] - values["y_true"]) / denominator
    values["origin_hour"] = pd.to_datetime(values["origin"]).dt.hour
    values["regime"] = "stable"
    for target in values["target"].astype(str).unique():
        target_mask = values["target"].astype(str) == target
        for suffix in ("ramp_up", "ramp_down", "startup", "shutdown"):
            column = f"feat_state_{target}_{suffix}"
            if column in values:
                mask = target_mask & (values[column].fillna(0.0) > 0.5)
                values.loc[mask, "regime"] = suffix
    missing_columns = [
        column for column in values if str(column).startswith("feat_missing__")
    ]
    outlier_columns = [
        column for column in values if str(column).startswith("feat_outlier__")
    ]
    values["has_missing"] = (
        values[missing_columns].fillna(0.0).sum(axis=1) > 0 if missing_columns else False
    )
    values["has_outlier"] = (
        values[outlier_columns].fillna(0.0).sum(axis=1) > 0 if outlier_columns else False
    )
    values["has_gas_gap"] = (
        values["feat_gas_total_supply_gap"].fillna(0.0) > 0.0
        if "feat_gas_total_supply_gap" in values
        else False
    )
    thresholds = values.groupby(["target", "horizon_steps"])["ape"].transform(
        lambda series: series.quantile(0.95)
    )
    largest = values.loc[values["ape"] >= thresholds].sort_values("ape", ascending=False)
    group_columns = [
        "target",
        "horizon_steps",
        "origin_hour",
        "regime",
        "has_missing",
        "has_outlier",
        "has_gas_gap",
    ]
    groups = values.groupby(group_columns, dropna=False).agg(
        samples=("ape", "size"),
        mean_mape=("ape", "mean"),
        p95_mape=("ape", lambda series: series.quantile(0.95)),
    ).reset_index()
    groups = groups.sort_values(["mean_mape", "samples"], ascending=[False, False])
    return largest, groups


def _data_notice(config: ProjectConfig) -> str:
    """根据数据配置返回真实且可审计的数据来源说明。"""

    compliance = config.raw.get("competition_compliance")
    if (
        isinstance(compliance, Mapping)
        and compliance.get("stage") == "preliminary"
        and compliance.get("official_data_only") is True
    ):
        return OFFICIAL_PRELIMINARY_DATA_NOTICE
    return SYNTHETIC_WARNING


def _feature_builder(config: ProjectConfig) -> CausalFeatureBuilder:
    return CausalFeatureBuilder(
        feature_config=config.section("features"),
        roles=config.section("data")["roles"],
        interval_minutes=int(config.section("optimization").get("interval_minutes", 15)),
        availability=FeatureAvailabilityRegistry.from_config(config),
        model_scope="long",
    )


def prepare_data(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared, features, quality = prepare_prepared_data(config)
    return prepared.model_input, features, quality


def prepare_prepared_data(
    config: ProjectConfig,
) -> tuple[PreparedForecastData, pd.DataFrame, dict[str, Any]]:
    config.ensure_runtime_dirs()
    loader = ConfiguredDataLoader(config)
    builder = _feature_builder(config)
    cache_steps = 3
    with progress_bar(
        total=loader.progress_steps() + builder.progress_steps() + cache_steps,
        desc="数据准备",
        unit="项",
        dynamic_ncols=True,
        leave=True,
        disable=not progress_enabled(config),
        mininterval=0.2,
    ) as data_progress:
        def advance(label: str) -> None:
            data_progress.set_postfix_str(f"完成: {label}", refresh=False)
            data_progress.update(1)

        data_progress.set_postfix_str("正在读取数据", refresh=True)
        prepared, quality = loader.load_prepared(progress_callback=advance, source="training")
        frame = prepared.model_input
        data_progress.set_postfix_str("正在构建特征", refresh=True)
        features = builder.transform(frame, progress_callback=advance)
        cache_dir = config.path("cache")
        timestamp_format = str(config.section("data")["timestamp_format"])
        data_progress.set_postfix_str("正在写入 processed.csv", refresh=True)
        write_time_frame(frame, cache_dir / "processed.csv", timestamp_format)
        advance("写入 processed.csv")
        data_progress.set_postfix_str("正在写入 features.csv", refresh=True)
        write_time_frame(features, cache_dir / "features.csv", timestamp_format)
        advance("写入 features.csv")
        quality_path = cache_dir / "data_quality.json"
        data_progress.set_postfix_str("正在写入 data_quality.json", refresh=True)
        quality.write_json(quality_path)
        advance("写入 data_quality.json")
    return prepared, features, quality.to_dict()


def generate_synthetic_pipeline(config: ProjectConfig) -> dict[str, Any]:
    config.ensure_runtime_dirs()
    return generate_synthetic_dataset(config)


def environment_pipeline(config: ProjectConfig) -> dict[str, Any]:
    report = check_high_accuracy_environment()
    _write_json(report, config.path("reports", "reports") / "environment.json")
    return report


def _is_preliminary(config: ProjectConfig) -> bool:
    compliance = config.raw.get("competition_compliance")
    return (
        isinstance(compliance, Mapping)
        and compliance.get("stage") == "preliminary"
    )


def _model_horizons(config: ProjectConfig) -> list[int]:
    """初赛模型只训练和验证短周期，避免无关的 96 步任务稀释目标。"""

    forecast = config.section("forecast")
    steps = int(forecast["short_steps"] if _is_preliminary(config) else forecast["long_steps"])
    return list(range(1, steps + 1))


def train_pipeline(config: ProjectConfig) -> dict[str, Any]:
    prepared, features, quality = prepare_prepared_data(config)
    prepared.assert_training_allowed()
    frame = prepared.model_input
    roles = config.section("data")["roles"]
    targets = [str(value) for value in roles["targets"]]
    horizons = _model_horizons(config)
    _assert_final_residual_gate(config)
    model = build_model(config)
    train_end = pd.Timestamp(frame.index.max())
    model.fit(
        frame,
        targets,
        horizons,
        train_end=train_end,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
    )
    model_path = config.path("models") / "forecast_model.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "model_type": type(model).__name__,
        "train_start": str(frame.index.min()),
        "train_end": str(train_end),
        "training_rows": len(frame),
        "targets": targets,
        "horizons": [horizons[0], horizons[-1]],
        "decision_inputs": "TRAINING_VALIDATION_ONLY",
        "seed": config.seed,
        "data_notice": _data_notice(config),
        "quality_report": quality,
        "feature_columns": list(features.columns),
        "raw_label_columns": targets,
        "training_data_source": prepared.source,
        "feature_source_whitelist": sorted(
            FeatureAvailabilityRegistry.from_config(config).allowed_source_columns("long")
        ),
    }
    _write_json(metadata, config.path("models") / "forecast_model_metadata.json")
    return metadata


def tune_pipeline(config: ProjectConfig) -> dict[str, Any]:
    """执行训练期粗筛、完整复核、OOF 融合，并冻结唯一可训练候选。"""

    prepared, features, _ = prepare_prepared_data(config)
    prepared.assert_training_allowed()
    settings = config.raw.get("optuna", {})
    settings = settings if isinstance(settings, Mapping) else {}
    tree_candidates, study = run_tree_search(
        config,
        prepared,
        _feature_builder(config),
        feature_matrix=features,
        n_trials=int(os.environ.get("GAS_POWER_TUNE_TRIALS", settings.get("n_trials", 30))),
        top_k=int(os.environ.get("GAS_POWER_TUNE_TOP_K", settings.get("top_k", 5))),
        coarse_folds=int(settings.get("coarse_folds", 4)),
        show_progress=progress_enabled(config),
    )
    deep_settings = config.section("forecast").get("deep_learning", {})
    deep_settings = deep_settings if isinstance(deep_settings, Mapping) else {}
    deep_candidates = (
        run_deep_search(
            config,
            prepared,
            _feature_builder(config),
            feature_matrix=features,
            coarse_folds=int(deep_settings.get("coarse_folds", 4)),
            top_k=int(deep_settings.get("search_top_k", 2)),
            coarse_epochs=int(deep_settings.get("coarse_epochs", 60)),
            show_progress=progress_enabled(config),
        )
        if bool(deep_settings.get("search_enabled", False))
        else []
    )
    candidates = sorted(
        [*tree_candidates, *deep_candidates], key=lambda item: item.selection_metric
    )
    reports = config.path("reports", "reports")
    results = config.path("results")
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    horizons = _model_horizons(config)
    selection = config.raw.get("model_selection", {})
    selection = selection if isinstance(selection, Mapping) else {}
    recent_selection_folds = int(selection.get("recent_folds", 4))
    cross_selection_folds = int(selection.get("cross_month_folds", 4))
    minimum_non_degraded = int(selection.get("minimum_non_degraded_folds", 5))
    maximum_worst_degradation = float(selection.get("maximum_worst_degradation", 0.01))
    recent_config = config.raw.get("recent_validation", {})
    recent_config = recent_config if isinstance(recent_config, Mapping) else {}
    splitter = RecentWindowSplitter(
        folds=recent_selection_folds,
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )
    validation = config.section("validation")
    cross_validation = dict(validation)
    cross_validation["folds"] = cross_selection_folds
    baseline = run_rolling_validation(
        frame=prepared.model_input,
        model_factory=LastValueModel,
        splitter=splitter,
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=50,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
    )
    baseline_cross = run_rolling_validation(
        frame=prepared.model_input,
        model_factory=LastValueModel,
        splitter=splitter_from_config(cross_validation),
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=50,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
    )
    def trend_factory() -> DampedTrendModel:
        return DampedTrendModel(
            window=5,
            damping=0.85,
            interval_minutes=15,
        )
    trend = run_rolling_validation(
        frame=prepared.model_input,
        model_factory=trend_factory,
        splitter=splitter,
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=50,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
    )
    trend_cross = run_rolling_validation(
        frame=prepared.model_input,
        model_factory=trend_factory,
        splitter=splitter_from_config(cross_validation),
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=50,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
    )

    baseline_selection = _combine_selection_predictions(
        baseline.predictions,
        baseline_cross.predictions,
        recent_folds=recent_selection_folds,
        cross_month_folds=cross_selection_folds,
    )
    keys = ["fold", "origin", "target_datetime", "target", "horizon_steps"]
    condition_columns = [
        str(column)
        for column in baseline_selection.columns
        if str(column).startswith(
            ("feat_state_", "feat_missing__", "feat_outlier__", "feat_holder_", "feat_gas_")
        )
    ]
    oof = baseline_selection[keys + ["y_true", "y_pred", *condition_columns]].rename(
        columns={"y_pred": "last_value"}
    )
    trend_selection = _combine_selection_predictions(
        trend.predictions,
        trend_cross.predictions,
        recent_folds=recent_selection_folds,
        cross_month_folds=cross_selection_folds,
    )
    oof = oof.merge(
        trend_selection[keys + ["y_pred"]].rename(columns={"y_pred": "damped_trend"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    _assert_selection_fold_coverage(
        oof,
        recent_folds=recent_selection_folds,
        cross_month_folds=cross_selection_folds,
        context="基线与趋势模型",
    )
    component_builders: dict[str, Any] = {"damped_trend": trend_factory}
    component_configs: dict[str, ProjectConfig] = {}
    for candidate_index, candidate in enumerate(candidates, start=1):
        name = f"candidate_{candidate_index}_trial_{candidate.trial_number}"
        candidate_selection = _combine_selection_predictions(
            candidate.recent_predictions,
            candidate.cross_month_predictions,
            recent_folds=recent_selection_folds,
            cross_month_folds=cross_selection_folds,
        )
        oof = oof.merge(
            candidate_selection[keys + ["y_pred"]].rename(columns={"y_pred": name}),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        _assert_selection_fold_coverage(
            oof,
            recent_folds=recent_selection_folds,
            cross_month_folds=cross_selection_folds,
            context=name,
        )
        component_configs[name] = candidate_config_from_descriptor(
            config, candidate.parameters
        )
        component_builders[name] = lambda item=component_configs[name]: build_model(item)

    column_weights: dict[str, dict[str, float]] = {}
    loo_parts: list[pd.DataFrame] = []
    column_gate_rows: list[dict[str, Any]] = []
    fused = oof[keys + ["y_true"]].copy()
    fused["y_pred"] = np.nan
    fused_loo = fused.copy()
    for target in targets:
        for horizon in horizons:
            eligible = ["last_value"]
            for name in component_builders:
                candidate_gate = evaluate_oof_column_gate(
                    oof,
                    target_column=target,
                    horizon=horizon,
                    candidate_column=name,
                    minimum_non_degraded_folds=minimum_non_degraded,
                    maximum_worst_degradation=maximum_worst_degradation,
                )
                column_gate_rows.append(
                    {
                        "target": target,
                        "horizon_steps": horizon,
                        "candidate": name,
                        "gate_type": "candidate",
                        **candidate_gate.to_dict(),
                    }
                )
                if candidate_gate.passed:
                    eligible.append(name)
            weights, loo = fit_oof_weights(
                oof,
                eligible,
                target_column=target,
                horizon=horizon,
            )
            if not loo.empty:
                loo.insert(0, "target", target)
                loo.insert(1, "horizon_steps", horizon)
                loo_parts.append(loo)
            mask = (oof["target"] == target) & (oof["horizon_steps"] == horizon)
            fused.loc[mask, "y_pred"] = apply_oof_weights(
                {name: oof.loc[mask, name].to_numpy(dtype=float) for name in eligible},
                weights,
            )
            for held_out in loo.itertuples(index=False):
                held_mask = mask & (oof["fold"] == held_out.fold)
                fused_loo.loc[held_mask, "y_pred"] = apply_oof_weights(
                    {
                        name: oof.loc[held_mask, name].to_numpy(dtype=float)
                        for name in eligible
                    },
                    held_out.weights,
                )

            fusion_frame = oof.assign(fused_prediction=fused_loo["y_pred"])
            fusion_gate = evaluate_oof_column_gate(
                fusion_frame,
                target_column=target,
                horizon=horizon,
                candidate_column="fused_prediction",
                minimum_non_degraded_folds=minimum_non_degraded,
                maximum_worst_degradation=maximum_worst_degradation,
            )
            column_gate_rows.append(
                {
                    "target": target,
                    "horizon_steps": horizon,
                    "candidate": "fused",
                    "gate_type": "fusion",
                    **fusion_gate.to_dict(),
                }
            )
            prediction_name = prediction_column(target, horizon)
            if fusion_gate.passed and len(eligible) > 1:
                column_weights[prediction_name] = {
                    name: float(weight)
                    for name, weight in zip(eligible, weights)
                    if float(weight) > 1.0e-10
                }
            else:
                column_weights[prediction_name] = {"last_value": 1.0}
                fused.loc[mask, "y_pred"] = oof.loc[mask, "last_value"]
                fused_loo.loc[mask, "y_pred"] = oof.loc[mask, "last_value"]

    if fused_loo["y_pred"].isna().any():
        raise ValueError("留一折融合预测未覆盖全部 OOF 样本")
    accepted_columns = sum(
        any(name != "last_value" and weight > 0.0 for name, weight in weights.items())
        for weights in column_weights.values()
    )
    used_model_names = {
        name
        for weights in column_weights.values()
        for name, weight in weights.items()
        if name != "last_value" and weight > 0.0
    }
    components: dict[str, Any] = {"last_value": LastValueModel()}
    if accepted_columns:
        components.update(
            {
                name: component_builders[name]()
                for name in sorted(used_model_names)
            }
        )
        final_model: Any = HorizonWeightedEnsembleModel(components, column_weights)
        selected_status = "COLUMN_OOF_GATE_PASSED"
    else:
        final_model = LastValueModel()
        selected_status = "FALLBACK_LAST_VALUE"
    final_model.fit(
        prepared.model_input,
        targets,
        horizons,
        train_end=pd.Timestamp(prepared.model_input.index.max()),
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
    )
    model_path = config.path("models") / "forecast_model.joblib"
    joblib.dump(final_model, model_path, compress=3)

    candidate_rows = [
        {
            "trial": item.trial_number,
            "recent_mape": item.recent_mape,
            "recent_worst_mape": item.recent_worst_mape,
            "cross_month_mape": item.cross_month_mape,
            "selection_metric": item.selection_metric,
            **{f"gate_{key}": value for key, value in item.gate.to_dict().items() if key != "reasons"},
            "gate_reasons": " | ".join(item.gate.reasons),
            "parameters": json.dumps(item.parameters, ensure_ascii=False, sort_keys=True),
        }
        for item in candidates
    ]
    pd.DataFrame(candidate_rows).to_csv(
        reports / "high_accuracy_candidates.csv", index=False, encoding="utf-8"
    )
    oof.to_csv(reports / "high_accuracy_oof.csv", index=False, encoding="utf-8")
    fused.to_csv(reports / "high_accuracy_fused_oof.csv", index=False, encoding="utf-8")
    fused_loo.to_csv(
        reports / "high_accuracy_fused_leave_one_fold.csv",
        index=False,
        encoding="utf-8",
    )
    if loo_parts:
        pd.concat(loo_parts, ignore_index=True).to_csv(
            reports / "high_accuracy_leave_one_fold.csv", index=False, encoding="utf-8"
        )
    column_gate_frame = pd.DataFrame(column_gate_rows)
    column_gate_frame.to_csv(
        reports / "high_accuracy_column_gates.csv", index=False, encoding="utf-8"
    )
    diagnostic_predictions = oof[keys + ["y_true", *condition_columns]].copy()
    diagnostic_predictions["y_pred"] = fused_loo["y_pred"].to_numpy(dtype=float)
    largest_errors, error_groups = _large_error_reports(diagnostic_predictions)
    largest_errors.to_csv(
        reports / "high_accuracy_large_errors.csv", index=False, encoding="utf-8"
    )
    error_groups.to_csv(
        reports / "high_accuracy_error_groups.csv", index=False, encoding="utf-8"
    )
    fusion_summary = {
        "passed": accepted_columns > 0,
        "accepted_columns": accepted_columns,
        "total_columns": len(targets) * len(horizons),
        "used_models": sorted(used_model_names),
        "recent_folds": recent_selection_folds,
        "cross_month_folds": cross_selection_folds,
        "minimum_non_degraded_folds": minimum_non_degraded,
        "maximum_worst_degradation": maximum_worst_degradation,
    }
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "status": selected_status,
        "model_type": type(final_model).__name__,
        "training_data_source": prepared.source,
        "decision_inputs": "TRAINING_VALIDATION_ONLY",
        "scoring_used_for_training": False,
        "raw_labels": True,
        "seed": config.seed,
        "targets": targets,
        "horizons": horizons,
        "feature_columns": list(features.columns),
        "fusion_gate": fusion_summary,
        "column_gates": column_gate_rows,
        "weights": column_weights,
        "model_sha256": model_hash,
        "optuna_best_value": float(study.best_value),
        "optuna_trials": len(study.trials),
    }
    _write_json(metadata, config.path("models") / "forecast_model_metadata.json")
    _write_json(metadata, reports / "high_accuracy_selection.json")
    return {
        "status": selected_status,
        "passed_candidates": len(used_model_names),
        "fusion_gate": fusion_summary,
        "model": str(model_path),
        "model_sha256": model_hash,
        "reports": str(reports),
        "results": str(results),
    }


def _prediction_origins(config: ProjectConfig, frame: pd.DataFrame) -> pd.DatetimeIndex:
    settings = config.section("forecast").get("prediction_origins", {})
    mode = str(settings.get("mode", "tail"))
    timestamp_format = str(config.section("data")["timestamp_format"])
    if mode == "tail":
        count = int(settings.get("count", 1))
        if count <= 0 or count > len(frame):
            raise ValueError("预测起点 count 必须在数据行数范围内")
        return pd.DatetimeIndex(frame.index[-count:])
    if mode == "file":
        configured_path = settings.get("file")
        if not configured_path:
            raise ValueError("prediction_origins.mode=file 时必须配置 file")
        path = Path(str(configured_path))
        path = path if path.is_absolute() else (config.root / path).resolve()
        origins_frame = pd.read_csv(path, encoding="utf-8")
        if "datetime" not in origins_frame:
            raise ValueError(f"预测起点文件缺少 datetime: {path}")
        origins = pd.to_datetime(
            origins_frame["datetime"], format=timestamp_format, errors="raise"
        )
        return pd.DatetimeIndex(origins)
    if mode == "scoring":
        if frame.empty:
            raise ValueError("评分期预测起点不能为空")
        return pd.DatetimeIndex(frame.index)
    raise ValueError(f"不支持的预测起点模式: {mode}")


def predict_pipeline(config: ProjectConfig) -> dict[str, Any]:
    inference_started = time.perf_counter()
    _assert_final_residual_gate(config)
    train_prepared, train_features, _ = prepare_prepared_data(config)
    train_frame = train_prepared.model_input
    model_path = config.path("models") / "forecast_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在，请先运行 train: {model_path}")
    model = joblib.load(model_path)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    forecast_config = config.section("forecast")
    short_horizons = list(range(1, int(forecast_config["short_steps"]) + 1))
    interval_minutes = int(config.section("optimization").get("interval_minutes", 15))
    timestamp_format = str(config.section("data")["timestamp_format"])
    origin_mode = str(forecast_config.get("prediction_origins", {}).get("mode", "tail"))

    if origin_mode == "scoring":
        scoring_raw = _load_scoring_prepared(config)
        history_points = int(
            config.raw.get("competition_compliance", {}).get("scoring_history_points", 672)
            if isinstance(config.raw.get("competition_compliance", {}), Mapping)
            else 672
        )
        scoring_prepared = prepare_scoring_with_history(
            train_prepared,
            scoring_raw,
            config.section("preprocessing"),
            history_points=history_points,
        )
        scoring_start = pd.Timestamp(scoring_prepared.model_input.index.min())
        frame = pd.concat(
            [train_frame.loc[train_frame.index < scoring_start], scoring_prepared.model_input],
            axis=0,
        ).sort_index()
        if not frame.index.is_unique:
            raise ValueError("训练历史与评分期输入拼接后存在重复时间戳")
        origins = _prediction_origins(config, scoring_prepared.model_input)
        features = _feature_builder(config).transform(frame)
        settings = config.raw.get("prediction_input", {})
        table_paths = settings.get("table_paths", {}) if isinstance(settings, Mapping) else {}
        if not isinstance(table_paths, Mapping) or not table_paths:
            raise ValueError("scoring 模式必须配置 prediction_input.table_paths")
        original_input = load_original_input_frame(
            config,
            config.path("scoring_data"),
            table_paths,
        )
        missing_input_origins = origins.difference(original_input.index)
        if len(missing_input_origins):
            raise ValueError(
                f"官方原始输入未覆盖全部预测起点: {missing_input_origins[:3].tolist()}"
            )
        training_tables = config.section("data")["tables"]
        training_paths = {
            str(table_name): str(training_tables[table_name]["path"])
            for table_name in table_paths
            if table_name in training_tables
        }
        training_original = load_original_input_frame(
            config,
            config.path("data"),
            training_paths,
        )
        input_sources, input_quality = prepare_submission_sources(
            training_original,
            original_input,
            config.section("preprocessing"),
            origins,
            history_points=history_points,
        )
    else:
        frame = train_frame
        features = train_features
        origins = _prediction_origins(config, frame)
        registry = FeatureAvailabilityRegistry.from_config(config)
        approved_sources = sorted(
            set(frame.columns).intersection(registry.allowed_source_columns("long"))
        )
        input_sources = frame.loc[origins, approved_sources]
        input_quality = {
            "invalid_columns": [],
            "missing_repairs": {},
            "outlier_repairs": {},
        }

    prediction_horizons = _model_horizons(config)
    raw_prediction = model.predict(frame, origins, targets, prediction_horizons)

    post_config = config.raw.get("postprocessing", {})
    post_result = PhysicalForecastPostprocessor(post_config, interval_minutes).apply(
        raw_prediction, frame, origins, targets, prediction_horizons
    )
    prediction = (
        post_result.predictions
        if bool(post_config.get("enabled", True))
        else raw_prediction
    )
    short_prediction = prediction[
        prediction_columns(targets, short_horizons, interval_minutes)
    ]
    results_dir = config.path("results")
    raw_short = raw_prediction[
        prediction_columns(targets, short_horizons, interval_minutes)
    ]
    write_time_frame(raw_short, results_dir / "raw_s_result.csv", timestamp_format)
    capacities = post_config.get("target_capacity_mw", {})
    submission_config = config.raw.get("submission", {})
    tolerance = float(submission_config.get("capacity_tolerance_mw", 1.0e-6))
    short_output = write_forecast_csv(
        short_prediction,
        results_dir / "s_result.csv",
        targets,
        short_horizons,
        timestamp_format,
        interval_minutes,
        expected_origins=origins,
        capacity_bounds=dict(capacities) if isinstance(capacities, dict) else None,
        enforce_target_consistency=bool(post_config.get("enforce_target_consistency", True)),
        capacity_tolerance=tolerance,
    )

    long_output: pd.DataFrame | None = None
    if not _is_preliminary(config):
        long_horizons = list(range(1, int(forecast_config["long_steps"]) + 1))
        write_time_frame(raw_prediction, results_dir / "raw_l_result.csv", timestamp_format)
        long_output = write_forecast_csv(
            prediction,
            results_dir / "l_result.csv",
            targets,
            long_horizons,
            timestamp_format,
            interval_minutes,
            expected_origins=origins,
            capacity_bounds=dict(capacities) if isinstance(capacities, dict) else None,
            enforce_target_consistency=bool(post_config.get("enforce_target_consistency", True)),
            capacity_tolerance=tolerance,
        )

    feature_output, feature_quality = sanitize_submission_features(
        train_features,
        features.loc[origins],
    )
    invalid_feature_names = [
        str(column) for column in feature_output.columns if not str(column).startswith("feat_")
    ]
    if invalid_feature_names:
        raise ValueError(f"派生输入字段必须以 feat_ 开头: {invalid_feature_names[:5]}")
    input_frame = pd.concat([input_sources, feature_output], axis=1)
    if input_frame.columns.duplicated().any():
        raise ValueError("input.csv 构建后存在重复字段")
    quality_settings = submission_config.get("quality_normalization", {})
    if isinstance(quality_settings, Mapping) and bool(quality_settings.get("enabled", False)):
        input_frame, matrix_quality = normalize_submission_input_frame(
            input_frame,
            quality_settings,
        )
    else:
        matrix_quality = {"enabled": False}
    if _is_preliminary(config) and len(origins) != 192:
        raise ValueError(f"初赛评分输入必须为 192 行，实际为 {len(origins)} 行")
    validate_preliminary_input_frame(
        input_frame,
        origins,
        interval_minutes=interval_minutes,
    )
    input_path = results_dir / "input.csv"
    write_time_frame(input_frame, input_path, timestamp_format)

    # 平台读取的是 CSV 而不是内存 DataFrame，必须按真实落盘值重新验收。
    persisted_input = pd.read_csv(input_path, encoding="utf-8")
    if "datetime" not in persisted_input:
        raise ValueError("落盘后的 input.csv 缺少 datetime 字段")
    persisted_input["datetime"] = pd.to_datetime(
        persisted_input["datetime"],
        format=timestamp_format,
        errors="raise",
    )
    persisted_input = persisted_input.set_index("datetime")
    persisted_input.index = pd.DatetimeIndex(persisted_input.index, name="datetime")
    validate_preliminary_input_frame(
        persisted_input,
        origins,
        interval_minutes=interval_minutes,
    )
    serialized_quality = inspect_submission_input_quality(
        persisted_input,
        iqr_multiplier=float(quality_settings.get("iqr_multiplier", 1.5))
        if isinstance(quality_settings, Mapping)
        else 1.5,
        iqr_interpolations=quality_settings.get(
            "iqr_interpolations",
            ["linear", "lower", "higher", "midpoint", "nearest"],
        )
        if isinstance(quality_settings, Mapping)
        else ["linear", "lower", "higher", "midpoint", "nearest"],
        zscore_threshold=float(quality_settings.get("zscore_threshold", 3.0))
        if isinstance(quality_settings, Mapping)
        else 3.0,
    )
    serialized_quality_passed = (
        serialized_quality["nonfinite_cells"] == 0
        and not serialized_quality["constant_columns"]
        and not serialized_quality["duplicate_columns"]
        and serialized_quality["iqr_outlier_cells_all_methods"] == 0
        and serialized_quality["zscore_outlier_cells"] == 0
    )
    if bool(matrix_quality.get("enabled", False)) and not serialized_quality_passed:
        raise ValueError(f"落盘后的 input.csv 质量门禁失败: {serialized_quality}")
    matrix_quality["serialized_quality"] = serialized_quality
    matrix_quality["serialization_passed"] = serialized_quality_passed
    _write_json(
        {
            "source_cleaning": input_quality,
            "feature_pruning": feature_quality,
            "matrix_normalization": matrix_quality,
            "rows": len(persisted_input),
            "columns": len(persisted_input.columns),
            "finite": True,
        },
        results_dir / "reports" / "submission_input_quality.json",
    )
    freeze_manifest = freeze_submission(results_dir)
    elapsed_seconds = float(time.perf_counter() - inference_started)
    seconds_per_sample = elapsed_seconds / max(1, len(origins))
    performance = config.raw.get("performance", {})
    max_per_sample = float(performance.get("max_seconds_per_sample", 30.0))
    max_total = float(performance.get("max_total_inference_seconds", 1800.0))
    runtime_report = {
        "origins": len(origins),
        "total_inference_seconds": elapsed_seconds,
        "seconds_per_sample": seconds_per_sample,
        "max_seconds_per_sample": max_per_sample,
        "max_total_inference_seconds": max_total,
        "within_limits": seconds_per_sample <= max_per_sample and elapsed_seconds <= max_total,
    }
    _write_json(runtime_report, results_dir / "reports" / "inference_runtime.json")
    if bool(performance.get("enforce_limits", True)) and not runtime_report["within_limits"]:
        raise RuntimeError(
            "预测推理超过 PDF 建议时限："
            f"单样本 {seconds_per_sample:.3f}s/{max_per_sample:.3f}s，"
            f"总计 {elapsed_seconds:.3f}s/{max_total:.3f}s"
        )
    result = {
        "origins": len(origins),
        "short_shape": list(short_output.shape),
        "s_result": str(results_dir / "s_result.csv"),
        "input": str(results_dir / "input.csv"),
        "input_original_columns": len(input_sources.columns),
        "submission_freeze": str(results_dir / "submission_freeze.json"),
        "submission_frozen": True,
        "submission_hashes": {
            name: details["sha256"]
            for name, details in freeze_manifest["files"].items()
        },
        "postprocessing_adjusted_cells": post_result.adjusted_cells,
        "postprocessing_adjustments": post_result.adjustments,
        "runtime": runtime_report,
    }
    if long_output is not None:
        result["long_shape"] = list(long_output.shape)
        result["l_result"] = str(results_dir / "l_result.csv")
    return result


def _load_scoring_prepared(config: ProjectConfig) -> PreparedForecastData:
    """从受限评分期目录读取预测输入，不修改训练配置或训练缓存。"""

    settings = config.raw.get("prediction_input", {})
    if not isinstance(settings, Mapping):
        raise TypeError("prediction_input 必须是字典")
    table_paths = settings.get("table_paths", {})
    if not isinstance(table_paths, Mapping) or not table_paths:
        raise ValueError("prediction_input.table_paths 必须是非空字典")
    scoring_raw = deepcopy(config.raw)
    scoring_raw["paths"]["data"] = str(config.path("scoring_data"))
    scoring_tables = scoring_raw["data"]["tables"]
    for table_name, table_path in table_paths.items():
        if table_name not in scoring_tables:
            raise ValueError(f"评分期输入配置包含未知数据表: {table_name}")
        scoring_tables[table_name]["path"] = str(table_path)
    scoring_config = ProjectConfig(
        raw=scoring_raw,
        source=config.source,
        root=config.root,
    )
    prepared, _ = ConfiguredDataLoader(scoring_config).load_prepared(source="scoring")
    return prepared


def validate_pipeline(config: ProjectConfig) -> dict[str, Any]:
    prepared, features, _ = prepare_prepared_data(config)
    prepared.assert_training_allowed()
    frame = prepared.model_input
    builder = _feature_builder(config)
    cutoff = pd.Timestamp(frame.index[len(frame) // 2])
    assert_feature_causality(builder, frame, cutoff)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    rolling_windows = [int(value) for value in config.section("features")["rolling_windows"]]
    assert_shift_before_rolling(builder, frame, targets[0], rolling_windows[0])

    horizons = _model_horizons(config)
    validation_config = config.section("validation")
    interval_minutes = int(config.section("optimization").get("interval_minutes", 15))
    artifacts = run_rolling_validation(
        frame=frame,
        model_factory=lambda: build_model(config),
        splitter=splitter_from_config(validation_config),
        target_columns=targets,
        horizons=horizons,
        interval_minutes=interval_minutes,
        near_zero_threshold=float(validation_config.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=int(validation_config.get("worst_error_count", 50)),
        feature_builder=builder,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
        show_progress=progress_enabled(config),
        progress_description="滚动验证",
    )
    recent_artifacts = None
    if _is_preliminary(config):
        recent_config = config.raw.get("recent_validation", {})
        if not isinstance(recent_config, Mapping):
            recent_config = {}
        recent_artifacts = run_rolling_validation(
            frame=frame,
            model_factory=lambda: build_model(config),
            splitter=RecentWindowSplitter(
                folds=int(recent_config.get("folds", 10)),
                validation_points=int(recent_config.get("validation_points", 192)),
                step_points=int(recent_config.get("step_points", 192)),
                rolling_train_points=(
                    int(recent_config["rolling_train_points"])
                    if recent_config.get("rolling_train_points") is not None
                    else None
                ),
            ),
            target_columns=targets,
            horizons=horizons,
            interval_minutes=interval_minutes,
            near_zero_threshold=float(validation_config.get("near_zero_threshold", 1.0e-6)),
            worst_error_count=int(validation_config.get("worst_error_count", 50)),
            feature_builder=builder,
            raw_targets=prepared.raw_targets,
            feature_matrix=features,
            data_source=prepared.source,
            show_progress=progress_enabled(config),
            progress_description="最近两天滚动验证",
        )
    results_dir = config.path("results")
    artifacts.metrics.to_csv(
        results_dir / "validation_metrics.csv", index=False, encoding="utf-8"
    )
    artifacts.worst_errors.to_csv(
        results_dir / "validation_worst_errors.csv", index=False, encoding="utf-8"
    )
    artifacts.predictions.to_csv(
        results_dir / "validation_predictions.csv", index=False, encoding="utf-8"
    )
    if recent_artifacts is not None:
        recent_artifacts.metrics.to_csv(
            results_dir / "recent_validation_metrics.csv", index=False, encoding="utf-8"
        )
        recent_artifacts.predictions.to_csv(
            results_dir / "recent_validation_predictions.csv", index=False, encoding="utf-8"
        )
    split_payload = [
        {
            "fold": split.fold,
            "train_start": str(split.train_start),
            "train_end": str(split.train_end),
            "validation_start": str(split.validation_origins.min()),
            "validation_end": str(split.validation_origins.max()),
            "validation_origins": len(split.validation_origins),
        }
        for split in artifacts.splits
    ]
    _write_json(split_payload, results_dir / "validation_splits.json")
    return {
        "folds": len(artifacts.splits),
        "prediction_pairs": len(artifacts.predictions),
        "metrics_rows": len(artifacts.metrics),
        "metric_files_generated": True,
        "warning": _data_notice(config),
        "leakage_checks": "passed",
        "validation_protocol": {
            "cross_month_folds": len(artifacts.splits),
            "recent_folds": len(recent_artifacts.splits) if recent_artifacts is not None else 0,
            "raw_labels": True,
            "scoring_used": False,
        },
    }


def audit_data_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, features, _ = prepare_data(config)
    registry = FeatureAvailabilityRegistry.from_config(config)
    artifacts = run_data_audit(config, frame, features, registry)
    reports = config.path("reports", "reports")
    _write_json(artifacts.summary, reports / "data_audit.json")
    (reports / "data_audit.md").write_text(artifacts.markdown, encoding="utf-8")
    artifacts.lag_correlations.to_csv(
        reports / "lag_correlation.csv", index=False, encoding="utf-8"
    )
    artifacts.suspicious_features.to_csv(
        reports / "suspicious_features.csv", index=False, encoding="utf-8"
    )
    future = artifacts.suspicious_features.loc[
        (artifacts.suspicious_features["risk"] == "red")
        & (
            pd.to_numeric(
                artifacts.suspicious_features["target_offset_steps"], errors="coerce"
            )
            > 0
        )
    ]
    return {
        "passed": artifacts.summary["passed"],
        "formal_feature_count": artifacts.summary["formal_feature_count"],
        "suspicious_feature_count": len(artifacts.suspicious_features),
        "future_red_risk_count": len(future),
        "future_red_fields": sorted(future["feature"].astype(str).unique().tolist()),
        "reports": str(reports),
        "warning": _data_notice(config),
    }


def benchmark_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, _, _ = prepare_data(config)
    settings = config.section("benchmark")
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    forecast = config.section("forecast")
    horizon_mode = str(settings.get("horizons", "long"))
    horizon_count = int(
        forecast["short_steps"] if horizon_mode == "short" else forecast["long_steps"]
    )
    horizons = list(range(1, horizon_count + 1))
    validation = config.section("validation")
    interval = int(config.section("optimization").get("interval_minutes", 15))
    artifacts = run_baseline_benchmark(
        frame,
        settings.get("baselines", []),
        splitter_from_config(validation),
        targets,
        horizons,
        interval,
        float(validation.get("near_zero_threshold", 1.0e-6)),
        settings.get("reachability", {}),
        show_progress=progress_enabled(config),
    )
    reports = config.path("reports", "reports")
    artifacts.metrics.to_csv(reports / "benchmark_metrics.csv", index=False, encoding="utf-8")
    artifacts.best.to_csv(reports / "benchmark_best.csv", index=False, encoding="utf-8")
    artifacts.predictions.to_csv(
        reports / "benchmark_predictions.csv", index=False, encoding="utf-8"
    )
    artifacts.alignment_diagnostics.to_csv(
        reports / "benchmark_alignment_diagnostics.csv", index=False, encoding="utf-8"
    )
    registry = FeatureAvailabilityRegistry.from_config(config)
    diagnostic_fields = [
        str(column)
        for column in frame.select_dtypes(include=[np.number]).columns
        if str(column) not in targets and not registry.get(str(column)).allowed("long")
    ]
    future_lag = calculate_lag_correlations(
        frame,
        diagnostic_fields,
        targets,
        1,
        int(config.section("audit").get("lag_max_steps", 96)),
        interval,
    )
    high_threshold = float(config.section("audit").get("high_correlation_threshold", 0.995))
    high_future = future_lag.loc[
        future_lag["absolute_correlation"] >= high_threshold
    ]
    artifacts.reachability["future_diagnostic_max_absolute_correlation"] = (
        float(future_lag["absolute_correlation"].max()) if not future_lag.empty else None
    )
    artifacts.reachability["future_diagnostic_fields"] = sorted(
        high_future["feature"].astype(str).unique().tolist()
    )
    if (
        artifacts.reachability["best_score_percent"]
        < float(settings.get("reachability", {}).get("leaderboard_score_percent", 99.9))
        and not high_future.empty
    ):
        artifacts.reachability["messages"].append(
            "合法基线未达到榜首区间，但存在未来高相关诊断字段；该差距属于违规风险证据，不能用于入模。"
        )
    score_rows = []
    scorer = ConfigurableScorer(config.section("scoring"))
    for baseline, group in artifacts.predictions.groupby("baseline", sort=True):
        score_rows.append({"baseline": baseline, **scorer.score(group).to_dict()})
    pd.DataFrame(score_rows).to_csv(
        reports / "benchmark_scores.csv", index=False, encoding="utf-8"
    )
    _write_json(artifacts.reachability, reports / "benchmark_reachability.json")
    return {
        "baselines": int(artifacts.metrics["baseline"].nunique()),
        "prediction_pairs": len(artifacts.predictions),
        "best_rows": len(artifacts.best),
        "reachability": artifacts.reachability,
        "warning": _data_notice(config),
    }


def discover_relations_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, _, _ = prepare_data(config)
    registry = FeatureAvailabilityRegistry.from_config(config)
    artifacts = discover_relations(
        frame,
        config.section("data")["roles"],
        registry,
        config.section("relations"),
    )
    reports = config.path("reports", "reports")
    _write_json(artifacts.summary, reports / "relation_summary.json")
    (reports / "relation_summary.md").write_text(artifacts.markdown, encoding="utf-8")
    artifacts.coefficients.to_csv(
        reports / "relation_coefficients.csv", index=False, encoding="utf-8"
    )
    artifacts.setpoints.to_csv(reports / "setpoint_analysis.csv", index=False, encoding="utf-8")
    artifacts.delays.to_csv(reports / "delay_relations.csv", index=False, encoding="utf-8")
    return {**artifacts.summary, "reports": str(reports), "warning": _data_notice(config)}


def _postprocess_validation_tidy(
    config: ProjectConfig,
    frame: pd.DataFrame,
    tidy: pd.DataFrame,
    targets: list[str],
    horizons: list[int],
    interval: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index_columns = ["fold", "origin"]
    wide_parts = []
    for (_, origin), group in tidy.groupby(index_columns, sort=False):
        values = {
            prediction_column(str(row.target), int(row.horizon_steps), interval): float(row.y_pred)
            for row in group.itertuples()
        }
        wide_parts.append(pd.Series(values, name=pd.Timestamp(origin)))
    wide = pd.DataFrame(wide_parts)
    wide.index = pd.DatetimeIndex(wide.index, name="datetime")
    wide = wide[~wide.index.duplicated(keep="first")]
    result = PhysicalForecastPostprocessor(
        config.raw.get("postprocessing", {}), interval
    ).apply(wide, frame, wide.index, targets, horizons)
    processed = tidy.copy()
    processed["y_pred"] = [
        result.predictions.at[
            pd.Timestamp(row.origin),
            prediction_column(str(row.target), int(row.horizon_steps), interval),
        ]
        for row in processed.itertuples()
    ]
    return processed, {
        "adjusted_cells": result.adjusted_cells,
        "adjustments": result.adjustments,
    }


def backtest_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, _, _ = prepare_data(config)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    horizons = list(range(1, int(config.section("forecast")["long_steps"]) + 1))
    interval = int(config.section("optimization").get("interval_minutes", 15))
    near_zero = float(config.section("validation").get("near_zero_threshold", 1.0e-6))
    builder = _feature_builder(config)
    report_frames: dict[str, list[pd.DataFrame]] = {
        "predictions": [], "fold": [], "aggregate": [], "condition": [],
        "horizon": [], "gain": [], "post": [],
    }
    coverage: dict[str, Any] = {}
    post_adjustments: dict[str, Any] = {}
    fold_gains: list[float] = []
    backtest_items = list(config.section("backtest").items())
    for split_name, split_config in track_progress(
        backtest_items,
        config=config,
        description="回测方案",
        total=len(backtest_items),
        unit="组",
        leave=False,
    ):
        splitter = splitter_from_config(split_config)
        model_artifacts = run_rolling_validation(
            frame, lambda: build_model(config), splitter, targets, horizons, interval,
            near_zero, int(config.section("validation").get("worst_error_count", 50)), builder,
            show_progress=progress_enabled(config),
            progress_description=f"{split_name} 模型",
        )
        baseline_artifacts = run_rolling_validation(
            frame, lambda: LastValueModel(interval), splitter, targets, horizons, interval,
            near_zero, 0, builder,
            show_progress=progress_enabled(config),
            progress_description=f"{split_name} 基线",
        )
        processed, adjustment = _postprocess_validation_tidy(
            config, frame, model_artifacts.predictions, targets, horizons, interval
        )
        post_adjustments[str(split_name)] = adjustment
        post_metrics = compare_postprocessing_metrics(
            model_artifacts.predictions, processed, near_zero
        )
        primary = processed if bool(config.raw.get("postprocessing", {}).get("enabled", True)) else model_artifacts.predictions
        details = summarize_detailed_validation(
            primary, baseline_artifacts.predictions, near_zero
        )
        model_overall = details.fold_metrics.loc[
            details.fold_metrics["scope"] == "overall", ["fold", "mape"]
        ]
        baseline_fold_rows = []
        for fold, group in baseline_artifacts.predictions.groupby("fold", sort=True):
            baseline_metric = ConfigurableScorer(
                {"formula": "one_minus_mape", "target_weights": {}}
            ).score(group)
            baseline_fold_rows.append(
                {"fold": int(fold), "mape": 1.0 - baseline_metric.prediction_score}
            )
        baseline_overall = pd.DataFrame(baseline_fold_rows).rename(
            columns={"mape": "baseline_mape"}
        )
        fold_comparison = model_overall.merge(baseline_overall, on="fold", how="inner")
        fold_gains.extend(
            (fold_comparison["baseline_mape"] - fold_comparison["mape"])
            .astype(float)
            .tolist()
        )
        for key, data_frame in (
            ("predictions", primary), ("fold", details.fold_metrics),
            ("aggregate", details.aggregate_metrics), ("condition", details.condition_metrics),
            ("horizon", details.horizon_curve), ("gain", details.baseline_gain),
            ("post", post_metrics),
        ):
            data_frame = data_frame.copy()
            data_frame.insert(0, "validation_type", str(split_name))
            report_frames[key].append(data_frame)
        coverage[str(split_name)] = details.coverage

    reports = config.path("reports", "reports")
    filenames = {
        "predictions": "backtest_predictions.csv",
        "fold": "backtest_fold_metrics.csv",
        "aggregate": "backtest_aggregate_metrics.csv",
        "condition": "backtest_condition_metrics.csv",
        "horizon": "backtest_horizon_curve.csv",
        "gain": "backtest_gain_vs_last_value.csv",
        "post": "postprocessing_metrics.csv",
    }
    counts = {}
    for key, parts in report_frames.items():
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        combined.to_csv(reports / filenames[key], index=False, encoding="utf-8")
        counts[key] = len(combined)
    _write_json(
        {"coverage": coverage, "postprocessing": post_adjustments},
        reports / "backtest_summary.json",
    )
    residual_gate = evaluate_residual_gate(config.raw.get("residual_model", {}), fold_gains)
    residual_gate["evaluated_model_type"] = str(
        config.section("forecast").get("model", {}).get("type", "")
    )
    _write_json(residual_gate, reports / "residual_gate.json")
    gpu_gate = evaluate_gpu_gate(
        config.raw.get("gpu", {}), cpu_baselines_complete=True, fold_gains=fold_gains
    )
    _write_json(gpu_gate.to_dict(), reports / "gpu_gate.json")
    return {
        "validation_types": list(config.section("backtest")),
        "rows": counts,
        "coverage": coverage,
        "residual_gate": residual_gate,
        "gpu_gate": gpu_gate.to_dict(),
        "warning": _data_notice(config),
    }


def _assert_final_residual_gate(config: ProjectConfig) -> None:
    model_spec = config.section("forecast").get("model", {})
    model_type = str(model_spec.get("type", "")) if isinstance(model_spec, dict) else ""
    if not model_type.startswith("residual_"):
        return
    residual = config.raw.get("residual_model", {})
    if not bool(residual.get("enabled", False)):
        raise ValueError("残差模型仅处于实验状态；请先完成 backtest 并显式启用 residual_model.enabled")
    gate_path = config.path("reports", "reports") / "residual_gate.json"
    if not gate_path.exists():
        raise ValueError("缺少 residual_gate.json；残差模型必须先完成全部时间折回测")
    with gate_path.open("r", encoding="utf-8") as stream:
        gate = json.load(stream)
    if not bool(gate.get("allowed", False)) or gate.get("evaluated_model_type") != model_type:
        raise ValueError("残差模型未通过当前模型类型的全部时间折稳定改善门控")


def validate_submission_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, features, _ = prepare_data(config)
    origins = _prediction_origins(config, frame)
    metadata_path = config.path("models") / "forecast_model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"模型元数据不存在，请先运行 train: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    manifest = validate_submission_bundle(
        config, origins, metadata, list(features.columns)
    )
    return {
        "manifest": str(config.path("results") / "submission_manifest.json"),
        "files": manifest["files"],
        "checks": manifest["checks"],
    }


def _configured_columns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [column for item in value for column in _configured_columns(item)]
    if isinstance(value, Mapping):
        return [column for item in value.values() for column in _configured_columns(item)]
    return []


def _sum_configured_series(
    frame: pd.DataFrame,
    value: Any,
    *,
    required: bool,
    role_name: str,
) -> pd.Series:
    columns = [column for column in _configured_columns(value) if column in frame]
    if not columns:
        if required:
            raise ValueError(f"资源边界缺少字段角色: {role_name}")
        return pd.Series(0.0, index=frame.index, dtype=float)
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=1)


def _resource_boundary_history(
    config: ProjectConfig,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    roles = config.section("data")["roles"]
    optimization = config.section("optimization")
    production_roles = roles["gas_production"]
    user_roles = roles["gas_user_demand"]
    process_roles = roles.get("gas_process_demand", {})
    history = pd.DataFrame(index=frame.index)
    target_names: dict[str, tuple[str, str]] = {}
    for gas_type in optimization["gas_types"]:
        production_name = f"resource_production_{gas_type}"
        demand_name = f"resource_priority_demand_{gas_type}"
        history[production_name] = _sum_configured_series(
            frame,
            production_roles[gas_type],
            required=True,
            role_name=f"gas_production.{gas_type}",
        )
        history[demand_name] = _sum_configured_series(
            frame,
            user_roles[gas_type],
            required=True,
            role_name=f"gas_user_demand.{gas_type}",
        ) + _sum_configured_series(
            frame,
            process_roles.get(gas_type, []),
            required=False,
            role_name=f"gas_process_demand.{gas_type}",
        )
        target_names[str(gas_type)] = (production_name, demand_name)
    history["resource_baseline_generation"] = pd.to_numeric(
        frame["generator_all"], errors="coerce"
    )
    return history, target_names


def _forecast_resource_boundaries(
    config: ProjectConfig,
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    periods: int,
    interval_minutes: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    history, target_names = _resource_boundary_history(config, frame)
    optimization = config.section("optimization")
    model_spec = optimization.get("resource_forecast", {"type": "seasonal_naive", "period_steps": 96})
    if not isinstance(model_spec, Mapping):
        raise ValueError("optimization.resource_forecast 必须是字典")
    model = baseline_from_spec(model_spec, interval_minutes)
    targets = [name for pair in target_names.values() for name in pair]
    targets.append("resource_baseline_generation")
    horizons = list(range(1, periods + 1))
    model.fit(history.loc[:origin], targets, horizons, train_end=origin)
    predictions = model.predict(
        history,
        pd.DatetimeIndex([origin]),
        targets,
        horizons,
    )

    production: dict[str, np.ndarray] = {}
    demand: dict[str, np.ndarray] = {}
    timestamps = pd.date_range(
        origin + pd.Timedelta(minutes=interval_minutes),
        periods=periods,
        freq=f"{interval_minutes}min",
    )
    report = pd.DataFrame(index=timestamps)
    report.index.name = "datetime"
    for gas_type, (production_name, demand_name) in target_names.items():
        production_values = np.asarray(
            [predictions.at[origin, prediction_column(production_name, horizon, interval_minutes)] for horizon in horizons],
            dtype=float,
        )
        demand_values = np.asarray(
            [predictions.at[origin, prediction_column(demand_name, horizon, interval_minutes)] for horizon in horizons],
            dtype=float,
        )
        production[gas_type] = np.clip(production_values, 0.0, None)
        demand[gas_type] = np.clip(demand_values, 0.0, None)
        report[f"production_{gas_type}_pred"] = production[gas_type]
        report[f"priority_demand_{gas_type}_pred"] = demand[gas_type]
        report[f"available_before_storage_{gas_type}_pred"] = (
            production[gas_type] - demand[gas_type]
        )
    baseline_generation = np.clip(
        np.asarray(
            [
                predictions.at[
                    origin,
                    prediction_column("resource_baseline_generation", horizon, interval_minutes),
                ]
                for horizon in horizons
            ],
            dtype=float,
        ),
        0.0,
        None,
    )
    report["baseline_generation_pred"] = baseline_generation
    return production, demand, baseline_generation, report


def _dispatch_input(
    config: ProjectConfig,
    frame: pd.DataFrame,
) -> tuple[DispatchInput, pd.DataFrame]:
    optimization = config.section("optimization")
    roles = config.section("data")["roles"]
    periods = int(optimization.get("horizon_steps", 96))
    interval_minutes = int(optimization.get("interval_minutes", 15))
    origin = pd.Timestamp(frame.index.max())
    timestamps = pd.date_range(
        origin + pd.Timedelta(minutes=interval_minutes),
        periods=periods,
        freq=f"{interval_minutes}min",
    )
    production, demand, baseline_generation, resource_report = _forecast_resource_boundaries(
        config, frame, origin, periods, interval_minutes
    )
    initial_storage: dict[str, float] = {}
    for gas_type in optimization["gas_types"]:
        holder_column = str(roles["gas_holder"][gas_type])
        storage_series = pd.to_numeric(frame[holder_column], errors="coerce").loc[:origin].dropna()
        initial_storage[gas_type] = float(storage_series.iloc[-1])

    if "electricity_price" in frame:
        price_model = baseline_from_spec(
            {"type": "seasonal_naive", "period_steps": 96}, interval_minutes
        )
        price_model.fit(frame.loc[:origin], ["electricity_price"], list(range(1, periods + 1)), origin)
        price_prediction = price_model.predict(
            frame,
            pd.DatetimeIndex([origin]),
            ["electricity_price"],
            list(range(1, periods + 1)),
        )
        price = np.asarray(
            [
                price_prediction.at[
                    origin, prediction_column("electricity_price", horizon, interval_minutes)
                ]
                for horizon in range(1, periods + 1)
            ],
            dtype=float,
        )
    else:
        fallback = optimization["fallback_price"]
        valley = configured_value(fallback["valley"], "optimization.fallback_price.valley")
        flat = configured_value(fallback["flat"], "optimization.fallback_price.flat")
        peak = configured_value(fallback["peak"], "optimization.fallback_price.peak")
        hours = timestamps.hour
        price = np.where(
            ((hours >= 8) & (hours < 11)) | ((hours >= 17) & (hours < 22)),
            peak,
            np.where((hours < 7) | (hours >= 23), valley, flat),
        ).astype(float)
    resource_report["electricity_price"] = price
    return (
        DispatchInput(
            timestamps=timestamps,
            production=production,
            user_demand=demand,
            initial_storage=initial_storage,
            electricity_price=price,
            baseline_generation_mw=baseline_generation,
        ),
        resource_report,
    )


def optimize_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, _, _ = prepare_data(config)
    optimization_config = config.section("optimization")
    optimizer = HighsDispatchOptimizer(optimization_config)
    dispatch_input, resource_report = _dispatch_input(config, frame)
    write_time_frame(
        resource_report,
        config.path("reports", str(config.path("results") / "reports"))
        / "resource_boundary_forecast.csv",
        str(config.section("data")["timestamp_format"]),
    )
    diagnostics_path = config.path("results") / "optimization_diagnostics.json"
    try:
        result = optimizer.solve(dispatch_input)
    except OptimizationError as exc:
        _write_json(exc.diagnostics.to_dict(), diagnostics_path)
        raise

    gas_columns = [
        str(optimization_config["output_columns"][gas_type])
        for gas_type in optimization_config["gas_types"]
    ]
    timestamp_format = str(config.section("data")["timestamp_format"])
    output = write_optimization_csv(
        result.gas_plan,
        config.path("results") / "opt_result.csv",
        gas_columns,
        timestamp_format,
    )
    write_time_frame(
        result.unit_plan,
        config.path("results") / "optimization_unit_plan.csv",
        timestamp_format,
    )
    write_time_frame(
        result.storage_plan,
        config.path("results") / "optimization_storage_plan.csv",
        timestamp_format,
    )
    _write_json(result.diagnostics.to_dict(), diagnostics_path)
    return {
        "rows": len(output),
        "opt_result": str(config.path("results") / "opt_result.csv"),
        "diagnostics": result.diagnostics.to_dict(),
    }


def demo_pipeline(config: ProjectConfig) -> dict[str, Any]:
    data_directory = config.path("data")
    if data_directory.name != "synthetic":
        # 演示数据始终放在独立子目录，避免覆盖 data 中的正式数据。
        config.section("paths")["data"] = str(data_directory / "synthetic")
    stages = (
        ("synthetic", generate_synthetic_pipeline),
        ("train", train_pipeline),
        ("validate", validate_pipeline),
        ("predict", predict_pipeline),
        ("optimize", optimize_pipeline),
    )
    results: dict[str, Any] = {}
    stage_iterator = track_progress(
        stages,
        config=config,
        description="完整流程",
        total=len(stages),
        unit="阶段",
    )
    for name, pipeline in stage_iterator:
        results[name] = pipeline(config)
    return results


def run_task_pipeline(config: ProjectConfig) -> dict[str, Any]:
    """执行当前赛段允许的训练、验证、预测及可选优化阶段。"""

    compliance = config.raw.get("competition_compliance", {})
    preliminary = (
        isinstance(compliance, Mapping)
        and compliance.get("stage") == "preliminary"
    )
    if preliminary:
        # 初赛默认入口必须先完成训练期搜索和原始标签门控，不能直接提交未选择的基线。
        stages = [
            ("tune", tune_pipeline),
            ("validate", validate_pipeline),
            ("predict", predict_pipeline),
        ]
    else:
        stages = [
            ("train", train_pipeline),
            ("validate", validate_pipeline),
            ("predict", predict_pipeline),
        ]
    if not preliminary:
        stages.append(("optimize", optimize_pipeline))
    results: dict[str, Any] = {}
    stage_iterator = track_progress(
        stages,
        config=config,
        description="预测任务",
        total=len(stages),
        unit="阶段",
    )
    for name, pipeline in stage_iterator:
        results[name] = pipeline(config)
    return results
