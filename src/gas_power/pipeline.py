"""数据、预测、验证与优化的可复用命令流水线。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gas_power.availability import FeatureAvailabilityRegistry
from gas_power.audit import calculate_lag_correlations, run_data_audit
from gas_power.benchmark import run_baseline_benchmark
from gas_power.config import ProjectConfig, configured_value
from gas_power.data import ConfiguredDataLoader, write_time_frame
from gas_power.features import (
    CausalFeatureBuilder,
    assert_feature_causality,
    assert_shift_before_rolling,
)
from gas_power.models import build_model
from gas_power.models.base import prediction_column, prediction_columns
from gas_power.models.baselines import LastValueModel
from gas_power.models.factory import baseline_from_spec
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
from gas_power.gpu_gate import evaluate_gpu_gate, evaluate_residual_gate
from gas_power.relations import discover_relations
from gas_power.runtime import progress_enabled, track_progress
from gas_power.scoring import ConfigurableScorer
from gas_power.submission import validate_submission_bundle
from gas_power.synthetic import SYNTHETIC_WARNING, generate_synthetic_dataset
from gas_power.validation import (
    run_rolling_validation,
    splitter_from_config,
    summarize_detailed_validation,
)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)


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
    config.ensure_runtime_dirs()
    with tqdm(
        total=3,
        desc="数据准备",
        unit="步",
        dynamic_ncols=True,
        leave=False,
        disable=not progress_enabled(config),
    ) as data_progress:
        frame, quality = ConfiguredDataLoader(config).load()
        data_progress.set_postfix_str("特征构建")
        data_progress.update(1)
        builder = _feature_builder(config)
        features = builder.transform(frame)
        data_progress.set_postfix_str("写入缓存")
        data_progress.update(1)
        cache_dir = config.path("cache")
        timestamp_format = str(config.section("data")["timestamp_format"])
        write_time_frame(frame, cache_dir / "processed.csv", timestamp_format)
        write_time_frame(features, cache_dir / "features.csv", timestamp_format)
        quality_path = cache_dir / "data_quality.json"
        quality.write_json(quality_path)
        data_progress.update(1)
    return frame, features, quality.to_dict()


def generate_synthetic_pipeline(config: ProjectConfig) -> dict[str, Any]:
    config.ensure_runtime_dirs()
    return generate_synthetic_dataset(config)


def train_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, features, quality = prepare_data(config)
    roles = config.section("data")["roles"]
    targets = [str(value) for value in roles["targets"]]
    long_steps = int(config.section("forecast")["long_steps"])
    horizons = list(range(1, long_steps + 1))
    _assert_final_residual_gate(config)
    model = build_model(config)
    train_end = pd.Timestamp(frame.index.max())
    model.fit(frame, targets, horizons, train_end=train_end)
    model_path = config.path("models") / "forecast_model.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "model_type": type(model).__name__,
        "train_start": str(frame.index.min()),
        "train_end": str(train_end),
        "training_rows": len(frame),
        "targets": targets,
        "horizons": [1, long_steps],
        "seed": config.seed,
        "synthetic_data_warning": SYNTHETIC_WARNING,
        "quality_report": quality,
        "feature_columns": list(features.columns),
        "feature_source_whitelist": sorted(
            FeatureAvailabilityRegistry.from_config(config).allowed_source_columns("long")
        ),
    }
    _write_json(metadata, config.path("models") / "forecast_model_metadata.json")
    return metadata


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
    raise ValueError(f"不支持的预测起点模式: {mode}")


def predict_pipeline(config: ProjectConfig) -> dict[str, Any]:
    inference_started = time.perf_counter()
    _assert_final_residual_gate(config)
    frame, features, _ = prepare_data(config)
    model_path = config.path("models") / "forecast_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在，请先运行 train: {model_path}")
    model = joblib.load(model_path)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    forecast_config = config.section("forecast")
    short_horizons = list(range(1, int(forecast_config["short_steps"]) + 1))
    long_horizons = list(range(1, int(forecast_config["long_steps"]) + 1))
    interval_minutes = int(config.section("optimization").get("interval_minutes", 15))
    timestamp_format = str(config.section("data")["timestamp_format"])
    origins = _prediction_origins(config, frame)

    raw_long_prediction = model.predict(frame, origins, targets, long_horizons)
    post_config = config.raw.get("postprocessing", {})
    post_result = PhysicalForecastPostprocessor(post_config, interval_minutes).apply(
        raw_long_prediction, frame, origins, targets, long_horizons
    )
    long_prediction = (
        post_result.predictions
        if bool(post_config.get("enabled", True))
        else raw_long_prediction
    )
    short_prediction = long_prediction[prediction_columns(targets, short_horizons, interval_minutes)]
    results_dir = config.path("results")
    raw_short = raw_long_prediction[
        prediction_columns(targets, short_horizons, interval_minutes)
    ]
    write_time_frame(raw_short, results_dir / "raw_s_result.csv", timestamp_format)
    write_time_frame(raw_long_prediction, results_dir / "raw_l_result.csv", timestamp_format)
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
    long_output = write_forecast_csv(
        long_prediction,
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

    registry = FeatureAvailabilityRegistry.from_config(config)
    approved_sources = sorted(
        set(frame.columns).intersection(registry.allowed_source_columns("long"))
    )
    input_frame = pd.concat([frame[approved_sources], features], axis=1).loc[origins]
    if input_frame.columns.duplicated().any():
        raise ValueError("input.csv 构建后存在重复字段")
    write_time_frame(input_frame, results_dir / "input.csv", timestamp_format)
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
    return {
        "origins": len(origins),
        "short_shape": list(short_output.shape),
        "long_shape": list(long_output.shape),
        "s_result": str(results_dir / "s_result.csv"),
        "l_result": str(results_dir / "l_result.csv"),
        "input": str(results_dir / "input.csv"),
        "postprocessing_adjusted_cells": post_result.adjusted_cells,
        "postprocessing_adjustments": post_result.adjustments,
        "runtime": runtime_report,
    }


def validate_pipeline(config: ProjectConfig) -> dict[str, Any]:
    frame, _, _ = prepare_data(config)
    builder = _feature_builder(config)
    cutoff = pd.Timestamp(frame.index[len(frame) // 2])
    assert_feature_causality(builder, frame, cutoff)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    rolling_windows = [int(value) for value in config.section("features")["rolling_windows"]]
    assert_shift_before_rolling(builder, frame, targets[0], rolling_windows[0])

    forecast = config.section("forecast")
    horizons = list(range(1, int(forecast["long_steps"]) + 1))
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
        show_progress=progress_enabled(config),
        progress_description="滚动验证",
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
        "warning": SYNTHETIC_WARNING,
        "leakage_checks": "passed",
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
        "warning": SYNTHETIC_WARNING,
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
        "warning": SYNTHETIC_WARNING,
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
    return {**artifacts.summary, "reports": str(reports), "warning": SYNTHETIC_WARNING}


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
        "warning": SYNTHETIC_WARNING,
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
    """使用 data 中的现有数据执行训练、验证、预测和优化。"""

    stages = (
        ("train", train_pipeline),
        ("validate", validate_pipeline),
        ("predict", predict_pipeline),
        ("optimize", optimize_pipeline),
    )
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
