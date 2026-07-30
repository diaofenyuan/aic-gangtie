"""在相同时间切分上比较强基线并按目标/步长选择策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gas_power.metrics import official_mape, summarize_predictions
from gas_power.models.factory import baseline_from_spec
from gas_power.validation import TimeSeriesRollingSplitter, run_rolling_validation


@dataclass
class BenchmarkArtifacts:
    metrics: pd.DataFrame
    best: pd.DataFrame
    predictions: pd.DataFrame
    reachability: dict[str, Any]
    alignment_diagnostics: pd.DataFrame


def _alignment_diagnostics(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    interval_minutes: int,
    near_zero_threshold: float,
) -> pd.DataFrame:
    """用验证标签检查整体错位；非零 offset 结果绝不参与模型选择。"""

    last = predictions.loc[predictions["baseline"] == "last_value"]
    rows: list[dict[str, Any]] = []
    for offset in (-2, -1, 0, 1, 2):
        actual: list[float] = []
        predicted: list[float] = []
        for row in last.itertuples():
            candidate_time = pd.Timestamp(row.target_datetime) + pd.Timedelta(
                minutes=offset * interval_minutes
            )
            if candidate_time not in frame.index:
                continue
            value = frame.at[candidate_time, str(row.target)]
            if pd.notna(value) and np.isfinite(float(row.y_pred)):
                actual.append(float(value))
                predicted.append(float(row.y_pred))
        if not actual:
            continue
        metric = official_mape(
            actual, predicted, near_zero_threshold=near_zero_threshold, emit_warning=False
        )
        rows.append(
            {
                "label_offset_steps": offset,
                "label_offset_minutes": offset * interval_minutes,
                "sample_count": metric.sample_count,
                "mape": metric.mape,
                "score_1_mape": metric.score_1_mape,
                "score_percent": metric.score_1_mape * 100.0,
                "near_zero_count": metric.near_zero_count,
                "risk": "red_future_diagnostic_only" if offset > 0 else (
                    "amber_wrong_alignment" if offset < 0 else "official_alignment"
                ),
                "used_for_model_selection": False if offset != 0 else True,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_target_strategy(
    predictions: pd.DataFrame,
    near_zero_threshold: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for baseline, group in predictions.groupby("baseline", sort=True):
        metrics = summarize_predictions(group, near_zero_threshold)
        target = metrics.loc[metrics["scope"] == "target"].copy()
        target["baseline"] = str(baseline)
        rows.append(target)
    if not rows:
        return pd.DataFrame()
    candidates = pd.concat(rows, ignore_index=True)
    best = (
        candidates.sort_values(["target", "mape", "rmse", "baseline"])
        .groupby("target", as_index=False)
        .first()
    )
    best["selection_scope"] = "target"
    return best


def run_baseline_benchmark(
    frame: pd.DataFrame,
    baseline_specs: Sequence[Mapping[str, Any]],
    splitter: TimeSeriesRollingSplitter,
    target_columns: Sequence[str],
    horizons: Sequence[int],
    interval_minutes: int,
    near_zero_threshold: float,
    reachability_config: Mapping[str, Any],
    show_progress: bool = False,
) -> BenchmarkArtifacts:
    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    baseline_iterator = tqdm(
        baseline_specs,
        total=len(baseline_specs),
        desc="基线评测",
        unit="模型",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for spec in baseline_iterator:
        name = str(spec.get("name", spec.get("type", "baseline")))
        baseline_iterator.set_postfix_str(name)
        artifacts = run_rolling_validation(
            frame=frame,
            model_factory=lambda spec=spec: baseline_from_spec(spec, interval_minutes),
            splitter=splitter,
            target_columns=target_columns,
            horizons=horizons,
            interval_minutes=interval_minutes,
            near_zero_threshold=near_zero_threshold,
            worst_error_count=0,
            feature_builder=None,
        )
        metrics = artifacts.metrics.copy()
        metrics.insert(0, "baseline", name)
        predictions = artifacts.predictions.copy()
        predictions.insert(0, "baseline", name)
        metric_parts.append(metrics)
        prediction_parts.append(predictions)
    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    horizon_candidates = metrics.loc[metrics["scope"] == "target_horizon"].copy()
    horizon_best = (
        horizon_candidates.sort_values(
            ["target", "horizon_steps", "mape", "rmse", "baseline"]
        )
        .groupby(["target", "horizon_steps"], as_index=False)
        .first()
    )
    horizon_best["selection_scope"] = "target_horizon"
    target_best = _aggregate_target_strategy(predictions, near_zero_threshold)
    best = pd.concat([horizon_best, target_best], ignore_index=True, sort=False)

    last_overall = metrics.loc[
        (metrics["baseline"] == "last_value") & (metrics["scope"] == "overall"),
        "score_1_mape",
    ]
    last_percent = float(last_overall.iloc[0] * 100.0) if len(last_overall) else float("nan")
    best_overall = metrics.loc[metrics["scope"] == "overall"].sort_values("mape").iloc[0]
    stable_threshold = float(reachability_config.get("stable_score_percent", 99.5))
    leaderboard_threshold = float(
        reachability_config.get("leaderboard_score_percent", 99.9)
    )
    messages: list[str] = []
    if np.isfinite(last_percent) and last_percent >= stable_threshold:
        messages.append("最后值保持已超过稳定性阈值，短周期负荷可能高度稳定。")
    if float(best_overall["score_1_mape"]) * 100.0 < leaderboard_threshold:
        messages.append("合法基线尚未接近榜首区间；任何依赖未来偏移的提升都必须按违规风险处理。")
    alignment = _alignment_diagnostics(
        frame, predictions, interval_minutes, near_zero_threshold
    )
    correct_score = alignment.loc[
        alignment["label_offset_steps"] == 0, "score_percent"
    ]
    wrong_scores = alignment.loc[
        alignment["label_offset_steps"] != 0, "score_percent"
    ]
    alignment_gain = (
        float(wrong_scores.max() - correct_score.iloc[0])
        if len(correct_score) and len(wrong_scores)
        else float("nan")
    )
    if np.isfinite(alignment_gain) and alignment_gain >= float(
        reachability_config.get("alignment_gain_percent", 1.0)
    ):
        messages.append("标签整体平移会显著改变基线得分，应优先核对预测起点和目标列口径。")
    reachability = {
        "last_value_score_percent": last_percent,
        "best_baseline": str(best_overall["baseline"]),
        "best_score_percent": float(best_overall["score_1_mape"]) * 100.0,
        "thresholds": dict(reachability_config),
        "best_wrong_alignment_gain_percent": alignment_gain,
        "messages": messages,
        "synthetic_only": True,
    }
    return BenchmarkArtifacts(metrics, best, predictions, reachability, alignment)
