"""官方预测指标及分组误差汇总。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricValue:
    sample_count: int
    mape: float
    score_1_mape: float
    near_zero_count: int
    zero_count: int
    mae: float
    rmse: float
    bias: float


def official_mape(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    near_zero_threshold: float = 1.0e-6,
    emit_warning: bool = True,
) -> MetricValue:
    """严格按 |y-y_hat|/|y| 求均值，不用 epsilon 改写官方分母。"""

    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true 与 y_pred 形状不一致")
    if actual.size == 0:
        raise ValueError("指标输入不能为空")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("官方指标输入必须为有限数值")

    absolute_actual = np.abs(actual)
    near_zero = absolute_actual <= float(near_zero_threshold)
    exact_zero = absolute_actual == 0.0
    if emit_warning and near_zero.any():
        warnings.warn(
            f"真实值接近零的样本有 {int(near_zero.sum())} 条，其中严格为零 "
            f"{int(exact_zero.sum())} 条；官方 MAPE 未使用 epsilon 修正，结果可能为 inf/nan。",
            RuntimeWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(actual - predicted) / absolute_actual
        mape = float(np.mean(ape))
    return MetricValue(
        sample_count=int(actual.size),
        mape=mape,
        score_1_mape=float(1.0 - mape),
        near_zero_count=int(near_zero.sum()),
        zero_count=int(exact_zero.sum()),
        mae=float(np.mean(np.abs(predicted - actual))),
        rmse=float(np.sqrt(np.mean(np.square(predicted - actual)))),
        bias=float(np.mean(predicted - actual)),
    )


def summarize_predictions(
    tidy: pd.DataFrame,
    near_zero_threshold: float,
) -> pd.DataFrame:
    """分别给出目标×步长、目标、步长及整体指标。"""

    required = {"target", "horizon_steps", "y_true", "y_pred"}
    missing = required.difference(tidy.columns)
    if missing:
        raise ValueError(f"误差明细缺少字段: {sorted(missing)}")
    finite = np.isfinite(tidy["y_true"].to_numpy(dtype=float)) & np.isfinite(
        tidy["y_pred"].to_numpy(dtype=float)
    )
    excluded = int((~finite).sum())
    valid = tidy.loc[finite].copy()
    if valid.empty:
        raise ValueError("没有可计算指标的有限标签-预测对")

    rows: list[dict[str, object]] = []

    def append_metric(scope: str, target: str, horizon: int, group: pd.DataFrame) -> None:
        metric = official_mape(
            group["y_true"],
            group["y_pred"],
            near_zero_threshold=near_zero_threshold,
            emit_warning=False,
        )
        rows.append(
            {
                "scope": scope,
                "target": target,
                "horizon_steps": horizon,
                "horizon_minutes": horizon * 15 if horizon >= 0 else -1,
                "sample_count": metric.sample_count,
                "mape": metric.mape,
                "score_1_mape": metric.score_1_mape,
                "near_zero_count": metric.near_zero_count,
                "zero_count": metric.zero_count,
                "mae": metric.mae,
                "rmse": metric.rmse,
                "bias": metric.bias,
                "excluded_non_finite_pairs": excluded if scope == "overall" else 0,
            }
        )

    for (target, horizon), group in valid.groupby(["target", "horizon_steps"], sort=True):
        append_metric("target_horizon", str(target), int(horizon), group)
    for target, group in valid.groupby("target", sort=True):
        append_metric("target", str(target), -1, group)
    for horizon, group in valid.groupby("horizon_steps", sort=True):
        append_metric("horizon", "__all__", int(horizon), group)
    append_metric("overall", "__all__", -1, valid)

    overall = rows[-1]
    if int(overall["near_zero_count"]) > 0:
        warnings.warn(
            f"验证集中真实值接近零的标签-预测对有 {overall['near_zero_count']} 条；"
            "报告保留官方未平滑 MAPE。",
            RuntimeWarning,
            stacklevel=2,
        )
    return pd.DataFrame(rows)


def largest_errors(tidy: pd.DataFrame, count: int = 50) -> pd.DataFrame:
    output = tidy.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        output["absolute_percentage_error"] = (
            np.abs(output["y_true"] - output["y_pred"]) / np.abs(output["y_true"])
        )
    output["absolute_error"] = np.abs(output["y_true"] - output["y_pred"])
    return output.sort_values(
        ["absolute_percentage_error", "absolute_error"], ascending=False
    ).head(int(count))
