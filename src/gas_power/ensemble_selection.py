"""基于训练期 OOF 预测的非负融合、留一折检验和候选门控。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateGate:
    """候选模型进入正式融合所需的原始标签门槛。"""

    passed: bool
    recent_mean_gain: float
    recent_worst_gain: float
    improved_folds: int
    target_gains: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "recent_mean_gain": self.recent_mean_gain,
            "recent_worst_gain": self.recent_worst_gain,
            "improved_folds": self.improved_folds,
            "target_gains": self.target_gains,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ColumnGate:
    """单个目标×步长候选相对最后值基线的稳定性门槛。"""

    passed: bool
    mean_gain: float
    worst_gain: float
    non_degraded_folds: int
    total_folds: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "mean_gain": self.mean_gain,
            "worst_gain": self.worst_gain,
            "non_degraded_folds": self.non_degraded_folds,
            "total_folds": self.total_folds,
            "reasons": list(self.reasons),
        }


def mape_sample_weights(
    y_true: Sequence[float],
    *,
    floor_quantile: float = 0.01,
    minimum_floor: float = 1.0e-6,
) -> np.ndarray:
    """返回近似最小化 MAPE 的有限样本权重。"""

    actual = np.abs(np.asarray(y_true, dtype=float))
    finite = np.isfinite(actual)
    positive = actual[finite & (actual > 0.0)]
    floor = float(np.quantile(positive, floor_quantile)) if positive.size else 1.0
    floor = max(floor, float(minimum_floor))
    weights = np.zeros_like(actual, dtype=float)
    weights[finite] = 1.0 / np.maximum(actual[finite], floor)
    normalizer = float(np.mean(weights[finite])) if finite.any() else 1.0
    return weights / max(normalizer, minimum_floor)


def _nnls_weights(actual: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[1] == 0:
        raise ValueError("OOF 预测矩阵必须为二维且至少包含一个候选")
    denominator = np.maximum(np.abs(actual), 1.0e-6)
    scaled_actual = actual / denominator
    scaled_predictions = predictions / denominator[:, None]
    try:
        from scipy.optimize import nnls

        weights, _ = nnls(scaled_predictions, scaled_actual)
    except ImportError:
        weights = np.maximum(
            np.linalg.lstsq(scaled_predictions, scaled_actual, rcond=None)[0],
            0.0,
        )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(predictions.shape[1], dtype=float)
    return weights / float(weights.sum())


def fit_oof_weights(
    oof: pd.DataFrame,
    model_names: Sequence[str],
    *,
    target_column: str,
    horizon: int,
    fold_column: str = "fold",
) -> tuple[np.ndarray, pd.DataFrame]:
    """按目标×步长拟合非负、和为一的 OOF 权重，并返回留一折结果。"""

    required = {fold_column, "target", "horizon_steps", "y_true", *model_names}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF 数据缺少字段: {sorted(missing)}")
    subset = oof.loc[
        (oof["target"].astype(str) == str(target_column))
        & (oof["horizon_steps"].astype(int) == int(horizon))
    ].copy()
    if subset.empty:
        raise ValueError(f"OOF 缺少 {target_column} t+{horizon} 的样本")
    values = subset[["y_true", *model_names]].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    subset = subset.loc[finite]
    actual = subset["y_true"].to_numpy(dtype=float)
    matrix = subset[list(model_names)].to_numpy(dtype=float)
    weights = _nnls_weights(actual, matrix)
    loo_rows: list[dict[str, object]] = []
    for fold, held_out in subset.groupby(fold_column, sort=True):
        train = subset.loc[subset[fold_column] != fold]
        if train.empty:
            continue
        loo = _nnls_weights(
            train["y_true"].to_numpy(dtype=float),
            train[list(model_names)].to_numpy(dtype=float),
        )
        pred = held_out[list(model_names)].to_numpy(dtype=float) @ loo
        denominator = np.maximum(np.abs(held_out["y_true"].to_numpy(dtype=float)), 1.0e-6)
        loo_rows.append(
            {
                "fold": fold,
                "mape": float(np.mean(np.abs(pred - held_out["y_true"].to_numpy(dtype=float)) / denominator)),
                "weights": loo.tolist(),
            }
        )
    return weights, pd.DataFrame(loo_rows)


def apply_oof_weights(
    predictions: Mapping[str, np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    names = list(predictions)
    if len(names) != len(weights):
        raise ValueError("融合候选数量与权重数量不一致")
    normalized = np.asarray(weights, dtype=float)
    if (normalized < 0).any() or not np.isfinite(normalized).all() or normalized.sum() <= 0:
        raise ValueError("融合权重必须是有限非负数且和大于零")
    normalized = normalized / normalized.sum()
    matrix = np.column_stack([np.asarray(predictions[name], dtype=float) for name in names])
    output = matrix @ normalized
    if not np.isfinite(output).all():
        raise ValueError("融合预测包含缺失值或无穷值")
    return output


def evaluate_oof_column_gate(
    oof: pd.DataFrame,
    *,
    target_column: str,
    horizon: int,
    candidate_column: str,
    baseline_column: str = "last_value",
    fold_column: str = "fold",
    minimum_mean_gain: float = 0.0,
    maximum_worst_degradation: float = 0.01,
    minimum_non_degraded_folds: int = 5,
) -> ColumnGate:
    """按单列和时间折判断候选是否有资格进入 OOF 融合。"""

    required = {
        fold_column,
        "target",
        "horizon_steps",
        "y_true",
        baseline_column,
        candidate_column,
    }
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"逐列门控数据缺少字段: {sorted(missing)}")
    subset = oof.loc[
        (oof["target"].astype(str) == str(target_column))
        & (oof["horizon_steps"].astype(int) == int(horizon)),
        [fold_column, "y_true", baseline_column, candidate_column],
    ].copy()
    values = subset[["y_true", baseline_column, candidate_column]].to_numpy(dtype=float)
    subset = subset.loc[np.isfinite(values).all(axis=1)]
    if subset.empty:
        return ColumnGate(False, float("-inf"), float("-inf"), 0, 0, ("没有有限 OOF 样本",))
    denominator = np.maximum(np.abs(subset["y_true"].to_numpy(dtype=float)), 1.0e-6)
    subset["baseline_ape"] = (
        np.abs(subset[baseline_column] - subset["y_true"]) / denominator
    )
    subset["candidate_ape"] = (
        np.abs(subset[candidate_column] - subset["y_true"]) / denominator
    )
    folds = subset.groupby(fold_column, sort=True).agg(
        baseline_mape=("baseline_ape", "mean"),
        candidate_mape=("candidate_ape", "mean"),
    )
    folds["gain"] = folds["baseline_mape"] - folds["candidate_mape"]
    mean_gain = float(folds["gain"].mean())
    worst_gain = float(folds["gain"].min())
    non_degraded = int((folds["gain"] >= 0.0).sum())
    required_folds = min(int(minimum_non_degraded_folds), len(folds))
    reasons: list[str] = []
    if mean_gain <= float(minimum_mean_gain):
        reasons.append("平均 MAPE 未改善")
    if non_degraded < required_folds:
        reasons.append(f"不退化折数不足 {required_folds}")
    if worst_gain < -float(maximum_worst_degradation):
        reasons.append(f"最差折退化超过 {maximum_worst_degradation:.4f}")
    return ColumnGate(
        passed=not reasons,
        mean_gain=mean_gain,
        worst_gain=worst_gain,
        non_degraded_folds=non_degraded,
        total_folds=len(folds),
        reasons=tuple(reasons),
    )


def evaluate_candidate_gate(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    target_column: str = "target",
    fold_column: str = "fold",
    minimum_mean_gain: float = 0.003,
    maximum_worst_degradation: float = 0.005,
    minimum_improved_folds: int = 7,
) -> CandidateGate:
    """使用原始标签比较候选和最后值基线，返回可审计门控结果。"""

    keys = [fold_column, target_column, "horizon_steps"]
    for optional in ("origin", "target_datetime"):
        if optional in baseline.columns and optional in candidate.columns:
            keys.append(optional)
    merged = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"))
    if merged.empty:
        raise ValueError("基线和候选没有可比较的 OOF 样本")
    denominator = np.maximum(np.abs(merged["y_true_baseline"].to_numpy(dtype=float)), 1.0e-6)
    merged["baseline_ape"] = np.abs(merged["y_pred_baseline"] - merged["y_true_baseline"]) / denominator
    merged["candidate_ape"] = np.abs(merged["y_pred_candidate"] - merged["y_true_candidate"]) / denominator
    fold_scores = merged.groupby(fold_column).agg(
        baseline_mape=("baseline_ape", "mean"), candidate_mape=("candidate_ape", "mean")
    )
    fold_scores["gain"] = fold_scores["baseline_mape"] - fold_scores["candidate_mape"]
    target_scores = merged.groupby(target_column).agg(
        baseline_mape=("baseline_ape", "mean"), candidate_mape=("candidate_ape", "mean")
    )
    target_scores["gain"] = target_scores["baseline_mape"] - target_scores["candidate_mape"]
    recent_mean_gain = float(fold_scores["gain"].mean())
    recent_worst_gain = float(fold_scores["gain"].min())
    improved_folds = int((fold_scores["gain"] > 0.0).sum())
    target_gains = {str(index): float(value) for index, value in target_scores["gain"].items()}
    reasons: list[str] = []
    if recent_mean_gain < minimum_mean_gain:
        reasons.append(f"最近折平均改善不足 {minimum_mean_gain:.4f}")
    if any(value < 0.0 for value in target_gains.values()):
        reasons.append("至少一个目标退化")
    if improved_folds < minimum_improved_folds:
        reasons.append(f"改善折数不足 {minimum_improved_folds}")
    if recent_worst_gain < -maximum_worst_degradation:
        reasons.append(f"最差折退化超过 {maximum_worst_degradation:.4f}")
    return CandidateGate(
        passed=not reasons,
        recent_mean_gain=recent_mean_gain,
        recent_worst_gain=recent_worst_gain,
        improved_folds=improved_folds,
        target_gains=target_gains,
        reasons=tuple(reasons),
    )


def project_forecasts(
    predictions: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    horizons: Sequence[int],
    capacities: Mapping[str, float] | None = None,
    enforce_target_consistency: bool = True,
) -> pd.DataFrame:
    """执行非负、容量和 generator_1 不超过 generator_all 的投影。"""

    output = predictions.copy()
    for column in output.columns:
        if column.endswith("_pred"):
            output[column] = pd.to_numeric(output[column], errors="coerce").clip(lower=0.0)
    if capacities:
        for target, capacity in capacities.items():
            for horizon in horizons:
                column = f"{target}_t+{int(horizon) * 15}_pred"
                if column in output:
                    output[column] = output[column].clip(upper=float(capacity))
    if enforce_target_consistency and {"generator_1", "generator_all"}.issubset(target_columns):
        for horizon in horizons:
            one = f"generator_1_t+{int(horizon) * 15}_pred"
            total = f"generator_all_t+{int(horizon) * 15}_pred"
            if one in output and total in output:
                output[one] = np.minimum(output[one], output[total])
    return output
