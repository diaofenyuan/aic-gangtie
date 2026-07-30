"""可配置评分公式；未确认的权重不会被隐式当作官方规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreResult:
    prediction_score: float
    final_score: float
    formula: str
    display_scale: float
    component_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_score": self.prediction_score,
            "final_score": self.final_score,
            "formula": self.formula,
            "display_scale": self.display_scale,
            "component_scores": self.component_scores,
        }


class ConfigurableScorer:
    """支持原始 1-MAPE、百分制、目标/步长加权及数据处理组合。"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.formula = str(config.get("formula", "one_minus_mape"))
        if self.formula not in {"one_minus_mape", "one_minus_mape_percent"}:
            raise ValueError(f"不支持的评分公式: {self.formula}")

    def score(self, tidy: pd.DataFrame, data_processing_score: float | None = None) -> ScoreResult:
        required = {"target", "horizon_steps", "y_true", "y_pred"}
        if missing := required.difference(tidy.columns):
            raise ValueError(f"评分明细缺少字段: {sorted(missing)}")
        target_weights = self.config.get("target_weights", {})
        horizon_weights = self.config.get("horizon_weights", {})
        mode = str(self.config.get("horizon_weight_mode", "equal"))
        rows: list[tuple[str, int, float, float]] = []
        for (target, horizon), group in tidy.groupby(["target", "horizon_steps"], sort=True):
            actual = group["y_true"].to_numpy(dtype=float)
            predicted = group["y_pred"].to_numpy(dtype=float)
            finite = np.isfinite(actual) & np.isfinite(predicted)
            actual, predicted = actual[finite], predicted[finite]
            if len(actual) == 0:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                mape = float(np.mean(np.abs(actual - predicted) / np.abs(actual)))
            target_weight = float(target_weights.get(str(target), 1.0))
            horizon_weight = (
                float(horizon_weights.get(str(int(horizon)), 1.0))
                if mode == "custom"
                else 1.0
            )
            rows.append((str(target), int(horizon), 1.0 - mape, target_weight * horizon_weight))
        if not rows:
            raise ValueError("没有可评分的有限预测对")
        weights = np.asarray([item[3] for item in rows], dtype=float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            raise ValueError("评分权重必须为有限非负数且总和大于 0")
        raw_score = float(np.average([item[2] for item in rows], weights=weights))
        scale = 100.0 if self.formula.endswith("percent") else float(
            self.config.get("display_scale", 1.0)
        )
        prediction_score = raw_score * scale
        components = {
            f"{target}_h{horizon}": float(score * scale)
            for target, horizon, score, _ in rows
        }
        final_score = prediction_score
        data_config = self.config.get("data_processing", {})
        if isinstance(data_config, Mapping) and bool(data_config.get("enabled", False)):
            if data_processing_score is None:
                raise ValueError("已启用数据处理组合评分，但未提供 data_processing_score")
            prediction_weight = data_config.get("prediction_weight")
            data_weight = data_config.get("data_weight")
            if prediction_weight is None or data_weight is None:
                raise ValueError("组合评分权重尚未确认，禁止自行假定")
            final_score = (
                prediction_score * float(prediction_weight)
                + float(data_processing_score) * float(data_weight)
            )
        return ScoreResult(prediction_score, final_score, self.formula, scale, components)
