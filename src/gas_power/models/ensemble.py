"""基线与学习模型的可配置加权融合。"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gas_power.models.base import ForecastModel


class WeightedEnsembleModel(ForecastModel):
    def __init__(
        self,
        components: Sequence[tuple[ForecastModel, float]],
        clip_min: float | None = None,
    ):
        if not components:
            raise ValueError("融合模型至少需要一个子模型")
        if any(float(weight) < 0.0 for _, weight in components):
            raise ValueError("融合权重不能为负数")
        weight_sum = sum(float(weight) for _, weight in components)
        if weight_sum <= 0.0:
            raise ValueError("融合权重之和必须大于 0")
        self.components = [
            (model, float(weight) / weight_sum) for model, weight in components
        ]
        self.clip_min = clip_min

    def fit(
        self,
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        horizons: Sequence[int],
        train_end: pd.Timestamp | None = None,
        *,
        raw_targets: pd.DataFrame | None = None,
        feature_matrix: pd.DataFrame | None = None,
        data_source: str = "training",
    ) -> "WeightedEnsembleModel":
        for model, _ in self.components:
            model.fit(
                frame,
                target_columns,
                horizons,
                train_end=train_end,
                raw_targets=raw_targets,
                feature_matrix=feature_matrix,
                data_source=data_source,
            )
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        weighted: pd.DataFrame | None = None
        for model, weight in self.components:
            prediction = model.predict(frame, origins, target_columns, horizons)
            weighted = prediction * weight if weighted is None else weighted.add(prediction * weight)
        if weighted is None:
            raise RuntimeError("融合模型没有产生预测")
        if self.clip_min is not None:
            weighted = weighted.clip(lower=float(self.clip_min))
        if not np.isfinite(weighted.to_numpy(dtype=float)).all():
            raise ValueError("融合结果包含缺失值或无穷值")
        return weighted


class HorizonWeightedEnsembleModel(ForecastModel):
    """按目标×步长使用训练期 OOF 学得的非负权重。"""

    def __init__(
        self,
        components: Mapping[str, ForecastModel],
        column_weights: Mapping[str, Mapping[str, float]],
    ):
        if not components:
            raise ValueError("OOF 融合至少需要一个子模型")
        self.components = dict(components)
        self.column_weights = {
            str(column): {str(name): float(weight) for name, weight in weights.items()}
            for column, weights in column_weights.items()
        }
        for column, weights in self.column_weights.items():
            unknown = set(weights).difference(self.components)
            if unknown:
                raise ValueError(f"融合列 {column} 引用了未知模型: {sorted(unknown)}")
            if any(weight < 0.0 for weight in weights.values()) or sum(weights.values()) <= 0.0:
                raise ValueError(f"融合列 {column} 的权重必须非负且和大于零")

    def fit(
        self,
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        horizons: Sequence[int],
        train_end: pd.Timestamp | None = None,
        *,
        raw_targets: pd.DataFrame | None = None,
        feature_matrix: pd.DataFrame | None = None,
        data_source: str = "training",
    ) -> "HorizonWeightedEnsembleModel":
        for model in self.components.values():
            model.fit(
                frame,
                target_columns,
                horizons,
                train_end=train_end,
                raw_targets=raw_targets,
                feature_matrix=feature_matrix,
                data_source=data_source,
            )
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        predictions = {
            name: model.predict(frame, origins, target_columns, horizons)
            for name, model in self.components.items()
        }
        expected_columns = next(iter(predictions.values())).columns
        output = pd.DataFrame(index=origins)
        output.index.name = "datetime"
        for column in expected_columns:
            weights = self.column_weights.get(str(column))
            if not weights:
                raise ValueError(f"缺少预测列的 OOF 融合权重: {column}")
            total = float(sum(weights.values()))
            output[column] = sum(
                predictions[name][column] * (weight / total)
                for name, weight in weights.items()
            )
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("OOF 融合结果包含缺失值或无穷值")
        return output
