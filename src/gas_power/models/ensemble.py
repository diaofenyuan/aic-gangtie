"""基线与学习模型的可配置加权融合。"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gas_power.models.base import ForecastModel, prediction_column
from gas_power.regimes import infer_regimes_from_history, prediction_column_targets


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
            weighted = (
                prediction * weight
                if weighted is None
                else weighted.add(prediction * weight)
            )
        if weighted is None:
            raise RuntimeError("融合模型没有产生预测")
        if self.clip_min is not None:
            weighted = weighted.clip(lower=float(self.clip_min))
        if not np.isfinite(weighted.to_numpy(dtype=float)).all():
            raise ValueError("融合结果包含缺失值或无穷值")
        return weighted


class HorizonWeightedEnsembleModel(ForecastModel):
    """按目标、步长及可选工况使用训练期 OOF 学得的非负权重。"""

    def __init__(
        self,
        components: Mapping[str, ForecastModel],
        column_weights: Mapping[str, Mapping[str, float]],
        *,
        regime_column_weights: Mapping[str, Mapping[str, Mapping[str, float]]]
        | None = None,
        online_threshold_mw: float = 1.0,
        stable_delta_threshold_mw: float = 2.0,
    ):
        if not components:
            raise ValueError("OOF 融合至少需要一个子模型")
        self.components = dict(components)
        self.column_weights = {
            str(column): {str(name): float(weight) for name, weight in weights.items()}
            for column, weights in column_weights.items()
        }
        self.regime_column_weights = {
            str(column): {
                str(regime): {
                    str(name): float(weight) for name, weight in weights.items()
                }
                for regime, weights in regimes.items()
            }
            for column, regimes in (regime_column_weights or {}).items()
        }
        self.online_threshold_mw = float(online_threshold_mw)
        self.stable_delta_threshold_mw = float(stable_delta_threshold_mw)
        for column, weights in self.column_weights.items():
            unknown = set(weights).difference(self.components)
            if unknown:
                raise ValueError(f"融合列 {column} 引用了未知模型: {sorted(unknown)}")
            if (
                any(weight < 0.0 for weight in weights.values())
                or sum(weights.values()) <= 0.0
            ):
                raise ValueError(f"融合列 {column} 的权重必须非负且和大于零")
        for column, regimes in self.regime_column_weights.items():
            if column not in self.column_weights:
                raise ValueError(f"工况融合列 {column} 缺少全局回退权重")
            for regime, weights in regimes.items():
                unknown = set(weights).difference(self.components)
                if unknown:
                    raise ValueError(
                        f"工况融合列 {column}/{regime} 引用了未知模型: {sorted(unknown)}"
                    )
                if (
                    any(weight < 0.0 for weight in weights.values())
                    or sum(weights.values()) <= 0.0
                ):
                    raise ValueError(
                        f"工况融合列 {column}/{regime} 的权重必须非负且和大于零"
                    )

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
        column_targets = prediction_column_targets(target_columns, horizons)
        regimes_by_target = {
            str(target): infer_regimes_from_history(
                frame,
                origins,
                str(target),
                online_threshold_mw=self.online_threshold_mw,
                stable_delta_threshold_mw=self.stable_delta_threshold_mw,
            )
            for target in target_columns
        }
        output = pd.DataFrame(index=origins)
        output.index.name = "datetime"
        for column in expected_columns:
            weights = self.column_weights.get(str(column))
            if not weights:
                raise ValueError(f"缺少预测列的 OOF 融合权重: {column}")
            regime_weights = self.regime_column_weights.get(str(column), {})
            if not regime_weights:
                total = float(sum(weights.values()))
                output[column] = sum(
                    predictions[name][column] * (weight / total)
                    for name, weight in weights.items()
                )
                continue

            target = column_targets.get(str(column))
            if target is None:
                raise ValueError(f"无法确定工况融合列对应的目标: {column}")
            labels = regimes_by_target[target]
            values = np.empty(len(origins), dtype=float)
            for regime in labels.unique():
                mask = labels.to_numpy(dtype=object) == str(regime)
                selected = regime_weights.get(str(regime), weights)
                total = float(sum(selected.values()))
                values[mask] = sum(
                    predictions[name][column].to_numpy(dtype=float)[mask]
                    * (weight / total)
                    for name, weight in selected.items()
                )
            output[column] = values
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("OOF 融合结果包含缺失值或无穷值")
        return output


class HourlyCalibratedModel(ForecastModel):
    """使用训练期交叉验证筛选出的目标时段乘法因子校准预测。"""

    def __init__(
        self,
        base_model: ForecastModel,
        calibrations: Mapping[str, Mapping[str, float]],
        *,
        hour_bin_size: int = 4,
        interval_minutes: int = 15,
    ):
        self.base_model = base_model
        self.calibrations = {
            str(column): {
                str(hour_bin): float(factor) for hour_bin, factor in bins.items()
            }
            for column, bins in calibrations.items()
        }
        self.hour_bin_size = int(hour_bin_size)
        self.interval_minutes = int(interval_minutes)

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
    ) -> "HourlyCalibratedModel":
        self.base_model.fit(
            frame,
            target_columns,
            horizons,
            train_end=train_end,
            raw_targets=raw_targets,
            feature_matrix=feature_matrix,
            data_source=data_source,
        )
        return self

    def fit_progress_steps(
        self, target_columns: Sequence[str], horizons: Sequence[int]
    ) -> int | None:
        return self.base_model.fit_progress_steps(target_columns, horizons)

    def set_fit_progress_callback(self, callback) -> None:
        self.base_model.set_fit_progress_callback(callback)

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        output = self.base_model.predict(frame, origins, target_columns, horizons)
        for target in target_columns:
            for horizon_value in horizons:
                horizon = int(horizon_value)
                column = prediction_column(str(target), horizon, self.interval_minutes)
                bins = self.calibrations.get(column)
                if not bins:
                    continue
                target_times = origins + pd.Timedelta(
                    minutes=horizon * self.interval_minutes
                )
                factors = np.asarray(
                    [
                        float(bins.get(str(int(hour) // self.hour_bin_size), 1.0))
                        for hour in target_times.hour
                    ],
                    dtype=float,
                )
                output[column] = output[column].to_numpy(dtype=float) * factors
        return output
