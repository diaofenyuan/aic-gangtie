"""目标参数化模型：分别预测 generator_1 和额外机组负荷，再重建总负荷。"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import pandas as pd

from gas_power.features import CausalFeatureBuilder
from gas_power.models.base import (
    FitProgressCallback,
    ForecastModel,
    prediction_column,
    validate_prediction_request,
)


class ComponentReconstructionModel(ForecastModel):
    """通过 ``generator_all-generator_1`` 降低总负荷目标的非平稳性。"""

    def __init__(self, base_model: ForecastModel, feature_builder: CausalFeatureBuilder):
        roles = dict(feature_builder.roles)
        original_targets = [str(value) for value in roles.get("targets", [])]
        if "generator_1" not in original_targets or "generator_all" not in original_targets:
            raise ValueError("目标重建要求 generator_1 和 generator_all")
        roles["targets"] = ["generator_1", "generator_extra"]
        self.feature_builder = replace(feature_builder, roles=roles)
        self.base_model = base_model
        if hasattr(self.base_model, "feature_builder"):
            self.base_model.feature_builder = self.feature_builder
        self.original_targets = original_targets
        self.component_targets = ["generator_1", "generator_extra"]
        self.train_metadata_: dict[str, object] = {}

    def fit_progress_steps(
        self,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> int | None:
        return self.base_model.fit_progress_steps(self.component_targets, horizons)

    def set_fit_progress_callback(
        self,
        callback: FitProgressCallback | None,
    ) -> None:
        self.base_model.set_fit_progress_callback(callback)

    @staticmethod
    def _with_extra(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["generator_extra"] = (
            pd.to_numeric(output["generator_all"], errors="coerce")
            - pd.to_numeric(output["generator_1"], errors="coerce")
        )
        return output

    @staticmethod
    def _labels_with_extra(raw_targets: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        labels = raw_targets.copy(deep=True)
        labels["generator_extra"] = (
            pd.to_numeric(labels["generator_all"], errors="coerce")
            - pd.to_numeric(labels["generator_1"], errors="coerce")
        )
        return labels

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
    ) -> "ComponentReconstructionModel":
        if data_source == "scoring":
            raise ValueError("评分期 scoring 数据禁止用于目标参数化模型拟合")
        work = self._with_extra(frame)
        labels = self._labels_with_extra(
            raw_targets if raw_targets is not None else frame[list(target_columns)], work
        )
        # 外部缓存按原始双目标生成，组件参数化后的目标集合不同，必须重新构造同口径特征。
        features = self.feature_builder.transform(work)
        self.base_model.fit(
            work,
            self.component_targets,
            horizons,
            train_end=train_end,
            raw_targets=labels,
            feature_matrix=features,
            data_source=data_source,
        )
        self.train_metadata_ = {
            "data_source": data_source,
            "parameterization": "generator_1_plus_extra",
            "raw_labels": True,
        }
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        validate_prediction_request(frame, origins, ["generator_1", "generator_all"], horizons)
        work = self._with_extra(frame)
        component_prediction = self.base_model.predict(
            work, origins, self.component_targets, horizons
        )
        output = pd.DataFrame(index=origins)
        output.index.name = "datetime"
        for horizon in horizons:
            one_column = prediction_column("generator_1", int(horizon))
            extra_column = prediction_column("generator_extra", int(horizon))
            output[one_column] = component_prediction[one_column]
            output[prediction_column("generator_all", int(horizon))] = (
                component_prediction[one_column] + component_prediction[extra_column]
            )
        return output
