"""目标参数化模型：分别预测 generator_1 和额外机组负荷，再重建总负荷。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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


def _configured_columns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [column for item in value for column in _configured_columns(item)]
    if isinstance(value, Mapping):
        return [column for item in value.values() for column in _configured_columns(item)]
    return []


def _sum_columns(frame: pd.DataFrame, value: Any) -> pd.Series | None:
    columns = [column for column in _configured_columns(value) if column in frame]
    if not columns:
        return None
    return frame[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


class GasAvailabilityForecastModel(ForecastModel):
    """先预测煤气资源平衡，再用非负转换关系重建机组负荷。"""

    def __init__(
        self,
        stage1_model: ForecastModel,
        feature_builder: CausalFeatureBuilder,
        *,
        interval_minutes: int = 15,
    ):
        roles = feature_builder.roles
        production = roles.get("gas_production", {})
        demand = roles.get("gas_user_demand", {})
        if not isinstance(production, Mapping) or not isinstance(demand, Mapping):
            raise ValueError("煤气可用量模型要求按煤气类型配置生产量和用户消耗量")
        self.configured_gas_types = sorted(set(production).intersection(demand))
        if not self.configured_gas_types:
            raise ValueError("煤气可用量模型没有可用的煤气类型")
        self.gas_types = list(self.configured_gas_types)
        self.stage1_model = stage1_model
        self.feature_builder = feature_builder
        self.roles = roles
        self.interval_minutes = int(interval_minutes)
        self.intermediate_targets = [
            name
            for gas_type in self.gas_types
            for name in (
                self._production_name(gas_type),
                self._consumption_name(gas_type),
                self._holder_change_name(gas_type),
            )
        ]
        self.calibrators_: dict[tuple[str, int], LinearRegression] = {}
        self.train_metadata_: dict[str, object] = {}
        self._fit_progress_callback: FitProgressCallback | None = None

    @staticmethod
    def _production_name(gas_type: str) -> str:
        return f"gas_stage1_production__{gas_type}"

    @staticmethod
    def _consumption_name(gas_type: str) -> str:
        return f"gas_stage1_consumption__{gas_type}"

    @staticmethod
    def _holder_change_name(gas_type: str) -> str:
        return f"gas_stage1_holder_change__{gas_type}"

    @staticmethod
    def _available_name(gas_type: str) -> str:
        return f"gas_available__{gas_type}"

    def fit_progress_steps(
        self,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> int | None:
        stage1 = self.stage1_model.fit_progress_steps(self.intermediate_targets, horizons)
        if stage1 is None:
            return None
        return int(stage1) + 2 * len(horizons)

    def set_fit_progress_callback(
        self,
        callback: FitProgressCallback | None,
    ) -> None:
        self._fit_progress_callback = callback
        self.stage1_model.set_fit_progress_callback(callback)

    def _with_intermediates(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        production = self.roles.get("gas_production", {})
        user_demand = self.roles.get("gas_user_demand", {})
        process_demand = self.roles.get("gas_process_demand", {})
        holders = self.roles.get("gas_holder", {})
        for gas_type in self.gas_types:
            produced = _sum_columns(output, production[gas_type])
            user = _sum_columns(output, user_demand[gas_type])
            process = (
                _sum_columns(output, process_demand.get(gas_type))
                if isinstance(process_demand, Mapping)
                else None
            )
            if produced is None or user is None:
                raise ValueError(f"煤气类型 {gas_type} 缺少生产量或用户消耗量")
            consumption = user if process is None else user.add(process, fill_value=0.0)
            holder = (
                _sum_columns(output, holders.get(gas_type))
                if isinstance(holders, Mapping)
                else None
            )
            holder_change = (
                holder.diff().fillna(0.0) / float(self.interval_minutes)
                if holder is not None
                else pd.Series(0.0, index=output.index)
            )
            output[self._production_name(gas_type)] = produced
            output[self._consumption_name(gas_type)] = consumption
            output[self._holder_change_name(gas_type)] = holder_change
            output[self._available_name(gas_type)] = produced - consumption - holder_change
        return output

    @staticmethod
    def _component_labels(
        raw_targets: pd.DataFrame,
        target_columns: Sequence[str],
    ) -> pd.DataFrame:
        labels = raw_targets.reindex(columns=list(target_columns)).copy(deep=True)
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
    ) -> "GasAvailabilityForecastModel":
        if data_source == "scoring":
            raise ValueError("评分期 scoring 数据禁止用于煤气可用量模型拟合")
        if not {"generator_1", "generator_all"}.issubset(target_columns):
            raise ValueError("煤气可用量模型要求 generator_1 和 generator_all")
        production_roles = self.roles.get("gas_production", {})
        demand_roles = self.roles.get("gas_user_demand", {})
        self.gas_types = [
            gas_type
            for gas_type in self.configured_gas_types
            if isinstance(production_roles, Mapping)
            and isinstance(demand_roles, Mapping)
            and _sum_columns(frame, production_roles.get(gas_type)) is not None
            and _sum_columns(frame, demand_roles.get(gas_type)) is not None
        ]
        if not self.gas_types:
            raise ValueError("训练数据没有同时具备生产量和用户消耗量的煤气类型")
        self.intermediate_targets = [
            name
            for gas_type in self.gas_types
            for name in (
                self._production_name(gas_type),
                self._consumption_name(gas_type),
                self._holder_change_name(gas_type),
            )
        ]
        work = self._with_intermediates(frame)
        effective_end = pd.Timestamp(train_end if train_end is not None else work.index.max())
        stage1_labels = work[self.intermediate_targets].copy(deep=True)
        stage1_features = self.feature_builder.transform(work)
        self.stage1_model.fit(
            work,
            self.intermediate_targets,
            horizons,
            train_end=effective_end,
            raw_targets=stage1_labels,
            feature_matrix=stage1_features,
            data_source=data_source,
        )

        labels = self._component_labels(
            raw_targets if raw_targets is not None else frame[list(target_columns)],
            target_columns,
        )
        component_targets = ["generator_1", "generator_extra"]
        self.calibrators_.clear()
        for component in component_targets:
            current = (
                pd.to_numeric(work["generator_1"], errors="coerce")
                if component == "generator_1"
                else pd.to_numeric(work["generator_all"], errors="coerce")
                - pd.to_numeric(work["generator_1"], errors="coerce")
            )
            for horizon_value in horizons:
                horizon = int(horizon_value)
                target = pd.to_numeric(labels[component], errors="coerce").shift(-horizon)
                x = pd.DataFrame({"current_load": current}, index=work.index)
                for gas_type in self.gas_types:
                    available = pd.to_numeric(
                        work[self._available_name(gas_type)], errors="coerce"
                    ).shift(-horizon)
                    x[f"available_{gas_type}"] = available.clip(lower=0.0)
                valid = (x.index + pd.Timedelta(minutes=horizon * self.interval_minutes)) <= effective_end
                valid &= target.notna() & np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
                x_train = x.loc[valid]
                y_train = target.loc[valid]
                if len(x_train) < 20:
                    raise ValueError(
                        f"煤气可用量二阶段 {component} 步长 {horizon} 有效样本不足 20 条"
                    )
                positive = y_train.abs().loc[y_train.abs() > 0.0]
                floor = float(positive.quantile(0.01)) if not positive.empty else 1.0
                weights = 1.0 / y_train.abs().clip(lower=max(floor, 1.0e-6))
                calibrator = LinearRegression(positive=True)
                calibrator.fit(x_train, y_train, sample_weight=weights)
                self.calibrators_[(component, horizon)] = calibrator
                if self._fit_progress_callback is not None:
                    self._fit_progress_callback(f"煤气可用量映射 {component}，步长 {horizon}")
        self.train_metadata_ = {
            "data_source": data_source,
            "parameterization": "gas_availability_then_generation",
            "gas_types": self.gas_types,
            "intermediate_targets": self.intermediate_targets,
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
        if not self.calibrators_:
            raise RuntimeError("煤气可用量模型尚未拟合")
        work = self._with_intermediates(frame)
        stage1 = self.stage1_model.predict(
            work,
            origins,
            self.intermediate_targets,
            horizons,
        )
        current_one = pd.to_numeric(work.loc[origins, "generator_1"], errors="coerce")
        current_extra = (
            pd.to_numeric(work.loc[origins, "generator_all"], errors="coerce") - current_one
        )
        output = pd.DataFrame(index=origins)
        output.index.name = "datetime"
        for horizon_value in horizons:
            horizon = int(horizon_value)
            availability = pd.DataFrame(index=origins)
            for gas_type in self.gas_types:
                production = stage1[prediction_column(self._production_name(gas_type), horizon)]
                consumption = stage1[prediction_column(self._consumption_name(gas_type), horizon)]
                holder_change = stage1[
                    prediction_column(self._holder_change_name(gas_type), horizon)
                ]
                availability[f"available_{gas_type}"] = (
                    production - consumption - holder_change
                ).clip(lower=0.0)
            component_predictions: dict[str, np.ndarray] = {}
            for component, current in (
                ("generator_1", current_one),
                ("generator_extra", current_extra),
            ):
                x = availability.copy()
                x.insert(0, "current_load", current.to_numpy(dtype=float))
                calibrator = self.calibrators_.get((component, horizon))
                if calibrator is None:
                    raise ValueError(f"缺少煤气可用量映射 {component} 步长 {horizon}")
                component_predictions[component] = np.maximum(
                    np.asarray(calibrator.predict(x), dtype=float), 0.0
                )
            one_column = prediction_column("generator_1", horizon)
            output[one_column] = component_predictions["generator_1"]
            output[prediction_column("generator_all", horizon)] = (
                component_predictions["generator_1"]
                + component_predictions["generator_extra"]
            )
        if not np.isfinite(output.to_numpy(dtype=float)).all():
            raise ValueError("煤气可用量模型预测包含 NaN/Inf")
        return output
