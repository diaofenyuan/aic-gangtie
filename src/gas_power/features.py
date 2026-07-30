"""严格因果的工业时间序列特征。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from gas_power.availability import FeatureAvailabilityRegistry, FeatureUsage


class LeakageError(AssertionError):
    """自动化检查发现未来信息依赖。"""


def _rolling_slope(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return float("nan")
    x = np.arange(len(values), dtype=float)[finite]
    y = values[finite].astype(float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def _role_columns(value: Any) -> list[str]:
    """递归展开角色配置中的字段名，支持按煤气类型配置多个消耗来源。"""

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [column for item in value for column in _role_columns(item)]
    if isinstance(value, Mapping):
        return [column for item in value.values() for column in _role_columns(item)]
    return []


def _flatten_role_columns(roles: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(_role_columns(roles)))


def _sum_role_series(frame: pd.DataFrame, value: Any) -> tuple[pd.Series | None, list[str]]:
    columns = [column for column in _role_columns(value) if column in frame]
    if not columns:
        return None, []
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=1), columns


@dataclass
class CausalFeatureBuilder:
    """所有窗口统计均先 shift(1)，再在历史样本上滚动。"""

    feature_config: Mapping[str, Any]
    roles: Mapping[str, Any]
    interval_minutes: int = 15
    availability: FeatureAvailabilityRegistry | None = None
    model_scope: str = "long"

    def _source_allowed(self, source_field: str, lag_steps: int, feature_name: str) -> bool:
        if self.availability is None:
            return True
        usage = FeatureUsage(
            feature_name=feature_name,
            source_field=source_field,
            source_offset_steps=-int(lag_steps),
            scope=self.model_scope,
        )
        return self.availability.validate_usage(
            usage, pd.Timestamp("2000-01-01"), self.interval_minutes
        ) is None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("特征输入索引必须是 DatetimeIndex")
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError("特征输入时间索引必须严格递增且唯一")

        feature_values: dict[str, Any] = {}
        self._add_time_features(frame.index, feature_values)
        base_columns = [column for column in _flatten_role_columns(self.roles) if column in frame]
        self._add_missing_features(frame, feature_values, base_columns)
        self._add_lag_and_rolling_features(frame, feature_values, base_columns)
        self._add_gas_features(frame, feature_values)
        self._add_operating_features(frame, feature_values)
        # 一次性组装宽表，避免真实数据上逐列 insert 导致内存碎片。
        result = pd.DataFrame(feature_values, index=frame.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def _add_time_features(
        self,
        index: pd.DatetimeIndex,
        result: MutableMapping[str, Any],
    ) -> None:
        minute_of_day = index.hour * 60 + index.minute
        day_angle = 2.0 * np.pi * minute_of_day / 1440.0
        week_angle = 2.0 * np.pi * (index.dayofweek * 1440 + minute_of_day) / (7.0 * 1440.0)
        result["feat_time_day_sin"] = np.sin(day_angle)
        result["feat_time_day_cos"] = np.cos(day_angle)
        result["feat_time_week_sin"] = np.sin(week_angle)
        result["feat_time_week_cos"] = np.cos(week_angle)
        result["feat_time_hour"] = index.hour.astype(np.int8)
        result["feat_time_dayofweek"] = index.dayofweek.astype(np.int8)
        result["feat_time_is_weekend"] = (index.dayofweek >= 5).astype(np.int8)

    def _add_missing_features(
        self,
        frame: pd.DataFrame,
        result: MutableMapping[str, Any],
        base_columns: Iterable[str],
    ) -> None:
        for column in base_columns:
            existing = f"feat_missing__{column}"
            if not self._source_allowed(column, 0, existing):
                continue
            if existing in frame:
                result[existing] = frame[existing].astype(np.int8)
            else:
                result[existing] = frame[column].isna().astype(np.int8)
        if "feat_time_gap_inserted" in frame:
            result["feat_time_gap_inserted"] = frame["feat_time_gap_inserted"].astype(np.int8)

    def _add_lag_and_rolling_features(
        self,
        frame: pd.DataFrame,
        result: MutableMapping[str, Any],
        base_columns: Iterable[str],
    ) -> None:
        lag_steps = [int(value) for value in self.feature_config.get("lag_steps", [])]
        windows = [int(value) for value in self.feature_config.get("rolling_windows", [])]
        statistics = {str(value) for value in self.feature_config.get("rolling_statistics", [])}

        for column in base_columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            for lag in lag_steps:
                feature_name = f"feat_lag_{column}_{lag}"
                if self._source_allowed(column, lag, feature_name):
                    result[feature_name] = series.shift(lag)

            # 这一行是泄漏控制核心：当前值先移出窗口，统计量只看 t-1 及更早数据。
            shifted = series.shift(1)
            if not self._source_allowed(column, 1, f"feat_roll_{column}"):
                continue
            for window in windows:
                minimum = max(2, window // 2)
                rolling = shifted.rolling(window=window, min_periods=minimum)
                prefix = f"feat_roll_{column}_{window}"
                if "mean" in statistics:
                    result[f"{prefix}_mean"] = rolling.mean()
                if "std" in statistics:
                    result[f"{prefix}_std"] = rolling.std(ddof=0)
                if "min" in statistics:
                    result[f"{prefix}_min"] = rolling.min()
                if "max" in statistics:
                    result[f"{prefix}_max"] = rolling.max()
                if "slope" in statistics:
                    result[f"{prefix}_slope"] = rolling.apply(_rolling_slope, raw=True)

    def _add_gas_features(
        self, frame: pd.DataFrame, result: MutableMapping[str, Any]
    ) -> None:
        production = self.roles.get("gas_production", {})
        demand = self.roles.get("gas_user_demand", {})
        process_demand = self.roles.get("gas_process_demand", {})
        holders = self.roles.get("gas_holder", {})
        if isinstance(production, Mapping) and isinstance(demand, Mapping):
            for gas_type in sorted(set(production).intersection(demand)):
                production_series, production_columns = _sum_role_series(
                    frame, production[gas_type]
                )
                demand_sources: list[Any] = [demand[gas_type]]
                if isinstance(process_demand, Mapping) and gas_type in process_demand:
                    demand_sources.append(process_demand[gas_type])
                demand_series, demand_columns = _sum_role_series(frame, demand_sources)
                feature_name = f"feat_gas_balance_{gas_type}"
                if (
                    production_series is not None
                    and demand_series is not None
                    and all(
                        self._source_allowed(column, 1, feature_name)
                        for column in [*production_columns, *demand_columns]
                    )
                ):
                    # 资源平衡同样只使用 t-1 及更早观测，避免结算延迟造成穿越。
                    result[feature_name] = (production_series - demand_series).shift(1)
        if isinstance(holders, Mapping):
            for gas_type, holder_column_value in holders.items():
                holder_column = str(holder_column_value)
                feature_name = f"feat_holder_change_rate_{gas_type}"
                if (
                    holder_column in frame
                    and self._source_allowed(holder_column, 1, feature_name)
                    and self._source_allowed(holder_column, 2, feature_name)
                ):
                    result[feature_name] = (
                        frame[holder_column].shift(1).diff() / float(self.interval_minutes)
                    )

    def _add_operating_features(
        self, frame: pd.DataFrame, result: MutableMapping[str, Any]
    ) -> None:
        online_threshold = float(self.feature_config.get("unit_online_threshold_mw", 1.0))
        stable_threshold = float(self.feature_config.get("stable_delta_threshold_mw", 2.0))
        for target in self.roles.get("targets", []):
            target_name = str(target)
            if target_name not in frame:
                continue
            if not self._source_allowed(target_name, 0, f"feat_state_{target_name}"):
                continue
            load = pd.to_numeric(frame[target_name], errors="coerce")
            delta = load.diff()
            online = load > online_threshold
            result[f"feat_state_{target_name}_online"] = online.astype(np.int8)
            result[f"feat_state_{target_name}_startup"] = (online & ~online.shift(1, fill_value=False)).astype(np.int8)
            result[f"feat_state_{target_name}_shutdown"] = (~online & online.shift(1, fill_value=False)).astype(np.int8)
            result[f"feat_state_{target_name}_stable"] = (
                online & (delta.abs() <= stable_threshold)
            ).astype(np.int8)
            result[f"feat_state_{target_name}_ramp_up"] = (delta > stable_threshold).astype(np.int8)
            result[f"feat_state_{target_name}_ramp_down"] = (delta < -stable_threshold).astype(np.int8)
            result[f"feat_state_{target_name}_ramp_rate"] = delta / float(self.interval_minutes)


def assert_feature_causality(
    builder: CausalFeatureBuilder,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> None:
    """扰动截止点之后的全部数值，历史特征必须逐元素保持不变。"""

    if cutoff not in frame.index:
        raise ValueError(f"泄漏检查截止点不在索引中: {cutoff}")
    baseline = builder.transform(frame).loc[:cutoff]
    perturbed = frame.copy()
    future_mask = perturbed.index > cutoff
    numeric = list(perturbed.select_dtypes(include=[np.number]).columns)
    for column in numeric:
        values = pd.to_numeric(perturbed.loc[future_mask, column], errors="coerce")
        perturbed.loc[future_mask, column] = values * -7.0 + 1_000_003.0
    candidate = builder.transform(perturbed).loc[:cutoff]
    try:
        assert_frame_equal(baseline, candidate, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise LeakageError(f"未来数据扰动改变了截止点之前的特征: {exc}") from exc


def assert_shift_before_rolling(
    builder: CausalFeatureBuilder,
    frame: pd.DataFrame,
    column: str,
    window: int,
) -> None:
    """抽查滚动均值等于 t-1 向前窗口，防止实现改动破坏 shift 语义。"""

    features = builder.transform(frame)
    feature_name = f"feat_roll_{column}_{window}_mean"
    if feature_name not in features:
        raise LeakageError(f"缺少待检查滚动特征: {feature_name}")
    positions = np.flatnonzero(features[feature_name].notna().to_numpy())
    if len(positions) == 0:
        raise LeakageError(f"滚动特征没有可检查的有效值: {feature_name}")
    position = int(positions[len(positions) // 2])
    expected = float(frame[column].iloc[max(0, position - window) : position].mean())
    actual = float(features[feature_name].iloc[position])
    if not np.isclose(actual, expected, equal_nan=True):
        raise LeakageError(
            f"滚动特征未遵循先 shift 后统计: {feature_name}, actual={actual}, expected={expected}"
        )
