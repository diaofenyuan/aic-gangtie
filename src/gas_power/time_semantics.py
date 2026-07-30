"""预测起点、目标时刻、列偏移和重采样语义的统一校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from gas_power.availability import FeatureAvailabilityRegistry, FeatureUsage


PREDICTION_COLUMN_RE = re.compile(r"^(?P<target>.+)_t\+(?P<minutes>\d+)_pred$")


@dataclass(frozen=True)
class HorizonColumn:
    target: str
    horizon_steps: int
    horizon_minutes: int
    column: str


def parse_horizon_column(column: str, interval_minutes: int = 15) -> HorizonColumn:
    match = PREDICTION_COLUMN_RE.fullmatch(str(column))
    if match is None:
        raise ValueError(f"预测列名不符合 target_t+分钟 格式: {column}")
    minutes = int(match.group("minutes"))
    if minutes <= 0 or minutes % interval_minutes != 0:
        raise ValueError(f"预测列步长不是 {interval_minutes} 分钟整数倍: {column}")
    return HorizonColumn(
        target=match.group("target"),
        horizon_steps=minutes // interval_minutes,
        horizon_minutes=minutes,
        column=str(column),
    )


def validate_prediction_columns(
    columns: Sequence[str],
    targets: Sequence[str],
    horizons: Sequence[int],
    interval_minutes: int = 15,
) -> list[HorizonColumn]:
    """校验列的目标、顺序和完整步长，防止整体错位或目标互换。"""

    expected = [
        f"{target}_t+{int(horizon) * interval_minutes}_pred"
        for target in targets
        for horizon in horizons
    ]
    if list(columns) != expected:
        raise ValueError(f"预测列顺序/完整性错误，期望={expected[:4]}...，实际={list(columns)[:4]}...")
    parsed = [parse_horizon_column(column, interval_minutes) for column in columns]
    if [item.target for item in parsed] != [str(t) for t in targets for _ in horizons]:
        raise ValueError("预测列目标顺序错误，可能发生 generator_1/generator_all 互换")
    return parsed


def validate_target_times(
    origins: Iterable[pd.Timestamp],
    target_times: Iterable[pd.Timestamp],
    horizon_steps: Iterable[int],
    interval_minutes: int = 15,
) -> None:
    origin_series = pd.to_datetime(list(origins))
    target_series = pd.to_datetime(list(target_times))
    steps = list(horizon_steps)
    if len(origin_series) != len(target_series) or len(steps) != len(origin_series):
        raise ValueError("起点、目标时刻和步长数量不一致")
    expected = origin_series + pd.to_timedelta(steps, unit="m") * int(interval_minutes)
    if not (expected == target_series).all():
        raise ValueError("target_time != forecast_origin_time + horizon，存在时间对齐错误")


def validate_feature_availability(
    registry: FeatureAvailabilityRegistry,
    usages: Iterable[FeatureUsage],
    origin: pd.Timestamp,
    interval_minutes: int = 15,
) -> None:
    issues = registry.audit_usages(usages, origin, interval_minutes)
    if issues:
        message = "; ".join(f"{item.feature_name}: {item.message}" for item in issues[:5])
        raise ValueError(f"特征可用时间校验失败: {message}")


def resampling_variants() -> list[tuple[str, str]]:
    """返回审计需要比较的 pandas 重采样边界组合。"""

    return [(label, closed) for label in ("left", "right") for closed in ("left", "right")]
