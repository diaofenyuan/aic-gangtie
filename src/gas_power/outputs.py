"""官方预测和优化 CSV 的生成与严格校验。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from gas_power.models.base import prediction_column, prediction_columns
from gas_power.time_semantics import validate_prediction_columns


class OutputValidationError(ValueError):
    """结果文件不满足官方宽表约束。"""


def _with_datetime_column(frame: pd.DataFrame, timestamp_format: str) -> pd.DataFrame:
    output = frame.copy()
    if "datetime" not in output.columns:
        if not isinstance(output.index, pd.DatetimeIndex):
            raise OutputValidationError("结果必须包含 datetime 列或 DatetimeIndex")
        output.insert(0, "datetime", output.index.strftime(timestamp_format))
    else:
        output["datetime"] = pd.to_datetime(output["datetime"], errors="raise").dt.strftime(
            timestamp_format
        )
    return output.reset_index(drop=True)


def _validate_datetime(output: pd.DataFrame, timestamp_format: str) -> None:
    values = output["datetime"]
    if not pd.api.types.is_string_dtype(values.dtype):
        raise OutputValidationError("datetime 必须是字符串类型")
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    if not values.map(lambda value: bool(pattern.fullmatch(str(value)))).all():
        raise OutputValidationError("datetime 格式必须为 YYYY-MM-DD HH:MM:SS")
    parsed = pd.to_datetime(values, format=timestamp_format, errors="raise")
    if parsed.duplicated().any():
        raise OutputValidationError("结果存在重复 datetime")
    if not parsed.is_monotonic_increasing:
        raise OutputValidationError("结果 datetime 必须严格按预测起点升序排列")


def _validate_numeric(output: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(output[column]):
            raise OutputValidationError(f"结果字段不是数值类型: {column}")
    values = output[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise OutputValidationError("结果包含缺失值或无穷值")


def validate_forecast_frame(
    frame: pd.DataFrame,
    targets: Sequence[str],
    horizons: Sequence[int],
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    interval_minutes: int = 15,
    expected_origins: pd.DatetimeIndex | None = None,
    capacity_bounds: dict[str, float] | None = None,
    enforce_target_consistency: bool = False,
    capacity_tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    output = _with_datetime_column(frame, timestamp_format)
    expected_numeric = prediction_columns(targets, horizons, interval_minutes)
    expected = ["datetime", *expected_numeric]
    if list(output.columns) != expected:
        missing = [column for column in expected if column not in output]
        extra = [column for column in output.columns if column not in expected]
        raise OutputValidationError(
            f"预测列名或顺序不正确；缺失={missing[:5]}，多余={extra[:5]}"
        )
    try:
        validate_prediction_columns(expected_numeric, targets, horizons, interval_minutes)
    except ValueError as exc:
        raise OutputValidationError(str(exc)) from exc
    if output.empty:
        raise OutputValidationError("预测结果不能为空")
    _validate_datetime(output, timestamp_format)
    _validate_numeric(output, expected_numeric)
    parsed_origins = pd.DatetimeIndex(
        pd.to_datetime(output["datetime"], format=timestamp_format, errors="raise")
    )
    if expected_origins is not None and not parsed_origins.equals(
        pd.DatetimeIndex(expected_origins)
    ):
        raise OutputValidationError(
            "datetime 与配置的预测起点不一致，可能把目标时刻写成了预测起点或发生缺行/错位"
        )
    if capacity_bounds:
        for target in targets:
            columns = [
                prediction_column(str(target), int(horizon), interval_minutes)
                for horizon in horizons
            ]
            values = output[columns].to_numpy(dtype=float)
            if (values < -capacity_tolerance).any():
                raise OutputValidationError(f"{target} 预测值违反非负边界")
            if target in capacity_bounds and (
                values > float(capacity_bounds[target]) + capacity_tolerance
            ).any():
                raise OutputValidationError(f"{target} 预测值超过配置装机容量")
    if enforce_target_consistency and {"generator_1", "generator_all"}.issubset(targets):
        for horizon in horizons:
            one = prediction_column("generator_1", int(horizon), interval_minutes)
            total = prediction_column("generator_all", int(horizon), interval_minutes)
            if (output[one] > output[total] + capacity_tolerance).any():
                raise OutputValidationError(
                    f"generator_1 在 t+{int(horizon) * interval_minutes} 超过 generator_all，"
                    "可能发生目标列互换"
                )
    return output


def validate_optimization_frame(
    frame: pd.DataFrame,
    gas_columns: Sequence[str],
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
) -> pd.DataFrame:
    output = _with_datetime_column(frame, timestamp_format)
    expected = ["datetime", *gas_columns]
    if list(output.columns) != expected:
        raise OutputValidationError(
            f"优化列名或顺序不正确，期望 {expected}，实际 {list(output.columns)}"
        )
    if not gas_columns or any(not column.startswith("opt_") for column in gas_columns):
        raise OutputValidationError("所有优化字段必须带 opt_ 前缀")
    if output.empty:
        raise OutputValidationError("优化结果不能为空")
    _validate_datetime(output, timestamp_format)
    _validate_numeric(output, gas_columns)
    if (output[list(gas_columns)].to_numpy(dtype=float) < -1.0e-9).any():
        raise OutputValidationError("发电用煤气量不能为负数")
    return output


def _write_and_verify_utf8(output: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")
    raw = path.read_bytes()
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OutputValidationError(f"结果文件不是有效 UTF-8: {path}") from exc


def write_forecast_csv(
    frame: pd.DataFrame,
    path: Path,
    targets: Sequence[str],
    horizons: Sequence[int],
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    interval_minutes: int = 15,
    expected_origins: pd.DatetimeIndex | None = None,
    capacity_bounds: dict[str, float] | None = None,
    enforce_target_consistency: bool = False,
    capacity_tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    output = validate_forecast_frame(
        frame,
        targets,
        horizons,
        timestamp_format=timestamp_format,
        interval_minutes=interval_minutes,
        expected_origins=expected_origins,
        capacity_bounds=capacity_bounds,
        enforce_target_consistency=enforce_target_consistency,
        capacity_tolerance=capacity_tolerance,
    )
    _write_and_verify_utf8(output, path)
    reread = pd.read_csv(path, encoding="utf-8")
    return validate_forecast_frame(
        reread,
        targets,
        horizons,
        timestamp_format=timestamp_format,
        interval_minutes=interval_minutes,
        expected_origins=expected_origins,
        capacity_bounds=capacity_bounds,
        enforce_target_consistency=enforce_target_consistency,
        capacity_tolerance=capacity_tolerance,
    )


def write_optimization_csv(
    frame: pd.DataFrame,
    path: Path,
    gas_columns: Sequence[str],
    timestamp_format: str = "%Y-%m-%d %H:%M:%S",
) -> pd.DataFrame:
    output = validate_optimization_frame(frame, gas_columns, timestamp_format)
    _write_and_verify_utf8(output, path)
    reread = pd.read_csv(path, encoding="utf-8")
    return validate_optimization_frame(reread, gas_columns, timestamp_format)
