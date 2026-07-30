"""配置驱动的多表读取、对齐与因果清洗。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from gas_power.config import ConfigError, ProjectConfig


class DataError(ValueError):
    """输入数据无法按配置安全处理。"""


@dataclass
class TableQuality:
    """单表清洗统计。"""

    path: str
    input_rows: int = 0
    invalid_timestamps: int = 0
    duplicate_rows: int = 0
    output_rows: int = 0
    missing_sources: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """多表处理过程中可审计的数据质量报告。"""

    tables: dict[str, TableQuality] = field(default_factory=dict)
    aligned_rows: int = 0
    inserted_time_points: int = 0
    outlier_counts: dict[str, int] = field(default_factory=dict)
    missing_before_imputation: dict[str, int] = field(default_factory=dict)
    missing_after_imputation: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise DataError(f"暂不支持的数据文件类型: {path}")


def _resolve_sources(columns: Iterable[str], mapping: Mapping[str, Any]) -> list[str]:
    available = list(columns)
    configured = mapping.get("sources", [])
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, list):
        raise ConfigError("字段映射 sources 必须是字符串或列表")
    selected = [str(name) for name in configured if str(name) in available]

    patterns = mapping.get("patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, list):
        raise ConfigError("字段映射 patterns 必须是字符串或列表")
    for pattern in patterns:
        compiled = re.compile(str(pattern))
        selected.extend(name for name in available if compiled.fullmatch(str(name)))
    return list(dict.fromkeys(selected))


def _combine_sources(frame: pd.DataFrame, sources: list[str], method: str) -> pd.Series:
    numeric = frame[sources].apply(pd.to_numeric, errors="coerce")
    if method == "sum":
        return numeric.sum(axis=1, min_count=1)
    if method == "mean":
        return numeric.mean(axis=1)
    if method == "first":
        return numeric.bfill(axis=1).iloc[:, 0]
    raise ConfigError(f"不支持的字段合并方式: {method}")


def _aggregate_duplicates(frame: pd.DataFrame, methods: Mapping[str, str]) -> pd.DataFrame:
    aggregations: dict[str, str] = {}
    for column in frame.columns:
        method = methods.get(column, "mean")
        if method not in {"mean", "sum", "min", "max", "first", "last", "median"}:
            raise ConfigError(f"字段 {column} 的聚合方式不受支持: {method}")
        aggregations[column] = method
    return frame.groupby(level=0, sort=True).agg(aggregations)


class ConfiguredDataLoader:
    """按 YAML 字段角色读取任意数量的时序表。"""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.data_config = config.section("data")
        self.preprocessing = config.section("preprocessing")

    def load(self) -> tuple[pd.DataFrame, DataQualityReport]:
        table_frames: list[pd.DataFrame] = []
        table_indexes: list[pd.DatetimeIndex] = []
        report = DataQualityReport()
        used_columns: set[str] = set()

        tables = self.data_config["tables"]
        for table_name, table_config in tables.items():
            table_path = self.config.path("data") / str(table_config["path"])
            required_table = bool(table_config.get("required", True))
            if not table_path.exists():
                if required_table:
                    raise DataError(f"必需数据表不存在: {table_path}")
                continue

            mapped, quality = self._load_one_table(table_path, table_config)
            report.tables[str(table_name)] = quality
            duplicate_columns = used_columns.intersection(mapped.columns)
            if duplicate_columns:
                raise DataError(f"多表映射产生重复字段: {sorted(duplicate_columns)}")
            used_columns.update(str(column) for column in mapped.columns)
            table_frames.append(mapped)
            table_indexes.append(mapped.index)

        if not table_frames:
            raise DataError("没有成功读取任何数据表")

        aligned = pd.concat(table_frames, axis=1, join="outer").sort_index()
        union_observed = table_indexes[0]
        for index in table_indexes[1:]:
            union_observed = union_observed.union(index)
        frequency = str(self.data_config.get("frequency", "15min"))
        full_index = pd.date_range(aligned.index.min(), aligned.index.max(), freq=frequency)
        inserted = ~full_index.isin(union_observed)
        aligned = aligned.reindex(full_index)
        aligned.index.name = "datetime"
        aligned["feat_time_gap_inserted"] = inserted.astype(np.int8)

        report.aligned_rows = len(aligned)
        report.inserted_time_points = int(inserted.sum())
        cleaned = clean_aligned_frame(aligned, self.preprocessing, report)
        self._validate_targets(cleaned)
        return cleaned, report

    def _load_one_table(
        self, path: Path, table_config: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, TableQuality]:
        raw = _read_table(path)
        quality = TableQuality(path=str(path), input_rows=len(raw))
        timestamp_column = str(table_config.get("timestamp", "datetime"))
        if timestamp_column not in raw.columns:
            if not bool(table_config.get("time_series", True)):
                # 静态电价/规则表交由业务层读取，不参与时序对齐。
                quality.output_rows = 0
                return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime")), quality
            if not bool(table_config.get("required", True)):
                quality.missing_sources.append(timestamp_column)
                quality.output_rows = 0
                return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime")), quality
            raise DataError(f"{path.name} 缺少时间戳字段 {timestamp_column}")

        timestamp_format = self.data_config.get("timestamp_format")
        parsed = pd.to_datetime(
            raw[timestamp_column],
            format=str(timestamp_format) if timestamp_format else None,
            errors="coerce",
        )
        quality.invalid_timestamps = int(parsed.isna().sum())
        valid = raw.loc[parsed.notna()].copy()
        valid.index = pd.DatetimeIndex(parsed.loc[parsed.notna()])
        valid.index.name = "datetime"

        mappings = table_config.get("mappings")
        if not isinstance(mappings, Mapping) or not mappings:
            raise ConfigError(f"数据表 {path.name} 缺少 mappings")
        mapped = pd.DataFrame(index=valid.index)
        aggregate_methods: dict[str, str] = {}
        for canonical_name, mapping_value in mappings.items():
            mapping = mapping_value if isinstance(mapping_value, Mapping) else {"sources": mapping_value}
            sources = _resolve_sources(valid.columns, mapping)
            required = bool(mapping.get("required", False))
            if not sources:
                quality.missing_sources.append(str(canonical_name))
                if required:
                    raise DataError(
                        f"{path.name} 无法映射必需字段 {canonical_name}；"
                        f"可用字段为 {list(valid.columns)}"
                    )
                continue
            combine = str(mapping.get("combine", "sum" if len(sources) > 1 else "first"))
            mapped[str(canonical_name)] = _combine_sources(valid, sources, combine)
            aggregate_methods[str(canonical_name)] = str(
                mapping.get("aggregation", table_config.get("aggregation", "mean"))
            )

        quality.duplicate_rows = int(mapped.index.duplicated(keep=False).sum())
        mapped = _aggregate_duplicates(mapped, aggregate_methods)
        frequency = str(self.data_config.get("frequency", "15min"))
        resampling = table_config.get("resampling", self.data_config.get("resampling", {}))
        if not isinstance(resampling, Mapping):
            raise ConfigError("data.resampling 必须是字典")
        label = str(resampling.get("label", "right"))
        closed = str(resampling.get("closed", "right"))
        if label not in {"left", "right"} or closed not in {"left", "right"}:
            raise ConfigError("重采样 label/closed 只能是 left 或 right")
        mapped = mapped.resample(frequency, label=label, closed=closed).agg(aggregate_methods)
        mapped = mapped.sort_index()
        quality.output_rows = len(mapped)
        return mapped, quality

    def _validate_targets(self, frame: pd.DataFrame) -> None:
        roles = self.data_config["roles"]
        targets = [str(column) for column in roles["targets"]]
        missing = [column for column in targets if column not in frame.columns]
        if missing:
            raise DataError(f"处理后缺少预测目标: {missing}")
        empty = [column for column in targets if frame[column].notna().sum() == 0]
        if empty:
            raise DataError(f"预测目标全为空: {empty}")


def clean_aligned_frame(
    frame: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    report: DataQualityReport | None = None,
) -> pd.DataFrame:
    """执行仅依赖当前及历史观测的异常处理和填补。"""

    cleaned = frame.copy()
    numeric_columns = [
        str(column)
        for column in cleaned.select_dtypes(include=[np.number]).columns
        if not str(column).startswith("feat_")
    ]
    missing_before = cleaned[numeric_columns].isna().sum().astype(int).to_dict()
    for column in numeric_columns:
        indicator = f"feat_missing__{column}"
        if indicator not in cleaned.columns:
            cleaned[indicator] = cleaned[column].isna().astype(np.int8)

    outlier_counts = apply_causal_outlier_filter(
        cleaned,
        numeric_columns,
        preprocessing.get("outliers", {}),
    )
    imputation = preprocessing.get("imputation", {})
    method = str(imputation.get("method", "ffill"))
    limit_value = imputation.get("limit")
    limit = int(limit_value) if limit_value is not None else None
    if method == "ffill":
        # 禁止使用 bfill 或双向插值，以免未来观测穿越到预测起点之前。
        cleaned[numeric_columns] = cleaned[numeric_columns].ffill(limit=limit)
    elif method == "none":
        pass
    else:
        raise ConfigError(f"仅支持因果填补方法 ffill/none，收到: {method}")

    if report is not None:
        report.outlier_counts = outlier_counts
        report.missing_before_imputation = missing_before
        report.missing_after_imputation = (
            cleaned[numeric_columns].isna().sum().astype(int).to_dict()
        )
    return cleaned


def apply_causal_outlier_filter(
    frame: pd.DataFrame,
    columns: Iterable[str],
    config: Mapping[str, Any],
) -> dict[str, int]:
    """用历史窗口 IQR 检测异常；窗口先 shift，当前值不参与阈值计算。"""

    counts: dict[str, int] = {str(column): 0 for column in columns}
    if not bool(config.get("enabled", False)):
        return counts
    window = int(config.get("window", 96))
    min_periods = int(config.get("min_periods", max(4, window // 4)))
    multiplier = float(config.get("iqr_multiplier", 4.0))
    replace_with = str(config.get("replace_with", "median"))
    if replace_with != "median":
        raise ConfigError("当前异常替换接口仅支持历史窗口 median")

    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        history = series.shift(1)
        rolling = history.rolling(window=window, min_periods=min_periods)
        q1 = rolling.quantile(0.25)
        median = rolling.median()
        q3 = rolling.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        valid_threshold = iqr.notna() & (iqr > 0)
        mask = valid_threshold & ((series < lower) | (series > upper))
        counts[str(column)] = int(mask.sum())
        frame.loc[mask, column] = median.loc[mask]
    return counts


def write_time_frame(frame: pd.DataFrame, path: Path, timestamp_format: str) -> None:
    """以统一时间戳和 UTF-8 编码写出时序表。"""

    output = frame.copy()
    output.insert(0, "datetime", output.index.strftime(timestamp_format))
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")


def read_time_frame(path: Path, timestamp_format: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    if "datetime" not in frame.columns:
        raise DataError(f"缓存文件缺少 datetime: {path}")
    parsed = pd.to_datetime(frame.pop("datetime"), format=timestamp_format, errors="raise")
    frame.index = pd.DatetimeIndex(parsed, name="datetime")
    return frame
