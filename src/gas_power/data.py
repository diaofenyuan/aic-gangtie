"""配置驱动的多表读取、对齐与因果清洗。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_power.config import ConfigError, ProjectConfig


class DataError(ValueError):
    """输入数据无法按配置安全处理。"""


@dataclass(frozen=True)
class PreparedForecastData:
    """将原始标签、模型输入和质量标记显式隔离的数据契约。"""

    model_input: pd.DataFrame
    raw_targets: pd.DataFrame
    raw_observations: pd.DataFrame
    missing_flags: pd.DataFrame
    anomaly_flags: pd.DataFrame
    label_valid_mask: pd.DataFrame
    source: str = "training"

    def __post_init__(self) -> None:
        indexes = {
            tuple(frame.index)
            for frame in (
                self.model_input,
                self.raw_targets,
                self.raw_observations,
                self.missing_flags,
                self.anomaly_flags,
                self.label_valid_mask,
            )
        }
        if len(indexes) != 1:
            raise DataError("PreparedForecastData 的各数据区必须使用同一时间索引")
        if self.source not in {"training", "scoring", "synthetic"}:
            raise DataError(f"未知数据来源: {self.source}")

    def assert_training_allowed(self) -> None:
        """评分期数据只能用于逐起点推理，不能进入任何拟合或调参过程。"""

        if self.source == "scoring":
            raise DataError("评分期 scoring 数据禁止用于拟合、OOF、早停、特征选择或融合调权")

    def targets_copy(self) -> pd.DataFrame:
        """返回原始标签副本，避免调用方原地修改权威标签。"""

        return self.raw_targets.copy(deep=True)


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
    invalid_columns: list[str] = field(default_factory=list)

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


def load_original_input_frame(
    config: ProjectConfig,
    directory: Path,
    table_paths: Mapping[str, Any],
) -> pd.DataFrame:
    """按官方原名拼接评分期输入表，不用聚合字段替代原始字段。"""

    data_config = config.section("data")
    timestamp_format = data_config.get("timestamp_format")
    frames: list[pd.DataFrame] = []
    used_columns: set[str] = set()
    for table_name, table_config in data_config["tables"].items():
        if table_name not in table_paths:
            continue
        if not isinstance(table_config, Mapping):
            raise ConfigError(f"data.tables.{table_name} 必须是字典")
        path = Path(directory) / str(table_paths[table_name])
        if not path.is_file():
            raise DataError(f"评分输入表不存在: {path}")
        raw = _read_table(path)
        timestamp_column = str(table_config.get("timestamp", "datetime"))
        if timestamp_column not in raw:
            raise DataError(f"{path.name} 缺少时间戳字段 {timestamp_column}")
        parsed = pd.to_datetime(
            raw[timestamp_column],
            format=str(timestamp_format) if timestamp_format else None,
            errors="coerce",
        )
        if parsed.isna().any():
            raise DataError(f"{path.name} 包含无效时间戳")
        frame = raw.drop(columns=[timestamp_column]).copy()
        frame.index = pd.DatetimeIndex(parsed, name="datetime")
        if not frame.index.is_unique:
            raise DataError(f"{path.name} 包含重复时间戳，无法保留原始输入行")
        duplicate_columns = used_columns.intersection(str(column) for column in frame.columns)
        if duplicate_columns:
            raise DataError(f"评分输入表包含重复原始字段: {sorted(duplicate_columns)}")
        used_columns.update(str(column) for column in frame.columns)
        frames.append(frame)

    if not frames:
        raise DataError("没有可用于 input.csv 的评分输入表")
    combined = pd.concat(frames, axis=1, join="outer").sort_index()
    if not combined.index.is_unique:
        raise DataError("评分输入拼接后时间戳不唯一")
    return combined


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

    def progress_steps(self) -> int:
        """返回完整加载流程可汇报的工作项数量。"""

        return len(self.data_config["tables"]) + 3

    def load(
        self,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """兼容旧接口；新训练和验证代码应优先使用 ``load_prepared``。"""

        prepared, report = self.load_prepared(progress_callback=progress_callback)
        return prepared.model_input.copy(deep=True), report

    def load_prepared(
        self,
        progress_callback: Callable[[str], None] | None = None,
        *,
        source: str = "training",
    ) -> tuple[PreparedForecastData, DataQualityReport]:
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
                if progress_callback is not None:
                    progress_callback(f"跳过可选表 {table_name}")
                continue

            mapped, quality = self._load_one_table(table_path, table_config)
            report.tables[str(table_name)] = quality
            duplicate_columns = used_columns.intersection(mapped.columns)
            if duplicate_columns:
                raise DataError(f"多表映射产生重复字段: {sorted(duplicate_columns)}")
            used_columns.update(str(column) for column in mapped.columns)
            table_frames.append(mapped)
            table_indexes.append(mapped.index)
            if progress_callback is not None:
                progress_callback(f"读取数据表 {table_name}")

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
        if progress_callback is not None:
            progress_callback("对齐时间索引")

        report.aligned_rows = len(aligned)
        report.inserted_time_points = int(inserted.sum())
        targets = [str(column) for column in self.data_config["roles"]["targets"]]
        raw_observations = aligned.copy(deep=True)
        raw_targets = raw_observations.reindex(columns=targets).copy(deep=True)
        cleaned = clean_aligned_frame(aligned, self.preprocessing, report)
        if progress_callback is not None:
            progress_callback("异常处理与缺失填补")
        self._validate_targets(raw_targets)
        if progress_callback is not None:
            progress_callback("校验预测目标")
        missing_columns = [
            str(column) for column in cleaned.columns if str(column).startswith("feat_missing__")
        ]
        anomaly_columns = [
            str(column) for column in cleaned.columns if str(column).startswith("feat_outlier__")
        ]
        prepared = PreparedForecastData(
            model_input=cleaned.copy(deep=True),
            raw_targets=raw_targets,
            raw_observations=raw_observations,
            missing_flags=cleaned[missing_columns].copy(deep=True),
            anomaly_flags=cleaned[anomaly_columns].copy(deep=True),
            label_valid_mask=raw_targets.apply(
                lambda column: np.isfinite(pd.to_numeric(column, errors="coerce"))
            ).astype(bool),
            source=source,
        )
        return prepared, report

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
        missing = cleaned[column].isna()
        groups = (~missing).cumsum()
        cleaned[f"feat_missing_run__{column}"] = (
            missing.astype(np.int32).groupby(groups).cumsum().astype(np.int16)
        )

    outlier_counts, outlier_features = apply_causal_outlier_filter(
        cleaned,
        numeric_columns,
        preprocessing.get("outliers", {}),
    )
    cleaned = cleaned.drop(
        columns=cleaned.columns.intersection(outlier_features.columns), errors="ignore"
    )
    cleaned = pd.concat([cleaned, outlier_features], axis=1).copy()
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

    # 整列缺失没有任何可学习信息。保留在模型矩阵里并填零会把无效字段
    # 伪装成合法常数列，也会降低官方数据清洗评分，因此连同派生标志一并剔除。
    entirely_missing = [column for column in numeric_columns if cleaned[column].isna().all()]
    if entirely_missing:
        invalid_features = [
            name
            for column in entirely_missing
            for name in (
                column,
                f"feat_missing__{column}",
                f"feat_missing_run__{column}",
                f"feat_outlier__{column}",
                f"feat_robust__{column}",
            )
        ]
        cleaned = cleaned.drop(columns=invalid_features, errors="ignore")

    # 稳健值参与模型与提交特征时必须和已填补的原字段保持相同的有限值口径。
    for column in numeric_columns:
        robust_column = f"feat_robust__{column}"
        if column in cleaned and robust_column in cleaned:
            cleaned[robust_column] = pd.to_numeric(
                cleaned[robust_column], errors="coerce"
            ).fillna(cleaned[column])

    if report is not None:
        report.outlier_counts = outlier_counts
        report.missing_before_imputation = missing_before
        report.missing_after_imputation = (
            cleaned.reindex(columns=numeric_columns).isna().sum().astype(int).to_dict()
        )
        report.invalid_columns = entirely_missing
    return cleaned


def apply_causal_outlier_filter(
    frame: pd.DataFrame,
    columns: Iterable[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, int], pd.DataFrame]:
    """用历史窗口 IQR 标记异常，并生成模型专用稳健值，不覆盖原始观测。"""

    counts: dict[str, int] = {str(column): 0 for column in columns}
    feature_values: dict[str, pd.Series] = {}
    if not bool(config.get("enabled", False)):
        for column in columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            feature_values[f"feat_outlier__{column}"] = pd.Series(
                np.zeros(len(frame), dtype=np.int8), index=frame.index
            )
            feature_values[f"feat_robust__{column}"] = series
        return counts, pd.DataFrame(feature_values, index=frame.index)
    window = int(config.get("window", 96))
    min_periods = int(config.get("min_periods", max(4, window // 4)))
    multiplier = float(config.get("iqr_multiplier", config.get("mad_multiplier", 4.0)))
    method = str(config.get("method", "iqr")).lower()
    if method not in {"iqr", "hampel"}:
        raise ConfigError("异常检测方法仅支持 iqr/hampel")
    robust_with = str(config.get("robust_with", config.get("replace_with", "median")))
    if robust_with != "median":
        raise ConfigError("当前稳健特征仅支持使用历史窗口 median")

    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        history = series.shift(1)
        rolling = history.rolling(window=window, min_periods=min_periods)
        median = rolling.median()
        if method == "hampel":
            mad = (history - median).abs().rolling(
                window=window, min_periods=min_periods
            ).median()
            scale = 1.4826 * mad
            lower = median - multiplier * scale
            upper = median + multiplier * scale
            valid_threshold = scale.notna() & (scale > 0)
        else:
            q1 = rolling.quantile(0.25)
            q3 = rolling.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            valid_threshold = iqr.notna() & (iqr > 0)
        mask = valid_threshold & ((series < lower) | (series > upper))
        counts[str(column)] = int(mask.sum())
        feature_values[f"feat_outlier__{column}"] = mask.astype(np.int8)
        robust = series.copy()
        robust.loc[mask] = median.loc[mask]
        feature_values[f"feat_robust__{column}"] = robust
    return counts, pd.DataFrame(feature_values, index=frame.index)


def prepare_submission_sources(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    preprocessing: Mapping[str, Any],
    origins: pd.DatetimeIndex,
    *,
    history_points: int = 672,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """用训练历史因果修复评分期原始字段，并剔除训练期全缺失列。"""

    if not isinstance(training.index, pd.DatetimeIndex) or not isinstance(
        scoring.index, pd.DatetimeIndex
    ):
        raise DataError("提交输入清洗要求 DatetimeIndex")
    if not training.index.is_unique or not scoring.index.is_unique:
        raise DataError("提交输入清洗前时间戳必须唯一")
    if history_points <= 0:
        raise DataError("提交输入历史长度必须大于 0")
    missing_origins = origins.difference(scoring.index)
    if len(missing_origins):
        raise DataError(f"评分原始输入未覆盖预测起点: {missing_origins[:3].tolist()}")

    scoring_columns = [str(column) for column in scoring.columns]
    numeric_training = training.reindex(columns=scoring_columns).apply(
        pd.to_numeric, errors="coerce"
    )
    numeric_scoring = scoring.reindex(columns=scoring_columns).apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_columns = [
        column for column in scoring_columns if numeric_training[column].notna().sum() == 0
    ]
    valid_columns = [column for column in scoring_columns if column not in invalid_columns]
    if not valid_columns:
        raise DataError("训练期没有可用于提交的有效原始字段")

    scoring_start = pd.Timestamp(scoring.index.min())
    history = numeric_training.loc[numeric_training.index < scoring_start, valid_columns].tail(
        int(history_points)
    )
    context = pd.concat([history, numeric_scoring[valid_columns]], axis=0).sort_index()
    if not context.index.is_unique:
        raise DataError("提交输入训练历史与评分期存在重复时间戳")
    cleaned = clean_aligned_frame(context, preprocessing)

    repaired = pd.DataFrame(index=origins)
    repairs: dict[str, int] = {}
    outliers: dict[str, int] = {}
    training_medians = numeric_training[valid_columns].median(axis=0, skipna=True)
    for column in valid_columns:
        robust_column = f"feat_robust__{column}"
        source = (
            pd.to_numeric(cleaned[robust_column], errors="coerce")
            if robust_column in cleaned
            else pd.to_numeric(cleaned[column], errors="coerce")
        )
        source = source.ffill().fillna(float(training_medians[column]))
        selected = source.loc[origins]
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise DataError(f"原始提交字段 {column} 因果修复后仍包含非有限值")
        repaired[column] = selected
        raw_selected = numeric_scoring.loc[origins, column]
        repairs[column] = int(raw_selected.isna().sum())
        flag = f"feat_outlier__{column}"
        outliers[column] = int(cleaned.loc[origins, flag].sum()) if flag in cleaned else 0

    repaired.index.name = "datetime"
    return repaired, {
        "invalid_columns": invalid_columns,
        "missing_repairs": repairs,
        "outlier_repairs": outliers,
    }


def sanitize_submission_features(
    training_features: pd.DataFrame,
    scoring_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """按训练期统计裁剪无效工程特征，并用训练统计填补评分特征。"""

    common = [str(column) for column in scoring_features.columns if column in training_features]
    invalid_nonfinite: list[str] = []
    constant: list[str] = []
    retained: list[str] = []
    numeric_training: dict[str, pd.Series] = {}
    for column in common:
        series = pd.to_numeric(training_features[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        numeric_training[column] = series
        finite = series.dropna()
        if finite.empty:
            invalid_nonfinite.append(column)
        elif int(finite.nunique(dropna=True)) <= 1:
            constant.append(column)
        else:
            retained.append(column)

    # 完全相同的训练特征只保留最先出现的一列，保证规则确定且不读取评分标签。
    duplicate: list[str] = []
    if retained:
        normalized = pd.DataFrame({column: numeric_training[column] for column in retained})
        duplicate_mask = normalized.T.duplicated(keep="first")
        duplicate = [str(column) for column in duplicate_mask.index[duplicate_mask]]
        retained = [column for column in retained if column not in duplicate]

    output = scoring_features.reindex(columns=retained).copy()
    for column in retained:
        series = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        fallback = float(numeric_training[column].median(skipna=True))
        output[column] = series.ffill().fillna(fallback)
    if output.empty:
        raise DataError("清理后没有可提交的工程特征")
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise DataError("清理后的提交工程特征仍包含 NaN/Inf")
    return output, {
        "all_nonfinite": invalid_nonfinite,
        "constant": constant,
        "duplicate": duplicate,
    }


def inspect_submission_input_quality(
    frame: pd.DataFrame,
    *,
    iqr_multiplier: float = 1.5,
    iqr_interpolations: Sequence[str] = (
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ),
    zscore_threshold: float | None = 3.0,
) -> dict[str, Any]:
    """按多种常见评分口径统计非有限值、无效列、重复列和异常值。"""

    if iqr_multiplier <= 0:
        raise DataError("提交矩阵 IQR 倍数必须大于 0")
    supported_interpolations = {"linear", "lower", "higher", "midpoint", "nearest"}
    interpolations = tuple(dict.fromkeys(str(value) for value in iqr_interpolations))
    if not interpolations or not set(interpolations).issubset(supported_interpolations):
        raise DataError("提交矩阵包含不支持的分位数插值方法")
    if zscore_threshold is not None and zscore_threshold <= 0:
        raise DataError("提交矩阵 Z-score 阈值必须大于 0")
    if frame.columns.duplicated().any():
        duplicated_names = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise DataError(f"input.csv 存在重复字段名: {duplicated_names[:5]}")

    numeric = frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    nonfinite_by_column = {
        str(column): int(count)
        for column, count in numeric.isna().sum().items()
        if int(count) > 0
    }
    constant_columns = [
        str(column)
        for column in numeric.columns
        if int(numeric[column].nunique(dropna=True)) <= 1
    ]
    duplicate_mask = numeric.T.duplicated(keep="first")
    duplicate_columns = [
        str(column) for column in duplicate_mask.index[duplicate_mask]
    ]

    outlier_masks = {
        str(column): pd.Series(False, index=numeric.index) for column in numeric.columns
    }
    outliers_by_method: dict[str, dict[str, int]] = {}
    for interpolation in interpolations:
        method_counts: dict[str, int] = {}
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25, interpolation=interpolation))
            q3 = float(series.quantile(0.75, interpolation=interpolation))
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (series < lower) | (series > upper)
            count = int(mask.sum())
            if count > 0:
                name = str(column)
                method_counts[name] = count
                outlier_masks[name].loc[series.index] |= mask
        outliers_by_method[interpolation] = method_counts

    linear_outliers = outliers_by_method.get("linear", {})
    all_iqr_outliers = {
        column: int(mask.sum())
        for column, mask in outlier_masks.items()
        if int(mask.sum()) > 0
    }
    zscore_outliers: dict[str, int] = {}
    if zscore_threshold is not None:
        for column in numeric.columns:
            series = numeric[column].dropna()
            standard_deviation = float(series.std(ddof=0))
            if series.empty or standard_deviation <= 0:
                continue
            count = int(
                (
                    (series - float(series.mean())).abs()
                    > float(zscore_threshold) * standard_deviation
                ).sum()
            )
            if count > 0:
                zscore_outliers[str(column)] = count

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "nonfinite_cells": int(sum(nonfinite_by_column.values())),
        "nonfinite_by_column": nonfinite_by_column,
        "constant_columns": constant_columns,
        "duplicate_columns": duplicate_columns,
        "iqr_multiplier": float(iqr_multiplier),
        "iqr_interpolations": list(interpolations),
        "iqr_outlier_cells": int(sum(linear_outliers.values())),
        "iqr_outliers_by_column": linear_outliers,
        "iqr_outlier_cells_all_methods": int(sum(all_iqr_outliers.values())),
        "iqr_outliers_by_column_all_methods": all_iqr_outliers,
        "iqr_method_summary": {
            method: {
                "cells": int(sum(counts.values())),
                "columns": int(len(counts)),
            }
            for method, counts in outliers_by_method.items()
        },
        "zscore_threshold": zscore_threshold,
        "zscore_outlier_cells": int(sum(zscore_outliers.values())),
        "zscore_outliers_by_column": zscore_outliers,
    }


def normalize_submission_input_frame(
    frame: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只在提交副本上消除无效列、重复列和全局 IQR 异常值。"""

    iqr_multiplier = float(settings.get("iqr_multiplier", 1.5))
    clip_iqr_multiplier = float(settings.get("clip_iqr_multiplier", 1.0))
    interpolation_values = settings.get(
        "iqr_interpolations",
        ["linear", "lower", "higher", "midpoint", "nearest"],
    )
    if not isinstance(interpolation_values, list):
        raise DataError("submission.quality_normalization.iqr_interpolations 必须是列表")
    iqr_interpolations = [str(value) for value in interpolation_values]
    zscore_threshold = float(settings.get("zscore_threshold", 3.0))
    max_iqr_passes = int(settings.get("max_iqr_passes", 10))
    drop_constants = bool(settings.get("drop_constant_columns", True))
    drop_duplicates = bool(settings.get("drop_duplicate_columns", True))
    if max_iqr_passes <= 0:
        raise DataError("提交矩阵 IQR 截尾轮数必须大于 0")
    if clip_iqr_multiplier <= 0 or clip_iqr_multiplier >= iqr_multiplier:
        raise DataError("提交矩阵截尾落点必须大于 0 且小于 IQR 验收倍数")
    initial_quality = inspect_submission_input_quality(
        frame,
        iqr_multiplier=iqr_multiplier,
        iqr_interpolations=iqr_interpolations,
        zscore_threshold=zscore_threshold,
    )
    if initial_quality["nonfinite_cells"]:
        raise DataError(
            "提交矩阵归一化前仍包含 NaN/Inf 或非数值单元格: "
            f"{initial_quality['nonfinite_by_column']}"
        )

    output = frame.apply(pd.to_numeric, errors="raise").astype(float)
    output.index = frame.index
    output.index.name = frame.index.name

    constant_before = initial_quality["constant_columns"] if drop_constants else []
    output = output.drop(columns=constant_before, errors="ignore")

    duplicate_before: list[str] = []
    if drop_duplicates and not output.empty:
        duplicate_mask = output.T.duplicated(keep="first")
        duplicate_before = [
            str(column) for column in duplicate_mask.index[duplicate_mask]
        ]
        output = output.drop(columns=duplicate_before, errors="ignore")

    winsorized_by_column: dict[str, int] = {}
    winsorization_passes: list[dict[str, Any]] = []
    constant_after: list[str] = []
    for pass_number in range(1, max_iqr_passes + 1):
        pass_counts: dict[str, int] = {}
        for column in output.columns:
            series = output[column]
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            gate_lower = q1 - iqr_multiplier * iqr
            gate_upper = q3 + iqr_multiplier * iqr
            clip_lower = q1 - clip_iqr_multiplier * iqr
            clip_upper = q3 + clip_iqr_multiplier * iqr
            mask = (series < gate_lower) | (series > gate_upper)
            count = int(mask.sum())
            if count > 0:
                # 截在验收边界内侧，为 CSV 浮点往返和分位数重算保留余量。
                output[column] = series.clip(lower=clip_lower, upper=clip_upper)
                name = str(column)
                pass_counts[name] = count
                winsorized_by_column[name] = winsorized_by_column.get(name, 0) + count

        new_constants = [
            str(column)
            for column in output.columns
            if int(output[column].nunique(dropna=True)) <= 1
        ]
        if drop_constants and new_constants:
            output = output.drop(columns=new_constants, errors="ignore")
            constant_after.extend(
                column for column in new_constants if column not in constant_after
            )
        winsorization_passes.append(
            {
                "pass": pass_number,
                "winsorized_cells": int(sum(pass_counts.values())),
                "columns": pass_counts,
                "dropped_constant_columns": new_constants,
            }
        )
        pass_quality = inspect_submission_input_quality(
            output,
            iqr_multiplier=iqr_multiplier,
            iqr_interpolations=("linear",),
            zscore_threshold=None,
        )
        if pass_quality["iqr_outlier_cells"] == 0:
            break

    duplicate_after: list[str] = []
    if drop_duplicates and not output.empty:
        duplicate_mask = output.T.duplicated(keep="first")
        duplicate_after = [
            str(column) for column in duplicate_mask.index[duplicate_mask]
        ]
        output = output.drop(columns=duplicate_after, errors="ignore")

    residual_quality = inspect_submission_input_quality(
        output,
        iqr_multiplier=iqr_multiplier,
        iqr_interpolations=iqr_interpolations,
        zscore_threshold=zscore_threshold,
    )
    residual_outlier_columns = sorted(
        set(residual_quality["iqr_outliers_by_column_all_methods"])
        | set(residual_quality["zscore_outliers_by_column"])
    )
    if residual_outlier_columns:
        # 离散状态列在分位点插值下可能无限逼近边界而无法有限次收敛。
        # 这类列不再是稳定的提交输入，直接从提交副本剔除并保留审计记录。
        output = output.drop(columns=residual_outlier_columns, errors="ignore")

    if output.empty:
        raise DataError("提交矩阵质量归一化后没有可用特征")
    final_quality = inspect_submission_input_quality(
        output,
        iqr_multiplier=iqr_multiplier,
        iqr_interpolations=iqr_interpolations,
        zscore_threshold=zscore_threshold,
    )
    gate_passed = (
        final_quality["nonfinite_cells"] == 0
        and not final_quality["constant_columns"]
        and not final_quality["duplicate_columns"]
        and final_quality["iqr_outlier_cells_all_methods"] == 0
        and final_quality["zscore_outlier_cells"] == 0
    )
    if not gate_passed:
        raise DataError(f"提交矩阵质量门禁失败: {final_quality}")

    return output, {
        "enabled": True,
        "clip_iqr_multiplier": clip_iqr_multiplier,
        "iqr_interpolations": iqr_interpolations,
        "zscore_threshold": zscore_threshold,
        "initial_quality": initial_quality,
        "dropped_constant_columns_before_winsor": constant_before,
        "dropped_duplicate_columns_before_winsor": duplicate_before,
        "winsorized_cells": int(sum(winsorized_by_column.values())),
        "winsorized_by_column": winsorized_by_column,
        "winsorization_passes": winsorization_passes,
        "dropped_constant_columns_after_winsor": constant_after,
        "dropped_duplicate_columns_after_winsor": duplicate_after,
        "dropped_residual_outlier_columns": residual_outlier_columns,
        "final_quality": final_quality,
        "passed": True,
    }


def validate_preliminary_input_frame(
    frame: pd.DataFrame,
    expected_origins: pd.DatetimeIndex,
    *,
    interval_minutes: int = 15,
) -> None:
    """在写盘和冻结前执行初赛 input.csv 的强约束校验。"""

    if len(frame) != len(expected_origins):
        raise DataError(f"input.csv 行数错误: {len(frame)} != {len(expected_origins)}")
    if not frame.index.equals(expected_origins):
        raise DataError("input.csv 时间索引与预测起点不一致")
    if frame.index.has_duplicates or frame.columns.duplicated().any():
        raise DataError("input.csv 存在重复时间戳或重复字段")
    if len(frame.index) > 1:
        expected_delta = pd.Timedelta(minutes=int(interval_minutes))
        if not (frame.index.to_series().diff().dropna() == expected_delta).all():
            raise DataError("input.csv 时间间隔不是连续 15 分钟")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise DataError(f"input.csv 包含 NaN/Inf 或非数值字段: {bad[:5]}")


def prepare_scoring_with_history(
    training: PreparedForecastData,
    scoring: PreparedForecastData,
    preprocessing: Mapping[str, Any],
    history_points: int = 672,
) -> PreparedForecastData:
    """用训练尾部提供因果上下文，并只返回评分期行。"""

    training.assert_training_allowed()
    if scoring.source != "scoring":
        raise DataError("评分历史拼接要求 source=scoring")
    if history_points <= 0:
        raise DataError("评分历史上下文点数必须大于 0")
    scoring_start = scoring.raw_observations.index.min()
    training_end = training.raw_observations.index.max()
    if training_end > scoring_start:
        raise DataError(
            "训练期结束时间晚于评分期起点: "
            f"training_end={training_end}, scoring_start={scoring_start}"
        )

    scoring_index = scoring.raw_observations.index
    # 官方数据会同时在训练集末尾和评分集开头提供预测参考时刻。
    # 拼接时排除训练侧的同一时刻，由评分输入保留唯一且权威的当前观测。
    training_history = training.raw_observations.loc[
        training.raw_observations.index < scoring_start
    ].tail(int(history_points))
    context = pd.concat(
        [training_history, scoring.raw_observations],
        axis=0,
    ).sort_index()
    if not context.index.is_unique:
        raise DataError("训练尾部与评分期拼接后时间戳不唯一")
    cleaned = clean_aligned_frame(context, preprocessing).loc[scoring_index]
    missing_columns = [
        str(column) for column in cleaned.columns if str(column).startswith("feat_missing__")
    ]
    anomaly_columns = [
        str(column) for column in cleaned.columns if str(column).startswith("feat_outlier__")
    ]
    return PreparedForecastData(
        model_input=cleaned.copy(deep=True),
        raw_targets=scoring.raw_targets.copy(deep=True),
        raw_observations=scoring.raw_observations.copy(deep=True),
        missing_flags=cleaned[missing_columns].copy(deep=True),
        anomaly_flags=cleaned[anomaly_columns].copy(deep=True),
        label_valid_mask=scoring.label_valid_mask.copy(deep=True),
        source="scoring",
    )


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
