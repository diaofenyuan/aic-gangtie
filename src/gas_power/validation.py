"""严格按时间顺序的 expanding/rolling 验证。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from tqdm.auto import tqdm

from gas_power.features import CausalFeatureBuilder, LeakageError
from gas_power.metrics import largest_errors, summarize_predictions
from gas_power.models.base import ForecastModel, prediction_column
from gas_power.time_semantics import validate_target_times


@dataclass(frozen=True)
class TimeSplit:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_origins: pd.DatetimeIndex


class TimeSeriesRollingSplitter:
    def __init__(
        self,
        mode: str,
        folds: int,
        initial_train_points: int,
        validation_points: int,
        step_points: int,
        rolling_train_points: int | None = None,
    ):
        if mode not in {"expanding", "rolling"}:
            raise ValueError("验证模式必须是 expanding 或 rolling")
        self.mode = mode
        self.folds = int(folds)
        self.initial_train_points = int(initial_train_points)
        self.validation_points = int(validation_points)
        self.step_points = int(step_points)
        self.rolling_train_points = int(rolling_train_points or initial_train_points)
        if min(self.folds, self.initial_train_points, self.validation_points, self.step_points) <= 0:
            raise ValueError("验证器各点数配置必须大于 0")

    def split(self, index: pd.DatetimeIndex, max_horizon: int) -> list[TimeSplit]:
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("滚动验证索引必须严格递增且唯一")
        splits: list[TimeSplit] = []
        for fold in range(self.folds):
            train_end_position = self.initial_train_points - 1 + fold * self.step_points
            validation_start = train_end_position + 1
            validation_end = validation_start + self.validation_points - 1
            latest_origin = len(index) - int(max_horizon) - 1
            if validation_end > latest_origin:
                break
            train_start_position = 0
            if self.mode == "rolling":
                train_start_position = max(
                    0, train_end_position - self.rolling_train_points + 1
                )
            origins = index[validation_start : validation_end + 1]
            split = TimeSplit(
                fold=fold,
                train_start=pd.Timestamp(index[train_start_position]),
                train_end=pd.Timestamp(index[train_end_position]),
                validation_origins=pd.DatetimeIndex(origins),
            )
            self._assert_order(split)
            splits.append(split)
        if len(splits) != self.folds:
            raise ValueError(
                f"数据长度不足以生成 {self.folds} 折验证，仅能生成 {len(splits)} 折；"
                "请调整 initial_train_points/validation_points/step_points"
            )
        return splits

    @staticmethod
    def _assert_order(split: TimeSplit) -> None:
        if split.train_start > split.train_end:
            raise LeakageError("训练起点晚于训练终点")
        if len(split.validation_origins) == 0:
            raise LeakageError("验证起点为空")
        if split.train_end >= split.validation_origins.min():
            raise LeakageError("训练区间与验证起点重叠")


class RecentWindowSplitter:
    """从训练尾部向前构造互不重叠的完整两天验证折。"""

    def __init__(
        self,
        folds: int = 10,
        validation_points: int = 192,
        step_points: int = 192,
        rolling_train_points: int | None = None,
    ):
        self.folds = int(folds)
        self.validation_points = int(validation_points)
        self.step_points = int(step_points)
        self.rolling_train_points = (
            int(rolling_train_points) if rolling_train_points is not None else None
        )
        if min(self.folds, self.validation_points, self.step_points) <= 0:
            raise ValueError("最近窗口验证器各点数配置必须大于 0")

    def split(self, index: pd.DatetimeIndex, max_horizon: int) -> list[TimeSplit]:
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("最近窗口验证索引必须严格递增且唯一")
        latest_end = len(index) - int(max_horizon) - 1
        starts = [
            latest_end - self.validation_points + 1 - offset * self.step_points
            for offset in reversed(range(self.folds))
        ]
        splits: list[TimeSplit] = []
        for fold, validation_start in enumerate(starts):
            validation_end = validation_start + self.validation_points - 1
            train_end_position = validation_start - 1
            if validation_start <= 0 or validation_end > latest_end:
                raise ValueError("数据长度不足以生成完整的最近两天验证折")
            train_start_position = 0
            if self.rolling_train_points is not None:
                train_start_position = max(
                    0, train_end_position - self.rolling_train_points + 1
                )
            split = TimeSplit(
                fold=fold,
                train_start=pd.Timestamp(index[train_start_position]),
                train_end=pd.Timestamp(index[train_end_position]),
                validation_origins=pd.DatetimeIndex(
                    index[validation_start : validation_end + 1]
                ),
            )
            TimeSeriesRollingSplitter._assert_order(split)
            splits.append(split)
        return splits


@dataclass
class ValidationArtifacts:
    metrics: pd.DataFrame
    worst_errors: pd.DataFrame
    predictions: pd.DataFrame
    splits: list[TimeSplit]


@dataclass
class DetailedValidationArtifacts:
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    condition_metrics: pd.DataFrame
    horizon_curve: pd.DataFrame
    baseline_gain: pd.DataFrame
    coverage: dict[str, object]


def assert_model_causality(
    model: ForecastModel,
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    targets: Sequence[str],
    horizons: Sequence[int],
) -> None:
    """扰动预测起点之后的数据，当前起点的预测必须完全不变。"""

    origins = pd.DatetimeIndex([origin])
    baseline = model.predict(frame, origins, targets, horizons)
    perturbed = frame.copy()
    future = perturbed.index > origin
    for column in perturbed.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(perturbed.loc[future, column], errors="coerce")
        perturbed.loc[future, column] = values * -11.0 + 9_999_991.0
    candidate = model.predict(perturbed, origins, targets, horizons)
    try:
        assert_frame_equal(baseline, candidate, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise LeakageError(f"未来数据扰动改变了当前起点预测: {exc}") from exc


def run_rolling_validation(
    frame: pd.DataFrame,
    model_factory: Callable[[], ForecastModel],
    splitter: TimeSeriesRollingSplitter,
    target_columns: Sequence[str],
    horizons: Sequence[int],
    interval_minutes: int,
    near_zero_threshold: float,
    worst_error_count: int,
    feature_builder: CausalFeatureBuilder | None = None,
    show_progress: bool = False,
    progress_description: str = "滚动验证",
    raw_targets: pd.DataFrame | None = None,
    feature_matrix: pd.DataFrame | None = None,
    data_source: str = "training",
) -> ValidationArtifacts:
    if data_source == "scoring":
        raise LeakageError("评分期 scoring 数据禁止用于滚动验证或模型选择")
    horizons = [int(value) for value in horizons]
    splits = splitter.split(frame.index, max(horizons))
    labels = frame[list(target_columns)] if raw_targets is None else raw_targets
    missing_label_columns = set(map(str, target_columns)).difference(labels.columns)
    if missing_label_columns:
        raise ValueError(f"原始标签缺少目标字段: {sorted(missing_label_columns)}")
    condition_features = (
        feature_matrix.reindex(frame.index)
        if feature_matrix is not None
        else (
            feature_builder.transform(frame)
            if feature_builder is not None
            else pd.DataFrame(index=frame.index)
        )
    )
    condition_columns = [
        column
        for column in condition_features.columns
        if column.startswith("feat_state_")
        or column.startswith("feat_gas_balance_")
        or column.startswith("feat_holder_change_rate_")
        or column.startswith("feat_missing__")
        or column == "feat_time_gap_inserted"
    ]

    tidy_rows: list[dict[str, object]] = []
    offset = pd.Timedelta(minutes=interval_minutes)
    split_iterator = tqdm(
        splits,
        total=len(splits),
        desc=progress_description,
        unit="折",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for split in split_iterator:
        model = model_factory()
        training = frame.loc[split.train_start : split.train_end]
        fit_steps = model.fit_progress_steps(target_columns, horizons)
        fit_progress = tqdm(
            total=fit_steps,
            desc=f"{progress_description} 第 {split.fold + 1}/{len(splits)} 折训练",
            unit="步",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress or fit_steps is None,
        )

        def advance_fit(label: str) -> None:
            fit_progress.set_postfix_str(label, refresh=False)
            fit_progress.update(1)

        model.set_fit_progress_callback(advance_fit if show_progress else None)
        try:
            model.fit(
                training,
                target_columns,
                horizons,
                train_end=split.train_end,
                raw_targets=labels.loc[split.train_start : split.train_end],
                feature_matrix=(
                    condition_features.loc[training.index]
                    if feature_matrix is not None
                    else None
                ),
                data_source=data_source,
            )
        finally:
            model.set_fit_progress_callback(None)
            fit_progress.close()
        predictions = model.predict(frame, split.validation_origins, target_columns, horizons)
        if split.fold == 0:
            assert_model_causality(
                model,
                frame,
                pd.Timestamp(split.validation_origins[0]),
                target_columns,
                [horizons[0], horizons[-1]],
            )

        for origin in split.validation_origins:
            conditions = {
                column: condition_features.at[origin, column] for column in condition_columns
            }
            for target in target_columns:
                for horizon in horizons:
                    target_time = pd.Timestamp(origin) + horizon * offset
                    row: dict[str, object] = {
                        "fold": split.fold,
                        "origin": pd.Timestamp(origin),
                        "target_datetime": target_time,
                        "target": str(target),
                        "horizon_steps": horizon,
                        "horizon_minutes": horizon * interval_minutes,
                        "y_true": labels.at[target_time, str(target)],
                        "y_pred": predictions.at[
                            origin,
                            prediction_column(str(target), horizon, interval_minutes),
                        ],
                    }
                    row.update(conditions)
                    row["condition_month_change"] = int(pd.Timestamp(origin).month != target_time.month)
                    row["condition_weekend"] = int(pd.Timestamp(origin).dayofweek >= 5)
                    tidy_rows.append(row)

    tidy = pd.DataFrame(tidy_rows)
    validate_target_times(
        tidy["origin"], tidy["target_datetime"], tidy["horizon_steps"], interval_minutes
    )
    metrics = summarize_predictions(tidy, near_zero_threshold=near_zero_threshold)
    worst = largest_errors(tidy, count=worst_error_count)
    return ValidationArtifacts(metrics=metrics, worst_errors=worst, predictions=tidy, splits=splits)


def _operating_condition(row: pd.Series) -> str:
    target = str(row["target"])
    for suffix, name in (
        ("startup", "startup"),
        ("shutdown", "shutdown"),
        ("ramp_up", "ramp_up"),
        ("ramp_down", "ramp_down"),
        ("stable", "stable"),
    ):
        if float(row.get(f"feat_state_{target}_{suffix}", 0.0) or 0.0) > 0.5:
            return name
    missing = sum(
        float(row.get(column, 0.0) or 0.0)
        for column in row.index
        if str(column).startswith("feat_missing__")
    )
    if missing > 0 or float(row.get("feat_time_gap_inserted", 0.0) or 0.0) > 0.5:
        return "high_missing"
    if int(row.get("condition_month_change", 0)) == 1:
        return "month_change"
    return "normal"


def summarize_detailed_validation(
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    near_zero_threshold: float,
) -> DetailedValidationArtifacts:
    """汇总每折、最差折、工况、步长曲线和相对最后值增益。"""

    fold_parts: list[pd.DataFrame] = []
    for fold, group in predictions.groupby("fold", sort=True):
        metrics = summarize_predictions(group, near_zero_threshold)
        metrics.insert(0, "fold", int(fold))
        fold_parts.append(metrics)
    fold_metrics = pd.concat(fold_parts, ignore_index=True)
    group_columns = ["scope", "target", "horizon_steps", "horizon_minutes"]
    aggregate = (
        fold_metrics.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_worst=("mape", "max"),
            score_mean=("score_1_mape", "mean"),
            score_std=("score_1_mape", "std"),
            score_worst=("score_1_mape", "min"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            bias_mean=("bias", "mean"),
            folds=("fold", "nunique"),
        )
        .fillna({"mape_std": 0.0, "score_std": 0.0})
    )

    conditioned = predictions.copy()
    conditioned["condition"] = conditioned.apply(_operating_condition, axis=1)
    condition_parts: list[pd.DataFrame] = []
    for condition, group in conditioned.groupby("condition", sort=True):
        metrics = summarize_predictions(group, near_zero_threshold)
        metrics.insert(0, "condition", str(condition))
        condition_parts.append(metrics)
    condition_metrics = pd.concat(condition_parts, ignore_index=True) if condition_parts else pd.DataFrame()

    horizon_curve = summarize_predictions(predictions, near_zero_threshold)
    horizon_curve = horizon_curve.loc[horizon_curve["scope"] == "target_horizon"].copy()
    baseline_curve = summarize_predictions(baseline_predictions, near_zero_threshold)
    baseline_curve = baseline_curve.loc[baseline_curve["scope"] == "target_horizon", [
        "target", "horizon_steps", "mape", "score_1_mape"
    ]].rename(columns={"mape": "baseline_mape", "score_1_mape": "baseline_score"})
    gain = horizon_curve.merge(baseline_curve, on=["target", "horizon_steps"], how="left")
    gain["mape_gain_vs_last_value"] = gain["baseline_mape"] - gain["mape"]
    gain["score_gain_vs_last_value"] = gain["score_1_mape"] - gain["baseline_score"]

    missing_columns = [
        column for column in conditioned.columns if str(column).startswith("feat_missing__")
    ]
    missing_rows = (
        conditioned[missing_columns].fillna(0.0).sum(axis=1) > 0
        if missing_columns
        else pd.Series(False, index=conditioned.index)
    )
    startup_rows = sum(
        int(
            conditioned.loc[
                conditioned["target"].astype(str) == target,
                f"feat_state_{target}_startup",
            ].fillna(0.0).sum()
        )
        for target in conditioned["target"].astype(str).unique()
        if f"feat_state_{target}_startup" in conditioned
    )
    shutdown_rows = sum(
        int(
            conditioned.loc[
                conditioned["target"].astype(str) == target,
                f"feat_state_{target}_shutdown",
            ].fillna(0.0).sum()
        )
        for target in conditioned["target"].astype(str).unique()
        if f"feat_state_{target}_shutdown" in conditioned
    )
    coverage = {
        "conditions": conditioned["condition"].value_counts().astype(int).to_dict(),
        "months": sorted({str(pd.Timestamp(value).to_period("M")) for value in predictions["origin"]}),
        "weekend_rows": int(predictions.get("condition_weekend", pd.Series(dtype=int)).sum()),
        "month_change_rows": int(predictions.get("condition_month_change", pd.Series(dtype=int)).sum()),
        "high_missing_rows": int(missing_rows.sum()),
        "startup_rows": startup_rows,
        "shutdown_rows": shutdown_rows,
        "holiday": "官方数据未提供节假日/生产节奏标识，当前仅报告周末代理变量",
        "selection_priority": "优先最差折和跨月份稳定性，不以平均分作为唯一依据",
    }
    return DetailedValidationArtifacts(
        fold_metrics, aggregate, condition_metrics, horizon_curve, gain, coverage
    )


def splitter_from_config(config: Mapping[str, object]) -> TimeSeriesRollingSplitter:
    mode = str(config.get("mode", "expanding"))
    if mode == "recent":
        return RecentWindowSplitter(
            folds=int(config.get("folds", 10)),
            validation_points=int(config.get("validation_points", 192)),
            step_points=int(config.get("step_points", 192)),
            rolling_train_points=(
                int(config["rolling_train_points"])
                if config.get("rolling_train_points") is not None
                else None
            ),
        )  # type: ignore[return-value]
    return TimeSeriesRollingSplitter(
        mode=mode,
        folds=int(config.get("folds", 2)),
        initial_train_points=int(config.get("initial_train_points", 1152)),
        validation_points=int(config.get("validation_points", 96)),
        step_points=int(config.get("step_points", 192)),
        rolling_train_points=int(config.get("rolling_train_points", 1152)),
    )
