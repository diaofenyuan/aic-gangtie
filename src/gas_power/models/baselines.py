"""无需训练依赖的工业时序基线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from gas_power.models.base import (
    ForecastModel,
    prediction_column,
    validate_prediction_request,
)
from gas_power.runtime import current_worker_count


class HistoricalBaseline(ForecastModel):
    """基线不学习未来标签，fit 仅记录训练边界供审计。"""

    def __init__(self, interval_minutes: int = 15):
        self.interval_minutes = int(interval_minutes)
        self.train_end_: pd.Timestamp | None = None

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
    ) -> "HistoricalBaseline":
        if data_source == "scoring":
            raise ValueError("评分期 scoring 数据禁止用于模型拟合")
        if frame.empty:
            raise ValueError("基线模型不能在空数据上拟合")
        self.train_end_ = pd.Timestamp(train_end if train_end is not None else frame.index.max())
        return self

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        raise NotImplementedError

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        validate_prediction_request(frame, origins, target_columns, horizons)
        series_by_target = {
            str(target): pd.to_numeric(frame[str(target)], errors="coerce")
            for target in target_columns
        }
        tasks = [
            (str(target), int(horizon))
            for target in target_columns
            for horizon in horizons
        ]

        def predict_target_horizon(task: tuple[str, int]) -> tuple[str, list[float]]:
            target, horizon = task
            series = series_by_target[target]
            values: list[float] = []
            for origin in origins:
                # 每个预测起点都显式截断未来数据，线程之间只共享只读序列。
                history = series.loc[:origin].dropna()
                if history.empty:
                    raise ValueError(f"{target} 在起点 {origin} 之前没有有效历史")
                values.append(self._predict_value(history, pd.Timestamp(origin), horizon))
            return (
                prediction_column(target, horizon, self.interval_minutes),
                values,
            )

        workers = min(current_worker_count(), len(tasks))
        if workers <= 1:
            results = map(predict_target_horizon, tasks)
            prediction_values = dict(results)
        else:
            # CPU 密集的 Python 计算使用独立进程绕过 GIL，子进程内 BLAS 限制为单线程。
            with parallel_config(backend="loky", inner_max_num_threads=1):
                prediction_values = dict(
                    Parallel(n_jobs=workers)(
                        delayed(predict_target_horizon)(task) for task in tasks
                    )
                )
        output = pd.DataFrame(prediction_values, index=origins)
        output.index.name = "datetime"
        return output


class LastValueModel(HistoricalBaseline):
    """最后值保持。"""

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        return float(history.iloc[-1])


@dataclass
class _WindowSettings:
    window: int


class WindowMeanModel(HistoricalBaseline):
    """最近固定窗口均值。"""

    def __init__(self, window: int = 8, interval_minutes: int = 15):
        super().__init__(interval_minutes)
        if window <= 0:
            raise ValueError("均值窗口必须大于 0")
        self.settings = _WindowSettings(int(window))

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        return float(history.iloc[-self.settings.window :].mean())


class WindowMedianModel(HistoricalBaseline):
    """最近固定窗口中位数，对孤立异常点更稳健。"""

    def __init__(self, window: int = 8, interval_minutes: int = 15):
        super().__init__(interval_minutes)
        if window <= 0:
            raise ValueError("中位数窗口必须大于 0")
        self.settings = _WindowSettings(int(window))

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        return float(history.iloc[-self.settings.window :].median())


class LinearTrendModel(HistoricalBaseline):
    """最近窗口最小二乘线性趋势外推。"""

    def __init__(self, window: int = 8, interval_minutes: int = 15):
        super().__init__(interval_minutes)
        if window < 2:
            raise ValueError("趋势窗口至少为 2")
        self.settings = _WindowSettings(int(window))

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        values = history.iloc[-self.settings.window :].to_numpy(dtype=float)
        if len(values) < 2:
            return float(values[-1])
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values, deg=1)
        return float(intercept + slope * (len(values) - 1 + horizon))


class DampedTrendModel(HistoricalBaseline):
    """逐步衰减最近趋势，避免长周期线性外推发散。"""

    def __init__(
        self,
        window: int = 5,
        damping: float = 0.85,
        interval_minutes: int = 15,
    ):
        super().__init__(interval_minutes)
        if window < 2:
            raise ValueError("阻尼趋势窗口至少为 2")
        if not 0.0 <= damping <= 1.0:
            raise ValueError("阻尼系数必须位于 [0, 1]")
        self.settings = _WindowSettings(int(window))
        self.damping = float(damping)

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        values = history.iloc[-self.settings.window :].to_numpy(dtype=float)
        if len(values) < 2:
            return float(values[-1])
        slope = float(np.polyfit(np.arange(len(values), dtype=float), values, deg=1)[0])
        if self.damping == 1.0:
            multiplier = float(horizon)
        else:
            multiplier = (1.0 - self.damping**horizon) / (1.0 - self.damping)
        return float(values[-1] + slope * multiplier)


class SeasonalNaiveModel(HistoricalBaseline):
    """昨日或上周同时刻；历史点缺失时回退到最后有效值。"""

    def __init__(self, period_steps: int, interval_minutes: int = 15):
        super().__init__(interval_minutes)
        if period_steps <= 0:
            raise ValueError("季节周期必须大于 0")
        self.period_steps = int(period_steps)

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        offset = pd.Timedelta(minutes=self.interval_minutes)
        historical_time = origin + horizon * offset - self.period_steps * offset
        if historical_time <= origin and historical_time in history.index:
            value = history.loc[historical_time]
            if isinstance(value, pd.Series):
                value = value.iloc[-1]
            if pd.notna(value):
                return float(value)
        return float(history.iloc[-1])


class SeasonalMedianModel(HistoricalBaseline):
    """多个历史同刻的中位数，对单日异常和偶发启停更不敏感。"""

    def __init__(
        self,
        period_steps: int,
        seasons: int = 3,
        interval_minutes: int = 15,
    ):
        super().__init__(interval_minutes)
        if period_steps <= 0:
            raise ValueError("季节周期必须大于 0")
        if seasons <= 0:
            raise ValueError("季节样本数必须大于 0")
        self.period_steps = int(period_steps)
        self.seasons = int(seasons)

    def _predict_value(self, history: pd.Series, origin: pd.Timestamp, horizon: int) -> float:
        offset = pd.Timedelta(minutes=self.interval_minutes)
        target_time = origin + int(horizon) * offset
        values: list[float] = []
        for season in range(1, self.seasons + 1):
            historical_time = target_time - season * self.period_steps * offset
            if historical_time > origin or historical_time not in history.index:
                continue
            value = history.loc[historical_time]
            if isinstance(value, pd.Series):
                value = value.iloc[-1]
            if pd.notna(value):
                values.append(float(value))
        return float(np.median(values)) if values else float(history.iloc[-1])
