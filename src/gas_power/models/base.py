"""短周期和长周期预测模型共用的预测接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import pandas as pd


FitProgressCallback = Callable[[str], None]


class OptionalDependencyError(ImportError):
    """用户选择了尚未安装的可选模型依赖。"""


def prediction_column(target: str, horizon_steps: int, interval_minutes: int = 15) -> str:
    return f"{target}_t+{horizon_steps * interval_minutes}_pred"


def prediction_columns(
    targets: Sequence[str], horizons: Sequence[int], interval_minutes: int = 15
) -> list[str]:
    return [
        prediction_column(str(target), int(horizon), interval_minutes)
        for target in targets
        for horizon in horizons
    ]


class ForecastModel(ABC):
    """统一多目标、多步预测协议。"""

    @abstractmethod
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
    ) -> "ForecastModel":
        """仅使用 train_end 及之前可获得的标签拟合。"""

    @abstractmethod
    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        """返回以预测起点为索引的官方宽表列。"""

    def fit_progress_steps(
        self,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> int | None:
        """返回一次拟合可观测的进度步数；不可细分时返回空值。"""

        return None

    def set_fit_progress_callback(
        self,
        callback: FitProgressCallback | None,
    ) -> None:
        """设置拟合进度回调；默认模型不提供细粒度进度。"""


def validate_prediction_request(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    targets: Sequence[str],
    horizons: Sequence[int],
) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("预测输入索引必须是 DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("预测输入索引必须严格递增且唯一")
    missing_targets = [str(target) for target in targets if str(target) not in frame]
    if missing_targets:
        raise ValueError(f"预测输入缺少目标历史: {missing_targets}")
    missing_origins = origins.difference(frame.index)
    if len(missing_origins):
        raise ValueError(f"预测起点不在历史时间轴中: {missing_origins[:3].tolist()}")
    if not horizons or any(int(horizon) <= 0 for horizon in horizons):
        raise ValueError("预测步长必须是正整数")
