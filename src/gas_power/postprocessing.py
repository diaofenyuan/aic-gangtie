"""预测物理约束后处理及前后效果对比。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_power.metrics import summarize_predictions
from gas_power.models.base import prediction_column


@dataclass
class PostprocessResult:
    predictions: pd.DataFrame
    adjusted_cells: int
    adjustments: dict[str, int]


class PhysicalForecastPostprocessor:
    """实施非负、容量、爬坡及目标一致性约束。"""

    def __init__(self, config: Mapping[str, Any], interval_minutes: int = 15):
        self.config = config
        self.interval_minutes = int(interval_minutes)

    def apply(
        self,
        predictions: pd.DataFrame,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        targets: Sequence[str],
        horizons: Sequence[int],
    ) -> PostprocessResult:
        output = predictions.copy()
        before = output.copy()
        adjustments = {"non_negative": 0, "capacity": 0, "ramp": 0, "consistency": 0}
        if bool(self.config.get("non_negative", True)):
            negative = output < 0.0
            adjustments["non_negative"] = int(negative.to_numpy().sum())
            output = output.clip(lower=0.0)

        capacities = self.config.get("target_capacity_mw", {})
        if isinstance(capacities, Mapping):
            for target in targets:
                if target not in capacities:
                    continue
                capacity = float(capacities[target])
                columns = [prediction_column(str(target), int(h), self.interval_minutes) for h in horizons]
                mask = output[columns] > capacity
                adjustments["capacity"] += int(mask.to_numpy().sum())
                output.loc[:, columns] = output[columns].clip(upper=capacity)

        ramp_value = self.config.get("ramp_limit_mw_per_step")
        if ramp_value is not None:
            ramp = float(ramp_value)
            for origin in origins:
                for target in targets:
                    previous = float(frame.at[origin, str(target)])
                    for horizon in sorted(int(value) for value in horizons):
                        column = prediction_column(str(target), horizon, self.interval_minutes)
                        current = float(output.at[origin, column])
                        clipped = float(np.clip(current, previous - ramp, previous + ramp))
                        adjustments["ramp"] += int(not np.isclose(current, clipped))
                        output.at[origin, column] = clipped
                        previous = clipped

        if bool(self.config.get("enforce_target_consistency", True)) and {
            "generator_1", "generator_all"
        }.issubset(targets):
            for horizon in horizons:
                one = prediction_column("generator_1", int(horizon), self.interval_minutes)
                total = prediction_column("generator_all", int(horizon), self.interval_minutes)
                inconsistent = output[one] > output[total]
                adjustments["consistency"] += int(inconsistent.sum())
                output.loc[inconsistent, one] = output.loc[inconsistent, total]

        changed = ~np.isclose(
            before.to_numpy(dtype=float), output.to_numpy(dtype=float), equal_nan=True
        )
        return PostprocessResult(output, int(changed.sum()), adjustments)


def compare_postprocessing_metrics(
    raw_tidy: pd.DataFrame,
    processed_tidy: pd.DataFrame,
    near_zero_threshold: float,
) -> pd.DataFrame:
    """分别计算约束前后指标，防止默认假定约束一定改善效果。"""

    raw = summarize_predictions(raw_tidy, near_zero_threshold)
    raw.insert(0, "stage", "before")
    processed = summarize_predictions(processed_tidy, near_zero_threshold)
    processed.insert(0, "stage", "after")
    return pd.concat([raw, processed], ignore_index=True)
