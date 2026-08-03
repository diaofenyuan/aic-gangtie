"""训练验证与在线推理共用的因果工况判定。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REGIME_NAMES = ("stable", "ramp_up", "ramp_down", "startup", "shutdown")


def regime_labels_from_features(
    frame: pd.DataFrame,
    *,
    target_column: str = "target",
) -> pd.Series:
    """从滚动验证携带的参考时刻特征恢复工况标签。"""

    if target_column not in frame:
        raise ValueError(f"工况标签数据缺少字段: {target_column}")
    labels = pd.Series("stable", index=frame.index, dtype="object")
    for target in frame[target_column].dropna().astype(str).unique():
        target_mask = frame[target_column].astype(str) == target
        for suffix in ("ramp_up", "ramp_down", "startup", "shutdown"):
            column = f"feat_state_{target}_{suffix}"
            if column in frame:
                mask = target_mask & (frame[column].fillna(0.0) > 0.5)
                labels.loc[mask] = suffix
    return labels


def infer_regimes_from_history(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    target: str,
    *,
    online_threshold_mw: float = 1.0,
    stable_delta_threshold_mw: float = 2.0,
) -> pd.Series:
    """仅使用预测起点及之前的负荷，复现特征工程中的工况规则。"""

    if target not in frame:
        raise ValueError(f"工况推理缺少目标历史: {target}")
    if len(origins.difference(frame.index)):
        raise ValueError("工况推理起点不在历史时间轴中")

    load = pd.to_numeric(frame[target], errors="coerce")
    current = load.reindex(origins)
    previous = load.shift(1).reindex(origins)
    delta = current - previous
    online = current > float(online_threshold_mw)
    previously_online = previous > float(online_threshold_mw)

    labels = np.full(len(origins), "stable", dtype=object)
    labels[(delta > float(stable_delta_threshold_mw)).fillna(False).to_numpy()] = (
        "ramp_up"
    )
    labels[(delta < -float(stable_delta_threshold_mw)).fillna(False).to_numpy()] = (
        "ramp_down"
    )
    labels[(online & ~previously_online).fillna(False).to_numpy()] = "startup"
    labels[(~online & previously_online).fillna(False).to_numpy()] = "shutdown"
    return pd.Series(labels, index=origins, dtype="object")


def prediction_column_targets(
    targets: Sequence[str],
    horizons: Sequence[int],
    *,
    interval_minutes: int = 15,
) -> dict[str, str]:
    """返回官方预测列到目标字段的确定映射。"""

    return {
        f"{str(target)}_t+{int(horizon) * int(interval_minutes)}_pred": str(target)
        for target in targets
        for horizon in horizons
    }
