"""传统预测模型的时间折稳定性门控。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def evaluate_residual_gate(
    config: Mapping[str, Any],
    fold_mape_gains: Sequence[float],
) -> dict[str, Any]:
    minimum = min((float(value) for value in fold_mape_gains), default=float("nan"))
    threshold = float(config.get("min_fold_mape_gain", 0.0001))
    require_all = bool(config.get("require_all_folds_improved", True))
    allowed = bool(fold_mape_gains) and (
        minimum >= threshold
        if require_all
        else sum(value >= threshold for value in fold_mape_gains)
        > len(fold_mape_gains) / 2
    )
    return {
        "allowed": allowed,
        "minimum_fold_mape_gain": minimum,
        "threshold": threshold,
        "require_all_folds_improved": require_all,
        "reason": "通过全部时间折门控" if allowed else "残差模型未在要求的时间折中稳定改善",
    }
