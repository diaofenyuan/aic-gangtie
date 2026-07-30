"""深度时序模型实验门控；本模块不引入任何 GPU 依赖。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GPUGateDecision:
    allowed: bool
    reason: str
    min_fold_gain: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "min_fold_gain": self.min_fold_gain,
        }


def evaluate_gpu_gate(
    config: Mapping[str, Any],
    cpu_baselines_complete: bool,
    fold_gains: Sequence[float],
) -> GPUGateDecision:
    if not bool(config.get("enabled", False)):
        return GPUGateDecision(False, "GPU 实验配置未启用", None)
    if bool(config.get("require_cpu_baselines_complete", True)) and not cpu_baselines_complete:
        return GPUGateDecision(False, "CPU 基线和提升模型实验尚未完成", None)
    if not fold_gains:
        return GPUGateDecision(False, "没有多个严格时间折的增益证据", None)
    minimum = min(float(value) for value in fold_gains)
    threshold = float(config.get("min_gain_all_folds", 0.001))
    if minimum < threshold:
        return GPUGateDecision(False, f"最差折增益 {minimum:.6f} 低于阈值 {threshold:.6f}", minimum)
    return GPUGateDecision(True, "所有时间折均达到配置增益阈值", minimum)


def evaluate_residual_gate(
    config: Mapping[str, Any],
    fold_mape_gains: Sequence[float],
) -> dict[str, Any]:
    minimum = min((float(value) for value in fold_mape_gains), default=float("nan"))
    threshold = float(config.get("min_fold_mape_gain", 0.0001))
    require_all = bool(config.get("require_all_folds_improved", True))
    allowed = bool(fold_mape_gains) and (
        minimum >= threshold if require_all else sum(value >= threshold for value in fold_mape_gains) > len(fold_mape_gains) / 2
    )
    return {
        "allowed": allowed,
        "minimum_fold_mape_gain": minimum,
        "threshold": threshold,
        "require_all_folds_improved": require_all,
        "reason": "通过全部时间折门控" if allowed else "残差模型未在要求的时间折中稳定改善",
    }
