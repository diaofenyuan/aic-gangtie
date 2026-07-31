"""高精度训练依赖和 CUDA 环境检查。"""

from __future__ import annotations

import importlib
import platform
from typing import Any


def check_high_accuracy_environment() -> dict[str, Any]:
    dependencies: dict[str, dict[str, Any]] = {}
    for module_name in ("lightgbm", "catboost", "optuna", "torch"):
        try:
            module = importlib.import_module(module_name)
            dependencies[module_name] = {
                "installed": True,
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except ImportError as exc:
            dependencies[module_name] = {
                "installed": False,
                "error": str(exc),
            }
    cuda: dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    if dependencies["torch"]["installed"]:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "torch_cuda_version": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "cuda": cuda,
        "ready_for_tree_search": all(
            dependencies[name]["installed"] for name in ("lightgbm", "catboost", "optuna")
        ),
        "ready_for_deep_learning": bool(cuda["available"]),
    }
