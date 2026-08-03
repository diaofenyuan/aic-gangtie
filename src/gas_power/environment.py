"""高精度训练可选依赖检查。"""

from __future__ import annotations

import importlib
import platform
from typing import Any


def check_high_accuracy_environment() -> dict[str, Any]:
    dependencies: dict[str, dict[str, Any]] = {}
    for module_name in ("lightgbm", "catboost", "optuna"):
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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependencies,
        "ready_for_tree_search": all(
            dependencies[name]["installed"] for name in ("lightgbm", "catboost", "optuna")
        ),
    }
