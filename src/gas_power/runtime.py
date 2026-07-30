"""统一管理并行度与进度显示配置。"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TypeVar

from gas_power.config import ProjectConfig


WORKERS_ENV = "GAS_POWER_WORKERS"
T = TypeVar("T")


def resolve_worker_count(config: ProjectConfig, override: int | None = None) -> int:
    """解析并校验工作线程数，且不超过当前机器的逻辑处理器数量。"""

    runtime = config.raw.get("runtime", {})
    configured = runtime.get("workers") if isinstance(runtime, dict) else None
    requested = override if override is not None else configured
    if requested is None:
        requested = os.environ.get(WORKERS_ENV, os.cpu_count() or 1)
    workers = int(requested)
    if workers <= 0:
        raise ValueError("工作线程数必须大于 0")
    return min(workers, os.cpu_count() or workers)


def configure_runtime(
    config: ProjectConfig,
    workers: int | None = None,
    show_progress: bool | None = None,
) -> tuple[int, bool]:
    """把命令行覆盖项写入本次运行配置，并同步原生计算库线程数。"""

    worker_count = resolve_worker_count(config, workers)
    runtime = config.raw.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime 配置必须是字典")
    progress_enabled = bool(runtime.get("progress", True))
    if show_progress is not None:
        progress_enabled = bool(show_progress)
    runtime["workers"] = worker_count
    runtime["progress"] = progress_enabled
    os.environ[WORKERS_ENV] = str(worker_count)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(worker_count)
    return worker_count, progress_enabled


def current_worker_count() -> int:
    """返回当前任务可使用的线程数。"""

    value = int(os.environ.get(WORKERS_ENV, os.cpu_count() or 1))
    return max(1, min(value, os.cpu_count() or value))


def progress_enabled(config: ProjectConfig) -> bool:
    runtime = config.raw.get("runtime", {})
    return bool(runtime.get("progress", True)) if isinstance(runtime, dict) else True


def track_progress(
    iterable: Iterable[T],
    *,
    config: ProjectConfig,
    description: str,
    total: int | None = None,
    unit: str = "项",
    leave: bool = True,
) -> Iterator[T]:
    """使用统一样式包装可迭代任务。"""

    from tqdm.auto import tqdm

    return iter(
        tqdm(
            iterable,
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            leave=leave,
            disable=not progress_enabled(config),
        )
    )
