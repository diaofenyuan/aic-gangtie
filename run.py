"""支持从源码目录直接启动项目命令行，无需预先安装项目包。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
# 项目采用 src 目录布局；直接执行本文件时需显式加入模块搜索路径。
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# IDE 无参数直接运行使用略低于正式配置的搜索预算，便于复用已有调参结果。
# 显式传入命令行参数或环境变量时仍按调用方指定的正式预算执行。
if len(sys.argv) == 1:
    os.environ.setdefault("GAS_POWER_TUNE_TRIALS", "10")
    os.environ.setdefault("GAS_POWER_TUNE_TOP_K", "4")

# 原生计算库通常在导入时读取线程配置，因此必须在导入项目入口前完成初始化。
# 调用方可通过 GAS_POWER_WORKERS 指定并行度；未指定时最多使用 16 个逻辑处理器。
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
os.environ.setdefault("GAS_POWER_WORKERS", str(DEFAULT_WORKERS))
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = os.environ["GAS_POWER_WORKERS"]

# 路径与运行时环境已完成初始化，此处延迟导入属于入口脚本的预期行为。
from gas_power.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
