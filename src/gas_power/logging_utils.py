"""统一日志与随机种子。"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_logging(log_dir: Path, level: str = "INFO", filename: str = "pipeline.log") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gas_power")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
