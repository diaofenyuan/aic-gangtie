"""项目配置读取、校验和路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """配置内容缺失或不合法。"""


@dataclass(frozen=True)
class ProjectConfig:
    """保留原始 YAML，同时统一解析所有工程目录。"""

    raw: dict[str, Any]
    source: Path
    root: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"配置缺少字典段: {name}")
        return value

    def path(self, name: str, default: str | None = None) -> Path:
        paths = self.section("paths")
        value = paths.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"paths.{name} 必须是非空字符串")
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def seed(self) -> int:
        return int(self.section("project").get("seed", 2026))

    def ensure_runtime_dirs(self) -> None:
        # 运行结果目录由 OutputSession 在单次运行目录中动态注入；默认配置不创建根级结果目录。
        for name in ("data", "cache", "logs", "models", "outputs", "results", "reports"):
            if name not in self.section("paths"):
                continue
            self.path(name).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> ProjectConfig:
    """读取 YAML，并以配置文件目录为基准解析项目根目录。"""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ConfigError(f"配置文件不存在: {source}")
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("配置文件顶层必须是字典")

    paths = raw.get("paths")
    if not isinstance(paths, Mapping):
        raise ConfigError("配置缺少 paths 段")
    root_value = paths.get("root", ".")
    if not isinstance(root_value, str):
        raise ConfigError("paths.root 必须是字符串")
    root_path = Path(root_value)
    root = root_path.resolve() if root_path.is_absolute() else (source.parent / root_path).resolve()

    config = ProjectConfig(raw=dict(raw), source=source, root=root)
    _validate_config(config)
    return config


def _validate_config(config: ProjectConfig) -> None:
    data = config.section("data")
    if not isinstance(data.get("tables"), dict) or not data["tables"]:
        raise ConfigError("data.tables 至少需要配置一张表")
    roles = data.get("roles")
    if not isinstance(roles, dict):
        raise ConfigError("data.roles 必须配置字段角色")
    targets = roles.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ConfigError("data.roles.targets 至少需要一个预测目标")

    forecast = config.section("forecast")
    short_steps = int(forecast.get("short_steps", 0))
    long_steps = int(forecast.get("long_steps", 0))
    if short_steps != 8 or long_steps != 96:
        raise ConfigError("当前赛题要求 short_steps=8 且 long_steps=96")

    optimization = config.section("optimization")
    units = optimization.get("units")
    if not isinstance(units, list) or len(units) != 6:
        raise ConfigError("优化配置必须包含 4 套 50MW 和 2 套 120MW 机组")
    rated_capacities = sorted(float(unit.get("rated_mw", 0.0)) for unit in units)
    if rated_capacities != [50.0, 50.0, 50.0, 50.0, 120.0, 120.0]:
        raise ConfigError("优化机组额定容量必须为 4×50MW 和 2×120MW")
    generator_1_units = [unit for unit in units if unit.get("group") == "generator_1"]
    if len(generator_1_units) != 4 or any(
        float(unit.get("rated_mw", 0.0)) != 50.0 for unit in generator_1_units
    ):
        raise ConfigError("generator_1 必须映射为四套 50MW 机组的合计")

    priority_mode = str(optimization.get("priority_mode", "lexicographic"))
    if priority_mode not in {"lexicographic", "weighted"}:
        raise ConfigError("optimization.priority_mode 只能是 lexicographic 或 weighted")

    submission = config.raw.get("submission", {})
    if not isinstance(submission, Mapping):
        raise ConfigError("submission 必须是字典")


def configured_value(item: Any, path: str) -> float:
    """读取带状态说明的占位参数，避免代码中散落未经确认的常数。"""

    if not isinstance(item, Mapping) or "value" not in item or "status" not in item:
        raise ConfigError(f"{path} 必须同时包含 value 和 status")
    return float(item["value"])
