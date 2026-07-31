"""项目配置读取、校验和路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    raw = _load_raw_config(source, ())

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


def _load_raw_config(source: Path, chain: Sequence[Path]) -> dict[str, Any]:
    """递归加载基础配置，子配置中的字典按键覆盖、列表整体替换。"""

    if source in chain:
        cycle = " -> ".join(str(path) for path in (*chain, source))
        raise ConfigError(f"配置继承存在循环: {cycle}")
    if not source.exists():
        raise ConfigError(f"配置文件不存在: {source}")
    with source.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ConfigError(f"配置文件顶层必须是字典: {source}")

    extends = loaded.pop("extends", None)
    if extends is None:
        return dict(loaded)
    if not isinstance(extends, str) or not extends.strip():
        raise ConfigError("extends 必须是非空字符串")
    base_path = Path(extends)
    base_source = (
        base_path.resolve()
        if base_path.is_absolute()
        else (source.parent / base_path).resolve()
    )
    base = _load_raw_config(base_source, (*chain, source))
    return _deep_merge(base, loaded)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """深度合并配置字典，保证正式配置只声明与默认值不同的部分。"""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _deep_merge(base_value, value)
        else:
            merged[str(key)] = value
    return merged


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

    _validate_competition_compliance(config)


def _validate_competition_compliance(config: ProjectConfig) -> None:
    """对正式赛事配置执行离线、数据边界和外部权重门禁。"""

    compliance = config.raw.get("competition_compliance")
    if compliance is None:
        return
    if not isinstance(compliance, Mapping):
        raise ConfigError("competition_compliance 必须是字典")
    if compliance.get("stage") != "preliminary":
        raise ConfigError("当前正式配置只允许 competition_compliance.stage=preliminary")
    if compliance.get("offline_only") is not True:
        raise ConfigError("正式赛事配置必须启用 offline_only")
    if compliance.get("official_data_only") is not True:
        raise ConfigError("正式赛事配置必须启用 official_data_only")
    if compliance.get("allow_external_pretrained_weights") is not False:
        raise ConfigError("正式赛事配置必须禁止外部预训练权重")

    allowed_commands = compliance.get("allowed_commands", [])
    if not isinstance(allowed_commands, list) or not allowed_commands or not all(
        isinstance(value, str) and value for value in allowed_commands
    ):
        raise ConfigError("competition_compliance.allowed_commands 必须是非空字符串列表")

    restricted_keys = compliance.get("restricted_path_keys", [])
    if not isinstance(restricted_keys, list) or not all(
        isinstance(value, str) and value for value in restricted_keys
    ):
        raise ConfigError("competition_compliance.restricted_path_keys 必须是字符串列表")
    training_path = config.path("data")
    for path_key in restricted_keys:
        restricted_path = config.path(path_key)
        if _paths_overlap(training_path, restricted_path):
            raise ConfigError(
                f"训练数据目录 paths.data 不得与受限目录 paths.{path_key} 重叠"
            )

    # 所有训练表必须是训练目录内的相对路径，防止单表绕过目录隔离读取评分集。
    for table_name, table_config in config.section("data")["tables"].items():
        if not isinstance(table_config, Mapping):
            raise ConfigError(f"data.tables.{table_name} 必须是字典")
        configured_path = Path(str(table_config.get("path", "")))
        if (
            not str(configured_path)
            or configured_path.is_absolute()
            or ".." in configured_path.parts
        ):
            raise ConfigError(
                f"data.tables.{table_name}.path 必须是 paths.data 下的相对路径"
            )

    origins = config.section("forecast").get("prediction_origins", {})
    if isinstance(origins, Mapping) and origins.get("mode") == "scoring":
        config.path("scoring_data")
        prediction_input = config.raw.get("prediction_input")
        if not isinstance(prediction_input, Mapping):
            raise ConfigError("scoring 模式必须配置 prediction_input")
        table_paths = prediction_input.get("table_paths")
        if not isinstance(table_paths, Mapping) or not table_paths:
            raise ConfigError("prediction_input.table_paths 必须是非空字典")
        for table_name, table_path in table_paths.items():
            if table_name not in config.section("data")["tables"]:
                raise ConfigError(f"prediction_input.table_paths 包含未知表: {table_name}")
            relative_path = Path(str(table_path))
            if (
                not str(relative_path)
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise ConfigError(
                    f"prediction_input.table_paths.{table_name} 必须是评分目录下的相对路径"
                )


def _paths_overlap(first: Path, second: Path) -> bool:
    """判断两个目录是否相同或存在父子包含关系。"""

    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def validate_competition_command(config: ProjectConfig, command: str) -> None:
    """正式配置只允许执行当前赛段明确开放的命令。"""

    compliance = config.raw.get("competition_compliance")
    if not isinstance(compliance, Mapping):
        return
    allowed = {str(value) for value in compliance.get("allowed_commands", [])}
    if command not in allowed:
        raise ConfigError(
            f"初赛正式配置禁止执行命令 {command}；允许命令为 {sorted(allowed)}"
        )


def configured_value(item: Any, path: str) -> float:
    """读取带状态说明的占位参数，避免代码中散落未经确认的常数。"""

    if not isinstance(item, Mapping) or "value" not in item or "status" not in item:
        raise ConfigError(f"{path} 必须同时包含 value 和 status")
    return float(item["value"])
