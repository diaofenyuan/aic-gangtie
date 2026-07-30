"""字段业务可用时间注册表与保守白名单。

字段的业务事件时间和采集时间在正式数据到达后必须由业务方确认。
未知字段默认禁止进入正式模型；本模块只负责审计和门控，不会自动放宽白名单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from gas_power.config import ProjectConfig


@dataclass(frozen=True)
class FieldAvailability:
    """单个字段的时间语义。"""

    name: str
    source_file: str = ""
    timestamp_meaning: str = "待确认"
    collection_time: str = "待确认"
    event_time: str = "待确认"
    available_at_origin: bool = False
    min_lag_steps: int = 1
    is_label: bool = False
    is_plan: bool = False
    allow_short: bool = False
    allow_long: bool = False
    pending: str = "字段未确认，默认禁止进入正式模型"
    collection_delay_steps: int = 0
    known_ahead_steps: int = 0

    def allowed(self, scope: str) -> bool:
        return bool(
            self.available_at_origin
            and (self.allow_short if scope == "short" else self.allow_long)
        )


@dataclass(frozen=True)
class FeatureUsage:
    """某个特征在预测起点使用的源字段和相对时间。"""

    feature_name: str
    source_field: str
    source_offset_steps: int = 0
    scope: str = "long"


@dataclass
class AvailabilityIssue:
    feature_name: str
    source_field: str
    message: str
    risk: str = "red"


@dataclass
class FeatureAvailabilityRegistry:
    """字段注册表；没有注册记录的字段一律按未知字段处理。"""

    fields: dict[str, FieldAvailability] = field(default_factory=dict)
    path: Path | None = None
    missing_file: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "FeatureAvailabilityRegistry":
        if not path.exists():
            return cls(path=path, missing_file=True)
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        entries = raw.get("fields", raw) if isinstance(raw, Mapping) else {}
        fields: dict[str, FieldAvailability] = {}
        if isinstance(entries, Mapping):
            for name, value in entries.items():
                if not isinstance(value, Mapping):
                    continue
                fields[str(name)] = FieldAvailability(
                    name=str(name),
                    source_file=str(value.get("source_file", "")),
                    timestamp_meaning=str(value.get("timestamp_meaning", "待确认")),
                    collection_time=str(value.get("collection_time", "待确认")),
                    event_time=str(value.get("event_time", "待确认")),
                    available_at_origin=bool(value.get("available_at_origin", False)),
                    min_lag_steps=int(value.get("min_lag_steps", 1)),
                    is_label=bool(value.get("is_label", False)),
                    is_plan=bool(value.get("is_plan", False)),
                    allow_short=bool(value.get("allow_short", False)),
                    allow_long=bool(value.get("allow_long", False)),
                    pending=str(value.get("pending", "待确认")),
                    collection_delay_steps=int(value.get("collection_delay_steps", 0)),
                    known_ahead_steps=int(value.get("known_ahead_steps", 0)),
                )
        return cls(fields=fields, path=path, missing_file=False)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "FeatureAvailabilityRegistry":
        section = config.raw.get("time_semantics", {})
        configured = section.get("availability_file", "config/feature_availability.yaml")
        configured_path = Path(str(configured))
        if configured_path.is_absolute():
            path = configured_path
        else:
            # 默认路径相对工程根；测试配置若没有复制注册表则使用保守空表。
            path = (config.root / configured_path).resolve()
            if not path.exists():
                bundled = (Path(__file__).resolve().parents[2] / configured_path).resolve()
                if bundled.exists():
                    path = bundled
        return cls.from_yaml(path)

    def get(self, field_name: str) -> FieldAvailability:
        return self.fields.get(str(field_name), FieldAvailability(name=str(field_name)))

    def allowed_source_columns(self, scope: str) -> set[str]:
        return {name for name, item in self.fields.items() if item.allowed(scope)}

    def validate_usage(
        self,
        usage: FeatureUsage,
        origin: Any,
        interval_minutes: int,
    ) -> AvailabilityIssue | None:
        item = self.get(usage.source_field)
        if usage.scope not in {"short", "long"}:
            return AvailabilityIssue(usage.feature_name, usage.source_field, "scope 必须为 short 或 long")
        if not item.allowed(usage.scope):
            return AvailabilityIssue(
                usage.feature_name,
                usage.source_field,
                f"字段未通过 {usage.scope} 白名单：available_at_origin={item.available_at_origin}, "
                f"allow_{usage.scope}={getattr(item, f'allow_{usage.scope}')}; {item.pending}",
            )
        if usage.source_offset_steps > -item.min_lag_steps:
            return AvailabilityIssue(
                usage.feature_name,
                usage.source_field,
                f"源字段最小滞后为 {item.min_lag_steps} 步，实际使用 offset={usage.source_offset_steps}",
            )
        # 事件时刻可在起点之后，但只有明确提前发布的计划量允许这样使用。
        effective_delay = item.collection_delay_steps - item.known_ahead_steps
        if usage.source_offset_steps + effective_delay > 0:
            return AvailabilityIssue(
                usage.feature_name,
                usage.source_field,
                "feature_available_time 晚于 forecast_origin_time，存在未来信息风险",
            )
        return None

    def audit_usages(
        self,
        usages: Iterable[FeatureUsage],
        origin: Any,
        interval_minutes: int,
    ) -> list[AvailabilityIssue]:
        return [
            issue
            for usage in usages
            if (issue := self.validate_usage(usage, origin, interval_minutes)) is not None
        ]


def registry_path(config: ProjectConfig) -> Path:
    """返回配置中声明的字段语义文件路径。"""

    return FeatureAvailabilityRegistry.from_config(config).path or (
        config.root / "config/feature_availability.yaml"
    )
