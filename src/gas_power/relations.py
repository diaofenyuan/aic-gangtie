"""确定性关系发现；所有发现只生成报告，不自动加入正式特征白名单。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gas_power.availability import FeatureAvailabilityRegistry


@dataclass
class RelationArtifacts:
    summary: dict[str, Any]
    coefficients: pd.DataFrame
    setpoints: pd.DataFrame
    delays: pd.DataFrame
    markdown: str


def _fit_linear_relation(
    frame: pd.DataFrame,
    name: str,
    target: str,
    fields: Sequence[str],
    threshold: float,
) -> dict[str, Any] | None:
    columns = [*fields, target]
    if any(column not in frame for column in columns):
        return None
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < max(30, len(fields) * 5):
        return None
    x = np.column_stack([np.ones(len(data)), data[list(fields)].to_numpy(dtype=float)])
    y = data[target].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = x @ coefficients
    residual = y - prediction
    denominator = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / max(denominator, 1.0e-12)
    monthly: dict[str, float] = {}
    for month, positions in pd.Series(np.arange(len(data)), index=data.index).groupby(
        data.index.to_period("M").astype(str)
    ):
        pos = positions.to_numpy(dtype=int)
        if len(pos) < 10:
            continue
        actual = y[pos]
        error = residual[pos]
        month_denominator = float(np.sum(np.square(actual - actual.mean())))
        monthly[str(month)] = 1.0 - float(np.sum(np.square(error))) / max(month_denominator, 1.0e-12)
    worst_positions = np.argsort(np.abs(residual))[-min(5, len(residual)) :][::-1]
    return {
        "relation": name,
        "target": target,
        "fields": json.dumps(list(fields), ensure_ascii=False),
        "coefficients": json.dumps(
            {field: float(coefficients[i + 1]) for i, field in enumerate(fields)},
            ensure_ascii=False,
        ),
        "intercept": float(coefficients[0]),
        "r2": r2,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_p50_abs": float(np.quantile(np.abs(residual), 0.50)),
        "residual_p95_abs": float(np.quantile(np.abs(residual), 0.95)),
        "stable_months": json.dumps(
            [month for month, value in monthly.items() if value >= threshold], ensure_ascii=False
        ),
        "monthly_r2": json.dumps(monthly, ensure_ascii=False),
        "failure_conditions": json.dumps(
            [
                {
                    "datetime": str(data.index[int(position)]),
                    "residual": float(residual[int(position)]),
                }
                for position in worst_positions
            ],
            ensure_ascii=False,
        ),
        "candidate": bool(r2 >= threshold),
        "auto_whitelisted": False,
    }


def _segment_statistics(series: pd.Series, tolerance: float) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    changed = values.diff().abs().fillna(0.0) > tolerance
    run_id = changed.cumsum()
    run_lengths = values.groupby(run_id).size()
    return {
        "stable_fraction": float((~changed).mean()),
        "change_points": int(changed.sum()),
        "longest_constant_run": int(run_lengths.max()) if len(run_lengths) else 0,
        "median_constant_run": float(run_lengths.median()) if len(run_lengths) else 0.0,
    }


def _delay_relations(
    frame: pd.DataFrame,
    registry: FeatureAvailabilityRegistry,
    max_delay_steps: int,
) -> pd.DataFrame:
    fields = [
        str(column)
        for column in frame.select_dtypes(include=[np.number]).columns
        if not str(column).startswith("feat_") and registry.get(str(column)).source_file
    ]
    rows: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(fields):
        for right_name in fields[left_index + 1 :]:
            left_source = registry.get(left_name).source_file
            right_source = registry.get(right_name).source_file
            if not left_source or left_source == right_source:
                continue
            best: tuple[float, int, float] | None = None
            for offset in range(-max_delay_steps, max_delay_steps + 1):
                paired = pd.concat(
                    [frame[left_name], frame[right_name].shift(-offset)], axis=1
                ).dropna()
                if len(paired) < 20 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
                    continue
                correlation = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
                score = abs(correlation)
                if best is None or score > best[0]:
                    best = (score, offset, correlation)
            if best is not None and best[0] >= 0.90:
                rows.append(
                    {
                        "left_field": left_name,
                        "left_source": left_source,
                        "right_field": right_name,
                        "right_source": right_source,
                        "right_target_offset_steps": best[1],
                        "correlation": best[2],
                        "risk": "red" if best[1] > 0 else "review",
                    }
                )
    columns = [
        "left_field", "left_source", "right_field", "right_source",
        "right_target_offset_steps", "correlation", "risk",
    ]
    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output
    return output.sort_values(
        "correlation", key=lambda value: value.abs(), ascending=False
    )


def discover_relations(
    frame: pd.DataFrame,
    roles: Mapping[str, Any],
    registry: FeatureAvailabilityRegistry,
    config: Mapping[str, Any],
) -> RelationArtifacts:
    targets = [str(value) for value in roles.get("targets", [])]
    threshold = float(config.get("linear_r2_threshold", 0.98))
    coefficient_rows: list[dict[str, Any]] = []

    components = [str(value) for value in roles.get("component_generators", [])]
    relation = _fit_linear_relation(
        frame, "generator_all_component_sum", "generator_all", components, threshold
    )
    if relation is not None:
        coefficient_rows.append(relation)

    gas_use_mapping = roles.get("generator_gas_use", {})
    gas_fields = [str(value) for value in gas_use_mapping.values()] if isinstance(gas_use_mapping, Mapping) else []
    relation = _fit_linear_relation(
        frame, "generation_vs_three_gas_use", "generator_all", gas_fields, threshold
    )
    if relation is not None:
        coefficient_rows.append(relation)

    production = roles.get("gas_production", {})
    demand = roles.get("gas_user_demand", {})
    holders = roles.get("gas_holder", {})
    if all(isinstance(value, Mapping) for value in (production, demand, holders, gas_use_mapping)):
        for gas_type in sorted(set(production).intersection(demand, holders, gas_use_mapping)):
            holder_name = str(holders[gas_type])
            if holder_name not in frame:
                continue
            delta_name = f"__holder_delta_{gas_type}"
            work = frame.copy()
            work[delta_name] = pd.to_numeric(work[holder_name], errors="coerce").diff()
            relation = _fit_linear_relation(
                work,
                f"material_balance_{gas_type}",
                delta_name,
                [str(production[gas_type]), str(demand[gas_type]), str(gas_use_mapping[gas_type])],
                threshold,
            )
            if relation is not None:
                coefficient_rows.append(relation)

    tolerance = float(config.get("constant_tolerance_mw", 0.05))
    setpoint_round = float(config.get("setpoint_round_mw", 1.0))
    target_diagnostics: dict[str, Any] = {}
    setpoint_rows: list[dict[str, Any]] = []
    for target in targets:
        if target not in frame:
            continue
        target_diagnostics[target] = _segment_statistics(frame[target], tolerance)
        rounded = (pd.to_numeric(frame[target], errors="coerce") / setpoint_round).round() * setpoint_round
        counts = rounded.value_counts(dropna=True).head(10)
        for value, count in counts.items():
            setpoint_rows.append(
                {
                    "target": target,
                    "setpoint": float(value),
                    "sample_count": int(count),
                    "fraction": float(count / max(1, rounded.notna().sum())),
                }
            )

    delays = _delay_relations(frame, registry, int(config.get("max_delay_steps", 12)))
    plan_fields = [name for name, item in registry.fields.items() if item.is_plan and name in frame]
    summary = {
        "warning": "关系发现结果不会自动进入正式模型；必须先确认字段业务可用时间。",
        "linear_relations": len(coefficient_rows),
        "candidate_relations": sum(bool(row["candidate"]) for row in coefficient_rows),
        "target_diagnostics": target_diagnostics,
        "plan_fields": [
            {
                "field": field,
                "available_at_origin": registry.get(field).available_at_origin,
                "allow_short": registry.get(field).allow_short,
                "allow_long": registry.get(field).allow_long,
                "pending": registry.get(field).pending,
            }
            for field in plan_fields
        ],
        "cross_file_delay_candidates": len(delays),
    }
    markdown = "\n".join(
        [
            "# 确定性关系发现",
            "",
            "> 本报告只提供候选关系，不修改正式特征白名单。",
            "",
            f"- 线性关系：{len(coefficient_rows)}",
            f"- 达到配置阈值的候选关系：{summary['candidate_relations']}",
            f"- 跨文件固定延迟候选：{len(delays)}",
            f"- 计划字段：{', '.join(plan_fields) if plan_fields else '未识别'}",
            "",
            "真实数据到达后应重点检查关系在不同月份、启停和爬坡工况下是否失效。",
        ]
    ) + "\n"
    return RelationArtifacts(
        summary=summary,
        coefficients=pd.DataFrame(coefficient_rows),
        setpoints=pd.DataFrame(setpoint_rows),
        delays=delays,
        markdown=markdown,
    )
