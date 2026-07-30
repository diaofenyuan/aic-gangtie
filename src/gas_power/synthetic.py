"""只用于工程连通性验证的 15 分钟合成数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gas_power.config import ProjectConfig


SYNTHETIC_WARNING = (
    "本数据由规则和随机数合成，只能验证读取、特征、预测、验证和优化代码流程；"
    "不得用于判断真实数据准确率、模型优劣或竞赛成绩。"
)


def _simulate_unit(
    index: pd.DatetimeIndex,
    rated_mw: float,
    phase: float,
    outage_period_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    steps_per_day = 96
    x = np.arange(len(index), dtype=float)
    daily = np.sin(2.0 * np.pi * x / steps_per_day + phase)
    weekly = np.sin(2.0 * np.pi * x / (7 * steps_per_day) + phase / 2.0)
    desired_fraction = np.clip(0.80 + 0.12 * daily + 0.04 * weekly, 0.60, 1.0)

    day_number = (x // steps_per_day).astype(int)
    step_in_day = (x % steps_per_day).astype(int)
    outage = (day_number % outage_period_days == outage_period_days - 1) & (
        (step_in_day >= 8 + int(phase * 3) % 16)
        & (step_in_day < 20 + int(phase * 3) % 16)
    )
    desired = np.where(outage, 0.0, rated_mw * desired_fraction)
    desired += rng.normal(0.0, rated_mw * 0.008, size=len(index))
    desired = np.clip(desired, 0.0, rated_mw)

    # 使用多个 15 分钟点完成启停过渡，以便特征模块观察稳定、停机和爬坡工况。
    load = np.zeros(len(index), dtype=float)
    max_demo_change = rated_mw * 0.12
    for position in range(1, len(index)):
        difference = desired[position] - load[position - 1]
        load[position] = load[position - 1] + np.clip(
            difference, -max_demo_change, max_demo_change
        )
    return np.clip(load, 0.0, rated_mw)


def _fallback_price(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour
    peak = ((hour >= 8) & (hour < 11)) | ((hour >= 17) & (hour < 22))
    valley = (hour < 7) | (hour >= 23)
    return np.where(peak, 1.0, np.where(valley, 0.35, 0.65)).astype(float)


def _inject_defects(
    clean: pd.DataFrame,
    settings: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = clean.copy()
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    missing_rows = int(len(frame) * float(settings.get("missing_row_fraction", 0.004)))
    if missing_rows > 0:
        candidates = np.arange(1, len(frame) - 1)
        dropped = rng.choice(candidates, size=min(missing_rows, len(candidates)), replace=False)
        frame = frame.drop(frame.index[dropped])

    missing_values = int(
        len(frame)
        * max(1, len(numeric_columns))
        * float(settings.get("missing_value_fraction", 0.002))
    )
    for _ in range(missing_values):
        row = int(rng.integers(0, len(frame)))
        column = str(rng.choice(numeric_columns))
        frame.iat[row, frame.columns.get_loc(column)] = np.nan

    outlier_count = int(settings.get("outliers_per_table", 3))
    safe_start = min(96, max(0, len(frame) // 4))
    safe_end = max(safe_start + 1, len(frame) - 96)
    for _ in range(outlier_count):
        row = int(rng.integers(safe_start, safe_end))
        column = str(rng.choice(numeric_columns))
        value = frame.iat[row, frame.columns.get_loc(column)]
        if pd.notna(value):
            frame.iat[row, frame.columns.get_loc(column)] = float(value) * 4.5

    duplicate_count = int(settings.get("duplicate_rows_per_table", 4))
    duplicate_count = min(duplicate_count, len(frame))
    duplicate_positions = rng.choice(len(frame), size=duplicate_count, replace=False)
    duplicates = frame.iloc[duplicate_positions].copy()
    for column in numeric_columns:
        duplicates[column] = duplicates[column] * (
            1.0 + rng.normal(0.0, 0.001, size=len(duplicates))
        )
    frame = pd.concat([frame, duplicates], axis=0)
    frame = frame.iloc[rng.permutation(len(frame))]
    return frame, {
        "missing_rows": missing_rows,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_count,
        "outlier_cells": outlier_count,
    }


def _write_raw_table(frame: pd.DataFrame, path: Path, timestamp_format: str) -> None:
    output = frame.copy()
    output.insert(0, "datetime", output.index.strftime(timestamp_format))
    output.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")


def generate_synthetic_scenarios(
    index: pd.DatetimeIndex,
    generator_all: np.ndarray | None = None,
) -> pd.DataFrame:
    """生成用于基线行为、时间错位和泄漏审计的独立场景。"""

    x = np.arange(len(index), dtype=float)
    total = (
        np.asarray(generator_all, dtype=float)
        if generator_all is not None
        else 180.0 + 20.0 * np.sin(2.0 * np.pi * x / 96.0)
    )
    piece_values = np.asarray([80.0, 120.0, 0.0, 160.0])
    piecewise = piece_values[((x // 96).astype(int)) % len(piece_values)]
    startup = np.where((x.astype(int) % 192) < 128, 110.0, 0.0)
    total_series = pd.Series(total, index=index)
    return pd.DataFrame(
        {
            "scenario_constant_load": np.full(len(index), 100.0),
            "scenario_piecewise_load": piecewise,
            "scenario_periodic_load": 100.0 + 20.0 * np.sin(2.0 * np.pi * x / 96.0),
            "scenario_linear_ramp_load": 20.0 + 0.25 * x,
            "scenario_start_stop_load": startup,
            # 合法计划量和故意泄漏字段均只用于审计，正式白名单默认关闭。
            "legal_dispatch_plan": total_series.shift(-4).ffill().to_numpy(),
            "future_generator_all_leak": total_series.shift(-1).ffill().to_numpy(),
            "timestamp_advanced_generator_all": total_series.shift(-2).ffill().to_numpy(),
            "delayed_generator_all_sensor": total_series.shift(2).bfill().to_numpy(),
            "boundary_signal": (x % 8).astype(float),
        },
        index=index,
    )


def generate_synthetic_dataset(config: ProjectConfig) -> dict[str, Any]:
    """生成含少量缺失、重复、异常点的多表数据并返回清单。"""

    settings = config.section("synthetic")
    frequency = str(config.section("data").get("frequency", "15min"))
    timestamp_format = str(
        config.section("data").get("timestamp_format", "%Y-%m-%d %H:%M:%S")
    )
    index = pd.date_range(
        str(settings.get("start", "2025-01-01 00:00:00")),
        periods=int(settings.get("periods", 2016)),
        freq=frequency,
    )
    rng = np.random.default_rng(config.seed)
    units = [
        _simulate_unit(index, 50.0, 0.2 + unit * 0.35, 9 + unit, rng)
        for unit in range(4)
    ] + [
        _simulate_unit(index, 120.0, 0.5 + unit * 0.7, 11 + unit * 2, rng)
        for unit in range(2)
    ]
    synthetic_x = np.arange(len(index), dtype=int)
    group_outage = (
        ((synthetic_x // 96) % 14 == 12)
        & ((synthetic_x % 96) >= 20)
        # 停机持续足够长，避免被 IQR 异常过滤器误判为孤立异常点，
        # 同时让回测能够覆盖停机和恢复启机两个边界。
        & ((synthetic_x % 96) < 84)
    )
    for unit in range(4):
        units[unit][group_outage] = 0.025
    generator_1 = np.sum(units[:4], axis=0)
    generator_all = np.sum(units, axis=0)

    x = np.arange(len(index), dtype=float)
    daily = np.sin(2.0 * np.pi * x / 96.0)
    weekly = np.sin(2.0 * np.pi * x / 672.0)
    blast_user = 11_000.0 + 900.0 * daily + 350.0 * weekly + rng.normal(0, 120, len(index))
    coke_user = 5_200.0 + 400.0 * np.sin(2.0 * np.pi * x / 96.0 + 0.8) + rng.normal(0, 70, len(index))
    converter_user = 3_300.0 + 350.0 * np.sin(2.0 * np.pi * x / 96.0 - 0.5) + rng.normal(0, 60, len(index))

    blast_use = generator_all * 0.50 / 0.012
    coke_use = generator_all * 0.30 / 0.025
    converter_use = generator_all * 0.20 / 0.017

    holders = {
        "blast": 105_000.0 + 24_000.0 * np.sin(2.0 * np.pi * x / 96.0 - 0.7) + 4_000.0 * weekly,
        "coke": 95_000.0 + 18_000.0 * np.sin(2.0 * np.pi * x / 96.0 + 0.4) + 3_000.0 * weekly,
        "converter": 90_000.0 + 16_000.0 * np.sin(2.0 * np.pi * x / 96.0 + 1.1) + 2_500.0 * weekly,
    }
    for name in holders:
        holders[name] = holders[name] + rng.normal(0.0, 250.0, len(index))

    holder_changes = {
        name: np.diff(values, prepend=values[0]) for name, values in holders.items()
    }
    blast_production = blast_user + blast_use + holder_changes["blast"]
    coke_production = coke_user + coke_use + holder_changes["coke"]
    converter_production = converter_user + converter_use + holder_changes["converter"]

    clean_tables = {
        "gas": pd.DataFrame(
            {
                "blast_furnace_1": np.clip(blast_production, 0.0, None),
                "coke_oven_1": np.clip(coke_production, 0.0, None),
                "converter_1": np.clip(converter_production, 0.0, None),
            },
            index=index,
        ),
        "gas_holder": pd.DataFrame(
            {
                "blast_furnace_gas_holder_1": holders["blast"],
                "coke_oven_gas_holder_1": holders["coke"],
                "converter_gas_holder_1": holders["converter"],
            },
            index=index,
        ),
        "gas_user": pd.DataFrame(
            {
                "blast_furnace_user1": blast_user,
                "coke_oven_user1": coke_user,
                "converter_user1": converter_user,
            },
            index=index,
        ),
        "load": pd.DataFrame(
            {
                "generator_1": generator_1,
                "generator_all": generator_all,
                "generator_2": units[4],
                "generator_3": units[5],
                "generator_use_blast_furnace_gas": blast_use,
                "generator_use_coke_gas": coke_use,
                "generator_use_converter_gas": converter_use,
            },
            index=index,
        ),
        "price": pd.DataFrame(
            {"electricity_price": _fallback_price(index)}, index=index
        ),
        "audit_scenarios": generate_synthetic_scenarios(index, generator_all),
    }

    data_dir = config.path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "warning": SYNTHETIC_WARNING,
        "seed": config.seed,
        "frequency": frequency,
        "clean_periods": len(index),
        "tables": {},
    }
    table_configs = config.section("data")["tables"]
    for table_name, clean in clean_tables.items():
        if table_name == "audit_scenarios":
            defective = clean
            defect_counts = {
                "missing_rows": 0,
                "missing_values": 0,
                "duplicate_rows": 0,
                "outlier_cells": 0,
            }
        else:
            defective, defect_counts = _inject_defects(clean, settings, rng)
            if table_name == "load":
                start = min(len(clean) - 6, max(1, int(len(clean) * 0.60)))
                high_missing_index = clean.index[start : start + 6]
                columns = ["generator_1", "generator_all"]
                defective.loc[defective.index.isin(high_missing_index), columns] = np.nan
                defect_counts["high_missing_block_cells"] = len(high_missing_index) * len(columns)
        filename = str(table_configs[table_name]["path"])
        _write_raw_table(defective, data_dir / filename, timestamp_format)
        manifest["tables"][table_name] = {
            "path": filename,
            "written_rows": len(defective),
            **defect_counts,
        }
    manifest_path = data_dir / "synthetic_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
