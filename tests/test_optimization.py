from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from gas_power.config import load_config
from gas_power.optimization import (
    DispatchInput,
    HighsDispatchOptimizer,
    OptimizationError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_highs_dispatch_satisfies_material_and_unit_constraints() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    optimization = config.section("optimization")
    periods = 8
    timestamps = pd.date_range("2025-02-01", periods=periods, freq="15min")
    gas_types = list(optimization["gas_types"])
    dispatch_input = DispatchInput(
        timestamps=timestamps,
        production={gas: np.full(periods, 15_000.0) for gas in gas_types},
        user_demand={gas: np.full(periods, 8_000.0) for gas in gas_types},
        initial_storage={gas: 100_000.0 for gas in gas_types},
        electricity_price=np.array([0.35, 0.35, 0.65, 0.65, 1.0, 1.0, 0.65, 0.35]),
        baseline_generation_mw=np.full(periods, 250.0),
        baseline_flare_volume=np.zeros(periods),
    )
    result = HighsDispatchOptimizer(optimization).solve(dispatch_input)
    assert result.diagnostics.success
    assert result.diagnostics.max_constraint_violation is not None
    assert result.diagnostics.max_constraint_violation <= 1.0e-5
    assert result.diagnostics.total_shortage == 0.0
    assert [stage["name"] for stage in result.diagnostics.stage_results] == [
        "minimize_shortage",
        "minimize_flare",
        "maximize_economic_benefit",
    ]
    assert result.diagnostics.price_weighted_load_ratio is not None
    assert result.diagnostics.relative_benefit_improvement is not None
    assert len(result.gas_plan) == periods
    assert all(column.startswith("opt_") for column in result.gas_plan.columns)
    for gas_type in gas_types:
        storage = result.storage_plan[f"storage_{gas_type}"]
        assert storage.min() >= 30_000.0 - 1.0e-5
        assert storage.max() <= 180_000.0 + 1.0e-5


def test_lexicographic_dispatch_rejects_plan_with_user_shortage() -> None:
    project = load_config(PROJECT_ROOT / "config" / "default.yaml")
    optimization = deepcopy(project.section("optimization"))
    optimization["horizon_steps"] = 1
    gas_types = list(optimization["gas_types"])
    dispatch_input = DispatchInput(
        timestamps=pd.date_range("2025-02-01", periods=1, freq="15min"),
        production={gas: np.zeros(1) for gas in gas_types},
        user_demand={gas: np.full(1, 100_000.0) for gas in gas_types},
        initial_storage={gas: 30_000.0 for gas in gas_types},
        electricity_price=np.ones(1),
    )

    with pytest.raises(OptimizationError, match="零供气不足") as exc_info:
        HighsDispatchOptimizer(optimization).solve(dispatch_input)

    assert exc_info.value.diagnostics.total_shortage is not None
    assert exc_info.value.diagnostics.total_shortage > 0.0
    assert exc_info.value.diagnostics.success is False
