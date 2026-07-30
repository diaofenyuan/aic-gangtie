"""基于 SciPy/HiGHS 的最小可运行煤气发电调度模型。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from gas_power.config import configured_value


@dataclass(frozen=True)
class DispatchInput:
    timestamps: pd.DatetimeIndex
    production: Mapping[str, np.ndarray]
    user_demand: Mapping[str, np.ndarray]
    initial_storage: Mapping[str, float]
    electricity_price: np.ndarray
    baseline_generation_mw: np.ndarray | None = None
    baseline_flare_volume: np.ndarray | None = None


@dataclass
class OptimizationDiagnostics:
    success: bool
    solver_status: int
    solver_message: str
    objective_value: float | None
    runtime_seconds: float
    max_constraint_violation: float | None
    total_shortage: float | None
    total_flare: float | None
    total_generation_mwh: float | None
    priority_mode: str = "weighted"
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    shortage_periods: int | None = None
    price_weighted_load_ratio: float | None = None
    optimized_benefit: float | None = None
    baseline_benefit: float | None = None
    relative_benefit_improvement: float | None = None
    economic_metric_status: str = "未计算"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchResult:
    gas_plan: pd.DataFrame
    unit_plan: pd.DataFrame
    storage_plan: pd.DataFrame
    diagnostics: OptimizationDiagnostics


class OptimizationError(RuntimeError):
    """求解失败或返回违反约束的方案。"""

    def __init__(self, message: str, diagnostics: OptimizationDiagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class _VariableLayout:
    def __init__(self, gas_count: int, unit_count: int, periods: int):
        self.gas_count = gas_count
        self.unit_count = unit_count
        self.periods = periods
        gas_block = gas_count * periods
        unit_block = unit_count * periods
        self.offsets = {
            "use": 0,
            "storage": gas_block,
            "flare": 2 * gas_block,
            "shortage": 3 * gas_block,
            "load": 4 * gas_block,
            "on": 4 * gas_block + unit_block,
            "start": 4 * gas_block + 2 * unit_block,
        }
        self.size = 4 * gas_block + 3 * unit_block

    def gas(self, block: str, gas: int, period: int) -> int:
        return self.offsets[block] + gas * self.periods + period

    def unit(self, block: str, unit: int, period: int) -> int:
        return self.offsets[block] + unit * self.periods + period


class _ConstraintBuilder:
    def __init__(self, variable_count: int):
        self.variable_count = variable_count
        self.rows: list[dict[int, float]] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: Mapping[int, float], lower: float, upper: float) -> None:
        self.rows.append(dict(coefficients))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self) -> tuple[LinearConstraint, coo_matrix]:
        row_indices: list[int] = []
        column_indices: list[int] = []
        data: list[float] = []
        for row_index, coefficients in enumerate(self.rows):
            for column_index, value in coefficients.items():
                if value != 0.0:
                    row_indices.append(row_index)
                    column_indices.append(column_index)
                    data.append(float(value))
        matrix = coo_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(self.rows), self.variable_count),
            dtype=float,
        ).tocsr()
        return (
            LinearConstraint(
                matrix,
                np.asarray(self.lower, dtype=float),
                np.asarray(self.upper, dtype=float),
            ),
            matrix,
        )


class HighsDispatchOptimizer:
    """连续煤气流量与二进制机组状态的可替换优化实现。"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.gas_config = config.get("gas_types", {})
        self.units = config.get("units", [])
        if not isinstance(self.gas_config, Mapping) or len(self.gas_config) != 3:
            raise ValueError("优化模型需要恰好三类煤气配置")
        if not isinstance(self.units, list) or len(self.units) != 6:
            raise ValueError("优化模型需要恰好六台机组配置")
        self.gas_types = list(self.gas_config.keys())
        self.interval_minutes = int(config.get("interval_minutes", 15))

    def solve(self, dispatch_input: DispatchInput) -> DispatchResult:
        self._validate_input(dispatch_input)
        periods = len(dispatch_input.timestamps)
        gas_count = len(self.gas_types)
        unit_count = len(self.units)
        layout = _VariableLayout(gas_count, unit_count, periods)
        economic_objective = np.zeros(layout.size, dtype=float)
        shortage_objective = np.zeros(layout.size, dtype=float)
        flare_objective = np.zeros(layout.size, dtype=float)
        lower_bounds = np.zeros(layout.size, dtype=float)
        upper_bounds = np.full(layout.size, np.inf, dtype=float)
        integrality = np.zeros(layout.size, dtype=np.int8)

        objective_config = self.config.get("objective", {})
        shortage_penalty = configured_value(
            objective_config.get("shortage_penalty"),
            "optimization.objective.shortage_penalty",
        )
        flare_penalty = configured_value(
            objective_config.get("flare_penalty"),
            "optimization.objective.flare_penalty",
        )
        startup_penalty = configured_value(
            objective_config.get("startup_penalty"),
            "optimization.objective.startup_penalty",
        )
        revenue_scale = configured_value(
            objective_config.get("energy_revenue_scale"),
            "optimization.objective.energy_revenue_scale",
        )
        interval_hours = self.interval_minutes / 60.0

        holder_min_fraction = float(self.config.get("holder_min_fraction", 0.15))
        holder_max_fraction = float(self.config.get("holder_max_fraction", 0.90))
        ramp_fraction = float(self.config.get("ramp_fraction_per_minute", 0.10))

        conversions: list[float] = []
        capacities: list[float] = []
        for gas_index, gas_type in enumerate(self.gas_types):
            gas_settings = self.gas_config[gas_type]
            capacity = configured_value(
                gas_settings.get("holder_capacity"),
                f"optimization.gas_types.{gas_type}.holder_capacity",
            )
            conversion = configured_value(
                gas_settings.get("conversion_mw_per_volume"),
                f"optimization.gas_types.{gas_type}.conversion_mw_per_volume",
            )
            capacities.append(capacity)
            conversions.append(conversion)
            demand = np.asarray(dispatch_input.user_demand[gas_type], dtype=float)
            for period in range(periods):
                storage_index = layout.gas("storage", gas_index, period)
                lower_bounds[storage_index] = capacity * holder_min_fraction
                upper_bounds[storage_index] = capacity * holder_max_fraction
                shortage_index = layout.gas("shortage", gas_index, period)
                flare_index = layout.gas("flare", gas_index, period)
                upper_bounds[shortage_index] = demand[period]
                shortage_objective[shortage_index] = 1.0
                flare_objective[flare_index] = 1.0

        for unit_index, unit in enumerate(self.units):
            rated = float(unit["rated_mw"])
            for period in range(periods):
                upper_bounds[layout.unit("load", unit_index, period)] = rated
                upper_bounds[layout.unit("on", unit_index, period)] = 1.0
                upper_bounds[layout.unit("start", unit_index, period)] = 1.0
                integrality[layout.unit("on", unit_index, period)] = 1
                integrality[layout.unit("start", unit_index, period)] = 1
                economic_objective[layout.unit("start", unit_index, period)] = startup_penalty
                economic_objective[layout.unit("load", unit_index, period)] = (
                    -float(dispatch_input.electricity_price[period])
                    * interval_hours
                    * revenue_scale
                )

        constraints = _ConstraintBuilder(layout.size)
        self._add_material_balances(constraints, layout, dispatch_input)
        self._add_generation_conversion(constraints, layout, conversions, periods)
        self._add_unit_constraints(constraints, layout, periods, ramp_fraction)
        bounds = Bounds(lower_bounds, upper_bounds)

        solver_config = self.config.get("solver", {})
        final_options = {
            "time_limit": float(solver_config.get("time_limit_seconds", 20.0)),
            "mip_rel_gap": float(solver_config.get("mip_relative_gap", 0.01)),
            "disp": False,
        }
        priority_options = dict(final_options)
        priority_options["mip_rel_gap"] = 0.0
        priority_mode = str(self.config.get("priority_mode", "lexicographic"))
        if priority_mode not in {"lexicographic", "weighted"}:
            raise ValueError("optimization.priority_mode 必须是 lexicographic 或 weighted")
        stage_results: list[dict[str, Any]] = []
        linear_constraint, matrix = constraints.build()

        def solve_stage(name: str, objective: np.ndarray, options: Mapping[str, Any]) -> Any:
            stage_started = time.perf_counter()
            stage_result = milp(
                c=objective,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints.build()[0],
                options=dict(options),
            )
            stage_results.append(
                {
                    "name": name,
                    "success": bool(stage_result.success),
                    "status": int(stage_result.status),
                    "message": str(stage_result.message),
                    "objective_value": (
                        float(stage_result.fun) if stage_result.fun is not None else None
                    ),
                    "runtime_seconds": float(time.perf_counter() - stage_started),
                }
            )
            return stage_result

        started = time.perf_counter()
        if priority_mode == "lexicographic":
            result = solve_stage("minimize_shortage", shortage_objective, priority_options)
            if bool(result.success) and result.x is not None:
                shortage_optimum = float(shortage_objective @ result.x)
                shortage_bound = {
                    int(index): 1.0
                    for index in np.flatnonzero(shortage_objective)
                }
                priority_tolerance = float(
                    solver_config.get("priority_tolerance", 1.0e-7)
                )
                constraints.add(
                    shortage_bound,
                    -np.inf,
                    shortage_optimum + priority_tolerance,
                )
                result = solve_stage("minimize_flare", flare_objective, priority_options)
            if bool(result.success) and result.x is not None:
                flare_optimum = float(flare_objective @ result.x)
                flare_bound = {
                    int(index): 1.0
                    for index in np.flatnonzero(flare_objective)
                }
                constraints.add(
                    flare_bound,
                    -np.inf,
                    flare_optimum
                    + float(solver_config.get("priority_tolerance", 1.0e-7)),
                )
                result = solve_stage("maximize_economic_benefit", economic_objective, final_options)
        else:
            weighted_objective = (
                economic_objective
                + shortage_penalty * shortage_objective
                + flare_penalty * flare_objective
            )
            result = solve_stage("weighted_objective", weighted_objective, final_options)
        runtime = time.perf_counter() - started
        linear_constraint, matrix = constraints.build()

        violation = None
        if result.x is not None:
            violation = self._maximum_violation(
                result.x,
                matrix,
                np.asarray(constraints.lower),
                np.asarray(constraints.upper),
                lower_bounds,
                upper_bounds,
                integrality,
            )
        diagnostics = self._diagnostics(
            result,
            runtime,
            violation,
            layout,
            periods,
            interval_hours,
            dispatch_input,
            priority_mode,
            stage_results,
            flare_penalty,
            revenue_scale,
        )
        tolerance = float(solver_config.get("feasibility_tolerance", 1.0e-5))
        if not bool(result.success) or result.x is None:
            raise OptimizationError(
                f"HiGHS 优化失败: status={result.status}, message={result.message}",
                diagnostics,
            )
        if violation is None or violation > tolerance:
            diagnostics.success = False
            raise OptimizationError(
                f"HiGHS 返回方案的最大约束违反量 {violation} 超过容差 {tolerance}",
                diagnostics,
            )
        shortage_tolerance = float(self.config.get("shortage_tolerance", 1.0e-6))
        if (
            bool(self.config.get("require_zero_shortage", True))
            and diagnostics.total_shortage is not None
            and diagnostics.total_shortage > shortage_tolerance
        ):
            diagnostics.success = False
            raise OptimizationError(
                "预测资源边界无法保证生产用户零供气不足，拒绝输出越界计划；"
                f"最小供气不足量={diagnostics.total_shortage}",
                diagnostics,
            )
        return self._extract_result(result.x, dispatch_input, layout, diagnostics)

    def _validate_input(self, dispatch_input: DispatchInput) -> None:
        periods = len(dispatch_input.timestamps)
        if periods <= 0:
            raise ValueError("优化时域不能为空")
        if not dispatch_input.timestamps.is_monotonic_increasing or not dispatch_input.timestamps.is_unique:
            raise ValueError("优化时间戳必须严格递增且唯一")
        prices = np.asarray(dispatch_input.electricity_price, dtype=float)
        if prices.shape != (periods,) or not np.isfinite(prices).all():
            raise ValueError("电价长度不匹配或包含非有限值")

        min_fraction = float(self.config.get("holder_min_fraction", 0.15))
        max_fraction = float(self.config.get("holder_max_fraction", 0.90))
        for gas_type in self.gas_types:
            if gas_type not in dispatch_input.production or gas_type not in dispatch_input.user_demand:
                raise ValueError(f"优化输入缺少 {gas_type} 的产量或用户需求")
            production = np.asarray(dispatch_input.production[gas_type], dtype=float)
            demand = np.asarray(dispatch_input.user_demand[gas_type], dtype=float)
            if production.shape != (periods,) or demand.shape != (periods,):
                raise ValueError(f"{gas_type} 的资源预测长度不匹配")
            if not np.isfinite(production).all() or not np.isfinite(demand).all():
                raise ValueError(f"{gas_type} 的资源预测包含非有限值")
            if (production < 0.0).any() or (demand < 0.0).any():
                raise ValueError(f"{gas_type} 的产量和用户需求不得为负")
            capacity = configured_value(
                self.gas_config[gas_type].get("holder_capacity"),
                f"optimization.gas_types.{gas_type}.holder_capacity",
            )
            storage = float(dispatch_input.initial_storage[gas_type])
            if not capacity * min_fraction <= storage <= capacity * max_fraction:
                raise ValueError(
                    f"{gas_type} 初始柜容 {storage} 不在安全区间 "
                    f"[{capacity * min_fraction}, {capacity * max_fraction}]"
                )

    def _add_material_balances(
        self,
        constraints: _ConstraintBuilder,
        layout: _VariableLayout,
        dispatch_input: DispatchInput,
    ) -> None:
        for gas_index, gas_type in enumerate(self.gas_types):
            production = np.asarray(dispatch_input.production[gas_type], dtype=float)
            demand = np.asarray(dispatch_input.user_demand[gas_type], dtype=float)
            initial_storage = float(dispatch_input.initial_storage[gas_type])
            for period in range(layout.periods):
                coefficients = {
                    layout.gas("storage", gas_index, period): 1.0,
                    layout.gas("use", gas_index, period): 1.0,
                    layout.gas("flare", gas_index, period): 1.0,
                    layout.gas("shortage", gas_index, period): -1.0,
                }
                rhs = float(production[period] - demand[period])
                if period == 0:
                    rhs += initial_storage
                else:
                    coefficients[layout.gas("storage", gas_index, period - 1)] = -1.0
                constraints.add(coefficients, rhs, rhs)

    def _add_generation_conversion(
        self,
        constraints: _ConstraintBuilder,
        layout: _VariableLayout,
        conversions: Sequence[float],
        periods: int,
    ) -> None:
        for period in range(periods):
            coefficients: dict[int, float] = {}
            for unit_index in range(layout.unit_count):
                coefficients[layout.unit("load", unit_index, period)] = 1.0
            for gas_index, conversion in enumerate(conversions):
                coefficients[layout.gas("use", gas_index, period)] = -float(conversion)
            constraints.add(coefficients, 0.0, 0.0)

    def _add_unit_constraints(
        self,
        constraints: _ConstraintBuilder,
        layout: _VariableLayout,
        periods: int,
        ramp_fraction: float,
    ) -> None:
        for unit_index, unit in enumerate(self.units):
            rated = float(unit["rated_mw"])
            minimum_load = rated * float(unit.get("min_fraction", 0.60))
            ramp_limit = rated * ramp_fraction * self.interval_minutes
            initial_on = float(unit.get("initial_on", 0))
            initial_load = float(unit.get("initial_load_mw", 0.0))
            for period in range(periods):
                load = layout.unit("load", unit_index, period)
                on = layout.unit("on", unit_index, period)
                start = layout.unit("start", unit_index, period)
                constraints.add({load: 1.0, on: -rated}, -np.inf, 0.0)
                constraints.add({load: -1.0, on: minimum_load}, -np.inf, 0.0)
                constraints.add({start: 1.0, on: -1.0}, -np.inf, 0.0)

                if period == 0:
                    constraints.add({on: 1.0, start: -1.0}, -np.inf, initial_on)
                    constraints.add({load: 1.0}, -np.inf, initial_load + ramp_limit)
                    constraints.add({load: -1.0}, -np.inf, ramp_limit - initial_load)
                else:
                    previous_on = layout.unit("on", unit_index, period - 1)
                    previous_load = layout.unit("load", unit_index, period - 1)
                    constraints.add(
                        {on: 1.0, previous_on: -1.0, start: -1.0},
                        -np.inf,
                        0.0,
                    )
                    constraints.add(
                        {load: 1.0, previous_load: -1.0}, -np.inf, ramp_limit
                    )
                    constraints.add(
                        {load: -1.0, previous_load: 1.0}, -np.inf, ramp_limit
                    )

    @staticmethod
    def _maximum_violation(
        solution: np.ndarray,
        matrix: coo_matrix,
        constraint_lower: np.ndarray,
        constraint_upper: np.ndarray,
        variable_lower: np.ndarray,
        variable_upper: np.ndarray,
        integrality: np.ndarray,
    ) -> float:
        values = matrix @ solution
        lower_violation = np.where(
            np.isfinite(constraint_lower), np.maximum(constraint_lower - values, 0.0), 0.0
        )
        upper_violation = np.where(
            np.isfinite(constraint_upper), np.maximum(values - constraint_upper, 0.0), 0.0
        )
        variable_lower_violation = np.maximum(variable_lower - solution, 0.0)
        variable_upper_violation = np.where(
            np.isfinite(variable_upper), np.maximum(solution - variable_upper, 0.0), 0.0
        )
        integer_values = solution[integrality == 1]
        integer_violation = (
            np.max(np.abs(integer_values - np.round(integer_values)))
            if len(integer_values)
            else 0.0
        )
        return float(
            max(
                np.max(lower_violation, initial=0.0),
                np.max(upper_violation, initial=0.0),
                np.max(variable_lower_violation, initial=0.0),
                np.max(variable_upper_violation, initial=0.0),
                integer_violation,
            )
        )

    def _diagnostics(
        self,
        result: Any,
        runtime: float,
        violation: float | None,
        layout: _VariableLayout,
        periods: int,
        interval_hours: float,
        dispatch_input: DispatchInput,
        priority_mode: str,
        stage_results: list[dict[str, Any]],
        flare_penalty: float,
        revenue_scale: float,
    ) -> OptimizationDiagnostics:
        total_shortage = None
        total_flare = None
        total_generation = None
        shortage_periods = None
        price_weighted_load_ratio = None
        optimized_benefit = None
        baseline_benefit = None
        relative_benefit_improvement = None
        economic_metric_status = "未计算"
        if result.x is not None:
            shortage_matrix = np.asarray(
                [
                    sum(
                        result.x[layout.gas("shortage", gas, period)]
                        for gas in range(layout.gas_count)
                    )
                    for period in range(periods)
                ],
                dtype=float,
            )
            flare_matrix = np.asarray(
                [
                    sum(
                        result.x[layout.gas("flare", gas, period)]
                        for gas in range(layout.gas_count)
                    )
                    for period in range(periods)
                ],
                dtype=float,
            )
            generation = np.asarray(
                [
                    sum(
                        result.x[layout.unit("load", unit, period)]
                        for unit in range(layout.unit_count)
                    )
                    for period in range(periods)
                ],
                dtype=float,
            )
            total_shortage = float(shortage_matrix.sum())
            total_flare = float(flare_matrix.sum())
            total_generation = float(generation.sum() * interval_hours)
            shortage_periods = int(np.count_nonzero(shortage_matrix > 1.0e-9))
            prices = np.asarray(dispatch_input.electricity_price, dtype=float)
            mean_price = float(prices.mean()) if len(prices) else 0.0
            denominator = float(generation.sum() * mean_price)
            if denominator > 0.0:
                price_weighted_load_ratio = float(
                    np.dot(generation, prices) / denominator
                )
            optimized_benefit = float(
                np.dot(generation, prices)
                * interval_hours
                * revenue_scale
                - total_flare * flare_penalty
            )
            baseline = dispatch_input.baseline_generation_mw
            baseline_flare = dispatch_input.baseline_flare_volume
            if baseline is not None:
                baseline = np.asarray(baseline, dtype=float)
                if baseline.shape != (periods,) or not np.isfinite(baseline).all():
                    raise ValueError("baseline_generation_mw 长度或数值无效")
                baseline_replacement = float(
                    np.dot(baseline, prices) * interval_hours * revenue_scale
                )
                if baseline_flare is not None:
                    baseline_flare = np.asarray(baseline_flare, dtype=float)
                    if baseline_flare.shape != (periods,) or not np.isfinite(baseline_flare).all():
                        raise ValueError("baseline_flare_volume 长度或数值无效")
                    baseline_benefit = baseline_replacement - float(
                        baseline_flare.sum() * flare_penalty
                    )
                    if abs(baseline_benefit) > 1.0e-12:
                        relative_benefit_improvement = float(
                            (optimized_benefit - baseline_benefit) / baseline_benefit
                        )
                    economic_metric_status = "已包含基准放散惩罚"
                else:
                    economic_metric_status = "仅有基准发电量，缺少基准放散量；相对收益不计算"
        return OptimizationDiagnostics(
            success=bool(result.success),
            solver_status=int(result.status),
            solver_message=str(result.message),
            objective_value=float(result.fun) if result.fun is not None else None,
            runtime_seconds=float(runtime),
            max_constraint_violation=violation,
            total_shortage=total_shortage,
            total_flare=total_flare,
            total_generation_mwh=total_generation,
            priority_mode=priority_mode,
            stage_results=stage_results,
            shortage_periods=shortage_periods,
            price_weighted_load_ratio=price_weighted_load_ratio,
            optimized_benefit=optimized_benefit,
            baseline_benefit=baseline_benefit,
            relative_benefit_improvement=relative_benefit_improvement,
            economic_metric_status=economic_metric_status,
        )

    def _extract_result(
        self,
        solution: np.ndarray,
        dispatch_input: DispatchInput,
        layout: _VariableLayout,
        diagnostics: OptimizationDiagnostics,
    ) -> DispatchResult:
        output_columns = self.config.get("output_columns", {})
        gas_plan = pd.DataFrame(index=dispatch_input.timestamps)
        storage_plan = pd.DataFrame(index=dispatch_input.timestamps)
        for gas_index, gas_type in enumerate(self.gas_types):
            output_name = str(output_columns[gas_type])
            gas_plan[output_name] = [
                solution[layout.gas("use", gas_index, period)]
                for period in range(layout.periods)
            ]
            storage_plan[f"storage_{gas_type}"] = [
                solution[layout.gas("storage", gas_index, period)]
                for period in range(layout.periods)
            ]
            storage_plan[f"flare_{gas_type}"] = [
                solution[layout.gas("flare", gas_index, period)]
                for period in range(layout.periods)
            ]
            storage_plan[f"shortage_{gas_type}"] = [
                solution[layout.gas("shortage", gas_index, period)]
                for period in range(layout.periods)
            ]

        unit_plan = pd.DataFrame(index=dispatch_input.timestamps)
        load_columns: list[str] = []
        generator_1_columns: list[str] = []
        for unit_index, unit in enumerate(self.units):
            name = str(unit["name"])
            load_column = f"load_{name}"
            load_columns.append(load_column)
            if str(unit.get("group", "")) == "generator_1":
                generator_1_columns.append(load_column)
            unit_plan[load_column] = [
                solution[layout.unit("load", unit_index, period)]
                for period in range(layout.periods)
            ]
            unit_plan[f"on_{name}"] = np.rint(
                [
                    solution[layout.unit("on", unit_index, period)]
                    for period in range(layout.periods)
                ]
            ).astype(np.int8)
            unit_plan[f"start_{name}"] = np.rint(
                [
                    solution[layout.unit("start", unit_index, period)]
                    for period in range(layout.periods)
                ]
            ).astype(np.int8)

        # PDF 明确 generator_1 是四套 50MW 机组的合计，generator_all 是六套总负荷。
        if len(generator_1_columns) != 4:
            raise ValueError("优化机组配置必须有四套机组归属于 generator_1")
        unit_plan["generator_1_plan_mw"] = unit_plan[generator_1_columns].sum(axis=1)
        unit_plan["generator_all_plan_mw"] = unit_plan[load_columns].sum(axis=1)

        for frame in (gas_plan, storage_plan, unit_plan):
            frame.index.name = "datetime"
        return DispatchResult(
            gas_plan=gas_plan,
            unit_plan=unit_plan,
            storage_plan=storage_plan,
            diagnostics=diagnostics,
        )
