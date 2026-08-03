"""LightGBM/CatBoost 多目标多步回归接口。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from gas_power.features import CausalFeatureBuilder
from gas_power.models.base import (
    FitProgressCallback,
    ForecastModel,
    OptionalDependencyError,
    prediction_column,
    validate_prediction_request,
)
from gas_power.runtime import current_worker_count


class BoostingMultiHorizonModel(ForecastModel):
    """支持每步独立模型和将步长作为特征的全局模型。"""

    def __init__(
        self,
        backend: str,
        strategy: str,
        target_mode: str,
        feature_builder: CausalFeatureBuilder,
        parameters: Mapping[str, Any] | None = None,
        seed: int = 2026,
        interval_minutes: int = 15,
        baseline_model: ForecastModel | None = None,
        sample_weighting: Mapping[str, Any] | None = None,
        training_window_days: int | None = None,
    ):
        if backend not in {"lightgbm", "catboost"}:
            raise ValueError(f"不支持的提升模型后端: {backend}")
        if strategy not in {"direct", "global"}:
            raise ValueError("多步策略必须是 direct 或 global")
        if target_mode not in {"delta", "absolute", "residual"}:
            raise ValueError("目标模式必须是 delta、absolute 或 residual")
        if target_mode == "residual" and baseline_model is None:
            raise ValueError("残差模式必须提供 baseline_model")
        self.backend = backend
        self.strategy = strategy
        self.target_mode = target_mode
        self.feature_builder = feature_builder
        self.parameters = dict(parameters or {})
        self.seed = int(seed)
        self.interval_minutes = int(interval_minutes)
        self.baseline_model = baseline_model
        self.sample_weighting = dict(sample_weighting or {})
        self.training_window_days = (
            int(training_window_days) if training_window_days is not None else None
        )
        self.models_: dict[tuple[str, int] | str, Any] = {}
        self.feature_columns_: list[str] = []
        self.training_metadata_: dict[str, Any] = {}
        self._fit_progress_callback: FitProgressCallback | None = None

    def fit_progress_steps(
        self,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> int:
        if self.strategy == "direct":
            return len(target_columns) * len(horizons)
        return len(target_columns)

    def set_fit_progress_callback(
        self,
        callback: FitProgressCallback | None,
    ) -> None:
        self._fit_progress_callback = callback

    def _new_estimator(self) -> Any:
        if self.backend == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise OptionalDependencyError(
                    "已选择 LightGBM，但当前环境未安装。请离线准备 wheel，或执行 "
                    "`python -m pip install '.[lightgbm]'`；无需该模型时可继续使用默认基线。"
                ) from exc
            parameters = dict(self.parameters)
            parameters.setdefault("objective", "regression_l1")
            for key in (
                "sample_weighting",
                "training_window_days",
                "recency_half_life_days",
                "mape_weight_floor_quantile",
            ):
                parameters.pop(key, None)
            parameters.setdefault("random_state", self.seed)
            parameters.setdefault("verbosity", -1)
            if int(parameters.get("n_jobs", -1)) < 0:
                parameters["n_jobs"] = current_worker_count()
            return LGBMRegressor(**parameters)

        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise OptionalDependencyError(
                "已选择 CatBoost，但当前环境未安装。请离线准备 wheel，或执行 "
                "`python -m pip install '.[catboost]'`；无需该模型时可继续使用默认基线。"
            ) from exc
        parameters = dict(self.parameters)
        if str(parameters.get("loss_function", "")).lower() in {"mae", "huber"}:
            pass
        else:
            parameters.setdefault("loss_function", "MAE")
        for key in (
            "sample_weighting",
            "training_window_days",
            "recency_half_life_days",
            "mape_weight_floor_quantile",
        ):
            parameters.pop(key, None)
        parameters.setdefault("random_seed", self.seed)
        parameters.setdefault("verbose", False)
        parameters.pop("n_jobs", None)
        if int(parameters.get("thread_count", -1)) < 0:
            parameters["thread_count"] = current_worker_count()
        return CatBoostRegressor(**parameters)

    def _fit_estimator(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series,
    ) -> tuple[Any, bool]:
        constant_response = int(y.nunique(dropna=False)) == 1
        estimator = (
            DummyRegressor(strategy="constant", constant=float(y.iloc[0]))
            if constant_response
            else self._new_estimator()
        )
        estimator.fit(x, y, sample_weight=sample_weight)
        return estimator, constant_response

    def _add_target_time_features(
        self,
        features: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """加入预测目标时刻特征，使全局模型显式区分跨时段的多步任务。"""

        values = features.copy()
        target_index = pd.DatetimeIndex(features.index) + pd.Timedelta(
            minutes=int(horizon) * self.interval_minutes
        )
        minute_of_day = target_index.hour * 60 + target_index.minute
        day_angle = 2.0 * np.pi * minute_of_day / 1440.0
        week_angle = 2.0 * np.pi * (
            target_index.dayofweek * 1440 + minute_of_day
        ) / (7.0 * 1440.0)
        values["feat_horizon_steps"] = int(horizon)
        values["feat_target_time_day_sin"] = np.sin(day_angle)
        values["feat_target_time_day_cos"] = np.cos(day_angle)
        values["feat_target_time_week_sin"] = np.sin(week_angle)
        values["feat_target_time_week_cos"] = np.cos(week_angle)
        values["feat_target_time_hour"] = target_index.hour.astype(np.int8)
        values["feat_target_time_minute"] = target_index.minute.astype(np.int8)
        values["feat_target_time_slot"] = (
            minute_of_day // self.interval_minutes
        ).astype(np.int16)
        values["feat_target_time_month"] = target_index.month.astype(np.int8)
        values["feat_target_time_dayofweek"] = target_index.dayofweek.astype(np.int8)
        values["feat_target_time_is_weekend"] = (target_index.dayofweek >= 5).astype(np.int8)
        return values

    def fit(
        self,
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        horizons: Sequence[int],
        train_end: pd.Timestamp | None = None,
        *,
        raw_targets: pd.DataFrame | None = None,
        feature_matrix: pd.DataFrame | None = None,
        data_source: str = "training",
    ) -> "BoostingMultiHorizonModel":
        if data_source == "scoring":
            raise ValueError("评分期 scoring 数据禁止用于提升模型拟合")
        if frame.empty:
            raise ValueError("提升模型不能在空数据上拟合")
        end = pd.Timestamp(train_end if train_end is not None else frame.index.max())
        if end > frame.index.max():
            raise ValueError("train_end 超出输入时间范围")
        labels = frame[list(target_columns)] if raw_targets is None else raw_targets
        missing_labels = set(map(str, target_columns)).difference(labels.columns)
        if missing_labels:
            raise ValueError(f"原始标签缺少目标字段: {sorted(missing_labels)}")
        features = (
            self.feature_builder.transform(frame)
            if feature_matrix is None
            else feature_matrix.reindex(frame.index)
        )
        eligible = features.loc[:end]
        if self.training_window_days is not None:
            window_start = end - pd.Timedelta(days=self.training_window_days)
            eligible = eligible.loc[eligible.index >= window_start]
        self.feature_columns_ = [
            str(column) for column in eligible.columns if eligible[column].notna().any()
        ]
        if not self.feature_columns_:
            raise ValueError("没有可用于提升模型的有效特征")
        self.models_.clear()
        self.training_metadata_ = {
            "data_source": data_source,
            "train_start": str(eligible.index.min()),
            "train_end": str(end),
            "raw_labels": True,
            "sample_weighting": dict(self.sample_weighting),
            "training_rows_before_augmentation": 0,
            "augmented_training_rows": 0,
            "constant_response_models": [],
        }
        if self.target_mode == "residual" and self.baseline_model is not None:
            self.baseline_model.fit(
                frame.loc[:end],
                target_columns,
                horizons,
                train_end=end,
                raw_targets=labels.loc[:end],
                data_source=data_source,
            )

        if self.strategy == "direct":
            for target in target_columns:
                for horizon_value in horizons:
                    horizon = int(horizon_value)
                    x, y, weight = self._training_slice(
                        frame, labels, features, str(target), horizon, end
                    )
                    estimator, constant_response = self._fit_estimator(x, y, weight)
                    if constant_response:
                        self.training_metadata_["constant_response_models"].append(
                            prediction_column(str(target), horizon, self.interval_minutes)
                        )
                    self.models_[(str(target), horizon)] = estimator
                    if self._fit_progress_callback is not None:
                        self._fit_progress_callback(
                            f"{target}，预测步长 {horizon}/{max(horizons)}"
                        )
        else:
            for target in target_columns:
                x_parts: list[pd.DataFrame] = []
                y_parts: list[pd.Series] = []
                weight_parts: list[pd.Series] = []
                for horizon_value in horizons:
                    horizon = int(horizon_value)
                    x, y, weight = self._training_slice(
                        frame, labels, features, str(target), horizon, end
                    )
                    x_parts.append(x)
                    y_parts.append(y)
                    weight_parts.append(weight)
                estimator, constant_response = self._fit_estimator(
                    pd.concat(x_parts, axis=0),
                    pd.concat(y_parts, axis=0),
                    pd.concat(weight_parts, axis=0),
                )
                if constant_response:
                    self.training_metadata_["constant_response_models"].append(str(target))
                self.models_[str(target)] = estimator
                if self._fit_progress_callback is not None:
                    self._fit_progress_callback(f"{target}，全局多步模型")
        return self

    def _training_slice(
        self,
        frame: pd.DataFrame,
        raw_targets: pd.DataFrame,
        features: pd.DataFrame,
        target: str,
        horizon: int,
        train_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        if target not in raw_targets:
            raise ValueError(f"训练数据缺少目标 {target}")
        offset = pd.Timedelta(minutes=horizon * self.interval_minutes)
        origin_mask = (features.index + offset) <= train_end
        future_target = pd.to_numeric(raw_targets[target], errors="coerce").shift(-horizon)
        response = future_target.copy()
        if self.target_mode == "delta":
            response = response - frame[target]
        elif self.target_mode == "residual":
            if self.baseline_model is None:
                raise RuntimeError("残差基线未初始化")
            candidate = pd.DatetimeIndex(features.index[origin_mask & future_target.notna()])
            baseline = self.baseline_model.predict(
                frame, candidate, [target], [horizon]
            ).iloc[:, 0]
            response = pd.Series(np.nan, index=frame.index, dtype=float)
            response.loc[candidate] = (
                pd.to_numeric(future_target.loc[candidate], errors="coerce").to_numpy(dtype=float)
                - baseline.to_numpy(dtype=float)
            )
        valid = origin_mask & response.notna()
        if self.training_window_days is not None:
            valid &= features.index >= train_end - pd.Timedelta(days=self.training_window_days)
        x = self._add_target_time_features(
            features.loc[valid, self.feature_columns_], horizon
        )
        y = pd.to_numeric(response.loc[valid], errors="coerce")
        finite_y = np.isfinite(y.to_numpy(dtype=float))
        x = x.loc[finite_y]
        y = y.loc[finite_y]
        if len(x) < 20:
            raise ValueError(f"{target} 步长 {horizon} 的有效训练样本不足 20 条")
        actual = future_target.loc[x.index].abs()
        positive = actual.loc[actual > 0.0]
        quantile = float(self.sample_weighting.get("floor_quantile", 0.01))
        floor = float(positive.quantile(quantile)) if not positive.empty else 1.0
        floor = max(floor, float(self.sample_weighting.get("minimum_floor", 1.0e-6)))
        if bool(self.sample_weighting.get("mape", True)):
            weight = 1.0 / actual.clip(lower=floor)
        else:
            weight = pd.Series(1.0, index=x.index)
        half_life = self.sample_weighting.get("recency_half_life_days")
        if half_life is not None:
            age_days = (train_end - x.index).total_seconds() / 86_400.0
            weight = weight * np.power(0.5, age_days / float(half_life))
        weight = pd.Series(
            np.asarray(weight, dtype=float) / float(np.mean(weight)),
            index=x.index,
            dtype=float,
        )
        self.training_metadata_["training_rows_before_augmentation"] += len(x)
        x, y, weight, augmented_rows = self._augment_training_slice(
            frame,
            target,
            horizon,
            x,
            y,
            weight,
            train_end,
        )
        self.training_metadata_["augmented_training_rows"] += augmented_rows
        return x, y, weight

    def _augment_training_slice(
        self,
        frame: pd.DataFrame,
        target: str,
        horizon: int,
        x: pd.DataFrame,
        y: pd.Series,
        weight: pd.Series,
        train_end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
        """在单个训练折内对相邻同工况样本做局部 Mixup。"""

        ratio = float(self.sample_weighting.get("temporal_mixup_ratio", 0.0))
        if ratio == 0.0:
            return x, y, weight, 0
        if not 0.0 < ratio <= 1.0:
            raise ValueError("temporal_mixup_ratio 必须位于 (0, 1] 区间")
        minimum_lambda = float(
            self.sample_weighting.get("temporal_mixup_min_lambda", 0.70)
        )
        maximum_lambda = float(
            self.sample_weighting.get("temporal_mixup_max_lambda", 0.90)
        )
        if not 0.5 <= minimum_lambda <= maximum_lambda < 1.0:
            raise ValueError("Mixup 插值系数必须满足 0.5 <= min <= max < 1")

        current = pd.to_numeric(frame[target], errors="coerce").reindex(x.index)
        previous = pd.to_numeric(frame[target], errors="coerce").shift(1).reindex(x.index)
        online_threshold = float(
            self.sample_weighting.get("regime_online_threshold_mw", 1.0)
        )
        stable_threshold = float(
            self.sample_weighting.get("regime_stable_delta_threshold_mw", 2.0)
        )
        current_online = current > online_threshold
        previous_online = previous > online_threshold
        delta = current - previous
        regimes = np.select(
            [
                current_online & ~previous_online,
                ~current_online & previous_online,
                current_online & (delta > stable_threshold),
                current_online & (delta < -stable_threshold),
                current_online,
            ],
            ["startup", "shutdown", "ramp_up", "ramp_down", "stable"],
            default="offline",
        )
        time_gap = pd.Series(x.index, index=x.index).diff()
        adjacent = time_gap.eq(pd.Timedelta(minutes=self.interval_minutes)).to_numpy()
        same_regime = np.r_[False, regimes[1:] == regimes[:-1]]
        eligible_positions = np.flatnonzero(adjacent & same_regime)
        augmented_rows = min(int(round(len(x) * ratio)), len(eligible_positions))
        if augmented_rows <= 0:
            return x, y, weight, 0

        stable_target_seed = sum((index + 1) * ord(char) for index, char in enumerate(target))
        timestamp_seed = int(pd.Timestamp(train_end).value // 10**9)
        random_seed = (
            self.seed + 1009 * int(horizon) + stable_target_seed + timestamp_seed
        ) % (2**32)
        generator = np.random.default_rng(random_seed)
        selected = np.sort(
            generator.choice(eligible_positions, size=augmented_rows, replace=False)
        )
        blend = generator.uniform(
            minimum_lambda, maximum_lambda, size=augmented_rows
        )

        left = x.iloc[selected].to_numpy(dtype=float)
        right = x.iloc[selected - 1].to_numpy(dtype=float)
        mixed = blend[:, None] * left + (1.0 - blend[:, None]) * right
        mixed = np.where(
            np.isfinite(left) & ~np.isfinite(right),
            left,
            np.where(~np.isfinite(left) & np.isfinite(right), right, mixed),
        )
        augmented_x = pd.DataFrame(mixed, columns=x.columns)
        y_values = y.to_numpy(dtype=float)
        augmented_y = pd.Series(
            blend * y_values[selected] + (1.0 - blend) * y_values[selected - 1],
            dtype=float,
        )
        weight_values = weight.to_numpy(dtype=float)
        augmented_weight = pd.Series(
            blend * weight_values[selected]
            + (1.0 - blend) * weight_values[selected - 1],
            dtype=float,
        )

        combined_x = pd.concat([x.reset_index(drop=True), augmented_x], ignore_index=True)
        combined_y = pd.concat([y.reset_index(drop=True), augmented_y], ignore_index=True)
        combined_weight = pd.concat(
            [weight.reset_index(drop=True), augmented_weight], ignore_index=True
        )
        combined_weight = combined_weight / float(combined_weight.mean())
        return combined_x, combined_y, combined_weight, augmented_rows

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        validate_prediction_request(frame, origins, target_columns, horizons)
        if not self.models_ or not self.feature_columns_:
            raise RuntimeError("提升模型尚未拟合")
        features = self.feature_builder.transform(frame).loc[origins, self.feature_columns_]
        output = pd.DataFrame(index=origins)
        output.index.name = "datetime"
        baseline_predictions = None
        if self.target_mode == "residual":
            if self.baseline_model is None:
                raise RuntimeError("残差基线未初始化")
            baseline_predictions = self.baseline_model.predict(
                frame, origins, target_columns, horizons
            )
        for target in target_columns:
            current = pd.to_numeric(frame.loc[origins, str(target)], errors="coerce").to_numpy(dtype=float)
            for horizon_value in horizons:
                horizon = int(horizon_value)
                if self.strategy == "direct":
                    estimator = self.models_.get((str(target), horizon))
                    if estimator is None:
                        raise ValueError(f"模型未训练目标 {target} 步长 {horizon}")
                    horizon_features = self._add_target_time_features(features, horizon)
                    prediction = np.asarray(estimator.predict(horizon_features), dtype=float)
                else:
                    estimator = self.models_.get(str(target))
                    if estimator is None:
                        raise ValueError(f"全局模型未训练目标 {target}")
                    global_features = self._add_target_time_features(features, horizon)
                    prediction = np.asarray(estimator.predict(global_features), dtype=float)
                if self.target_mode == "delta":
                    prediction = current + prediction
                column = prediction_column(str(target), horizon, self.interval_minutes)
                if self.target_mode == "residual":
                    if baseline_predictions is None:
                        raise RuntimeError("残差基线预测不存在")
                    prediction = baseline_predictions[column].to_numpy(dtype=float) + prediction
                output[column] = prediction
        return output
