"""评估与逐步提升树互补的多输出自回归候选。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("generator_1", "generator_all")
HORIZONS = tuple(range(1, 9))
KEYS = ["fold", "origin", "target_datetime", "target", "horizon_steps"]
COMPACT_LAG_PREFIXES = (
    "feat_lag_generator_1_",
    "feat_lag_generator_all_",
    "feat_lag_blast_furnace_1_",
    "feat_lag_coke_oven_1_",
    "feat_lag_converter_1_",
)
COMPACT_ROLL_PREFIXES = tuple(
    prefix.replace("feat_lag_", "feat_roll_") for prefix in COMPACT_LAG_PREFIXES
)
COMPACT_LAG_SUFFIXES = tuple(
    f"_{step}" for step in (1, 2, 4, 8, 16, 32, 96, 192)
)
COMPACT_ROLL_SUFFIXES = tuple(
    f"_{window}_{statistic}"
    for window in (4, 8, 16, 32, 96)
    for statistic in ("mean", "std", "slope")
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default="2026-08-02_20-02-43*")
    parser.add_argument("--extra-trees", action="store_true")
    parser.add_argument(
        "--only-extra-trees",
        action="store_true",
        help="跳过已经完成的 Ridge 网格，仅评估 ExtraTrees",
    )
    parser.add_argument("--relative-lightgbm", action="store_true")
    parser.add_argument(
        "--only-relative-lightgbm",
        action="store_true",
        help="跳过其他候选，仅评估相对变化率 LightGBM",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "cache"
        / "experiments"
        / "multioutput_sklearn_oof",
    )
    return parser.parse_args()


def _baseline_directory(pattern: str) -> Path:
    matches = sorted((PROJECT_ROOT / "outputs").glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"基线运行目录应唯一匹配，实际找到 {len(matches)} 个: {pattern}"
        )
    return matches[0]


def _is_target_feature(column: str) -> bool:
    return (
        column.startswith("feat_time_")
        or column.startswith("feat_current_generator_")
        or column.startswith("feat_diff_generator_")
        or column.startswith("feat_accel_generator_")
        or column.startswith("feat_ewma_generator_")
        or column.startswith("feat_missing__generator_")
        or column.startswith("feat_outlier__generator_")
        or (
            column.startswith(
                ("feat_lag_generator_1_", "feat_lag_generator_all_")
            )
            and column.endswith(COMPACT_LAG_SUFFIXES)
        )
        or (
            column.startswith(
                ("feat_roll_generator_1_", "feat_roll_generator_all_")
            )
            and column.endswith(COMPACT_ROLL_SUFFIXES)
        )
    )


def _is_compact_feature(column: str) -> bool:
    return (
        _is_target_feature(column)
        or column.startswith(
            (
                "feat_gas_balance_",
                "feat_holder_",
                "feat_state_",
                "feat_missing__",
            )
        )
        or (
            column.startswith(COMPACT_LAG_PREFIXES)
            and column.endswith(COMPACT_LAG_SUFFIXES)
        )
        or (
            column.startswith(COMPACT_ROLL_PREFIXES)
            and column.endswith(COMPACT_ROLL_SUFFIXES)
        )
    )


def _feature_sets(columns: list[str]) -> dict[str, list[str]]:
    numeric = [column for column in columns if column != "datetime"]
    return {
        "target": [column for column in numeric if _is_target_feature(column)],
        "compact": [column for column in numeric if _is_compact_feature(column)],
        "all": numeric,
    }


def _future_ratios(
    processed: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    current = pd.to_numeric(processed[target], errors="coerce")
    output = pd.DataFrame(index=processed.index)
    for horizon in HORIZONS:
        future = current.shift(-horizon)
        output[horizon] = future / current - 1.0
    return output


def _add_target_time_features(
    features: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """与正式提升树保持一致地加入目标时刻周期特征。"""

    output = features.copy()
    target_index = pd.DatetimeIndex(features.index) + pd.Timedelta(
        minutes=15 * int(horizon)
    )
    minute = target_index.hour * 60 + target_index.minute
    day_angle = 2.0 * np.pi * minute / 1440.0
    week_angle = 2.0 * np.pi * (
        target_index.dayofweek * 1440 + minute
    ) / (7.0 * 1440.0)
    output["feat_horizon_steps"] = int(horizon)
    output["feat_target_time_day_sin"] = np.sin(day_angle)
    output["feat_target_time_day_cos"] = np.cos(day_angle)
    output["feat_target_time_week_sin"] = np.sin(week_angle)
    output["feat_target_time_week_cos"] = np.cos(week_angle)
    output["feat_target_time_hour"] = target_index.hour.astype(np.int8)
    output["feat_target_time_dayofweek"] = target_index.dayofweek.astype(np.int8)
    output["feat_target_time_is_weekend"] = (
        target_index.dayofweek >= 5
    ).astype(np.int8)
    return output


def _mape(frame: pd.DataFrame, column: str) -> float:
    actual = frame["y_true"].to_numpy(dtype=float)
    predicted = frame[column].to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    return float(
        np.mean(
            np.abs(predicted[finite] - actual[finite])
            / np.maximum(np.abs(actual[finite]), 1.0e-6)
        )
    )


def _score(frame: pd.DataFrame, column: str) -> float:
    mapes = [
        _mape(group, column)
        for _, group in frame.groupby(["target", "horizon_steps"], sort=True)
    ]
    return 100.0 * (1.0 - float(np.mean(mapes)))


def _prediction_rows(
    fold: str,
    origins: pd.DatetimeIndex,
    target: str,
    current: pd.Series,
    ratios: np.ndarray,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for origin_index, origin in enumerate(origins):
        level = float(current.at[origin])
        for horizon_index, horizon in enumerate(HORIZONS):
            rows.append(
                {
                    "fold": fold,
                    "origin": origin,
                    "target_datetime": origin
                    + pd.Timedelta(minutes=15 * horizon),
                    "target": target,
                    "horizon_steps": horizon,
                    method: level * (1.0 + ratios[origin_index, horizon_index]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _arguments()
    baseline = _baseline_directory(args.baseline_run)
    incumbent = pd.read_csv(
        baseline / "reports" / "high_accuracy_fused_leave_one_fold.csv",
        encoding="utf-8",
    ).rename(columns={"y_pred": "incumbent"})
    incumbent["origin"] = pd.to_datetime(incumbent["origin"], errors="raise")
    incumbent["target_datetime"] = pd.to_datetime(
        incumbent["target_datetime"], errors="raise"
    )

    processed = pd.read_csv(
        PROJECT_ROOT / "cache" / "processed.csv", encoding="utf-8"
    )
    processed["datetime"] = pd.to_datetime(processed["datetime"], errors="raise")
    processed = processed.set_index("datetime").sort_index()
    feature_header = pd.read_csv(
        PROJECT_ROOT / "cache" / "features.csv", nrows=0, encoding="utf-8"
    ).columns.astype(str).tolist()
    feature_sets = _feature_sets(feature_header)
    selected_columns = sorted(
        set(column for columns in feature_sets.values() for column in columns)
    )
    features = pd.read_csv(
        PROJECT_ROOT / "cache" / "features.csv",
        usecols=["datetime", *selected_columns],
        encoding="utf-8",
    )
    features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
    features = features.set_index("datetime").sort_index()
    features = features.replace([np.inf, -np.inf], np.nan)

    only_special = args.only_extra_trees or args.only_relative_lightgbm
    methods = (
        []
        if only_special
        else [
            f"ridge_{mode}_a{alpha:g}"
            for mode in feature_sets
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)
        ]
    )
    if args.extra_trees or args.only_extra_trees:
        methods.extend(
            f"extra_compact_leaf{leaf}" for leaf in (2, 5, 10)
        )
    if args.relative_lightgbm or args.only_relative_lightgbm:
        methods.append("relative_lgbm_trial4")
    prediction_parts: dict[str, list[pd.DataFrame]] = {
        method: [] for method in methods
    }
    folds = list(incumbent.groupby("fold", sort=True))
    for fold_index, (fold, fold_frame) in enumerate(folds, start=1):
        origins = pd.DatetimeIndex(sorted(fold_frame["origin"].unique()))
        train_end = origins.min() - pd.Timedelta(minutes=15)
        if str(fold).startswith("cross_month_"):
            train_start = train_end - pd.Timedelta(minutes=15 * (3840 - 1))
            train_start = max(train_start, processed.index.min())
        else:
            train_start = processed.index.min()
        last_train_origin = train_end - pd.Timedelta(minutes=15 * max(HORIZONS))
        train_index = features.loc[train_start:last_train_origin].index
        print(
            f"多输出候选折 {fold_index}/{len(folds)} {fold}: "
            f"训练 {len(train_index)} 行，验证 {len(origins)} 个起点",
            flush=True,
        )
        if not only_special:
            for mode, columns in feature_sets.items():
                train_x = features.loc[train_index, columns]
                validation_x = features.loc[origins, columns]
                for target in TARGETS:
                    targets = _future_ratios(processed, target).loc[train_index]
                    valid = np.isfinite(targets.to_numpy(dtype=float)).all(axis=1)
                    target_x = train_x.loc[valid]
                    target_y = targets.loc[valid].to_numpy(dtype=float)
                    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                        method = f"ridge_{mode}_a{alpha:g}"
                        model = make_pipeline(
                            SimpleImputer(strategy="median"),
                            StandardScaler(),
                            Ridge(alpha=alpha, solver="lsqr"),
                        )
                        model.fit(target_x, target_y)
                        ratios = np.asarray(model.predict(validation_x), dtype=float)
                        prediction_parts[method].append(
                            _prediction_rows(
                                str(fold),
                                origins,
                                target,
                                processed[target],
                                ratios,
                                method,
                            )
                        )

        if args.extra_trees or args.only_extra_trees:
            columns = feature_sets["compact"]
            imputer = SimpleImputer(strategy="median")
            train_x = imputer.fit_transform(features.loc[train_index, columns])
            validation_x = imputer.transform(features.loc[origins, columns])
            for target in TARGETS:
                targets = _future_ratios(processed, target).loc[train_index]
                valid = np.isfinite(targets.to_numpy(dtype=float)).all(axis=1)
                for leaf in (2, 5, 10):
                    method = f"extra_compact_leaf{leaf}"
                    model = ExtraTreesRegressor(
                        n_estimators=160,
                        min_samples_leaf=leaf,
                        max_features=0.70,
                        n_jobs=-1,
                        random_state=2026,
                    )
                    model.fit(
                        train_x[valid], targets.loc[valid].to_numpy(dtype=float)
                    )
                    ratios = np.asarray(model.predict(validation_x), dtype=float)
                    prediction_parts[method].append(
                        _prediction_rows(
                            str(fold),
                            origins,
                            target,
                            processed[target],
                            ratios,
                            method,
                        )
                    )

        if args.relative_lightgbm or args.only_relative_lightgbm:
            from lightgbm import LGBMRegressor

            columns = feature_sets["all"]
            for target in TARGETS:
                current = pd.to_numeric(processed[target], errors="coerce")
                for horizon in HORIZONS:
                    method = "relative_lgbm_trial4"
                    horizon_train_end = train_end - pd.Timedelta(
                        minutes=15 * horizon
                    )
                    horizon_index = features.loc[train_start:horizon_train_end].index
                    future = current.shift(-horizon).loc[horizon_index]
                    origin_level = current.loc[horizon_index]
                    valid = (
                        np.isfinite(future.to_numpy(dtype=float))
                        & np.isfinite(origin_level.to_numpy(dtype=float))
                        & origin_level.ne(0.0).to_numpy()
                    )
                    horizon_index = horizon_index[valid]
                    future = future.loc[horizon_index]
                    origin_level = origin_level.loc[horizon_index]
                    train_x = _add_target_time_features(
                        features.loc[horizon_index, columns], horizon
                    )
                    train_y = future / origin_level - 1.0
                    sample_weight = (
                        origin_level.abs()
                        / future.abs().clip(lower=1.0e-6)
                    )
                    sample_weight = sample_weight / float(sample_weight.mean())
                    model = LGBMRegressor(
                        objective="regression_l1",
                        colsample_bytree=0.8568157829250096,
                        learning_rate=0.021056202896220434,
                        max_depth=10,
                        min_child_samples=49,
                        n_estimators=150,
                        num_leaves=120,
                        reg_lambda=0.3570619074938019,
                        subsample=0.7971451699592889,
                        random_state=2026,
                        n_jobs=8,
                        verbosity=-1,
                    )
                    model.fit(train_x, train_y, sample_weight=sample_weight)
                    validation_x = _add_target_time_features(
                        features.loc[origins, columns], horizon
                    )
                    ratio = np.asarray(model.predict(validation_x), dtype=float)
                    level = current.loc[origins].to_numpy(dtype=float)
                    rows = pd.DataFrame(
                        {
                            "fold": str(fold),
                            "origin": origins,
                            "target_datetime": origins
                            + pd.Timedelta(minutes=15 * horizon),
                            "target": target,
                            "horizon_steps": horizon,
                            method: level * (1.0 + ratio),
                        }
                    )
                    prediction_parts[method].append(rows)
                print(
                    f"  {fold} {target} 相对变化率 LightGBM 完成",
                    flush=True,
                )

    combined = incumbent[[*KEYS, "y_true", "incumbent"]].copy()
    for method, parts in prediction_parts.items():
        candidate = pd.concat(parts, ignore_index=True)
        combined = combined.merge(
            candidate,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    if len(combined) != len(incumbent):
        raise RuntimeError("多输出候选未完整对齐锁定 OOF")

    for method in methods:
        for share in (0.05, 0.10, 0.20, 0.30, 0.50):
            combined[f"blend_{method}_s{share:g}"] = (
                (1.0 - share) * combined["incumbent"]
                + share * combined[method]
            )

    score_rows: list[dict[str, object]] = []
    prediction_columns = [
        column
        for column in combined.columns
        if column == "incumbent"
        or column in methods
        or column.startswith("blend_")
    ]
    split_frames = {
        "all": combined,
        "cross_month": combined.loc[
            combined["fold"].astype(str).str.startswith("cross_month_")
        ],
        "recent": combined.loc[
            combined["fold"].astype(str).str.startswith("recent_")
        ],
    }
    for method in prediction_columns:
        for split, split_frame in split_frames.items():
            score_rows.append(
                {
                    "method": method,
                    "split": split,
                    "target": "all",
                    "score": _score(split_frame, method),
                }
            )
            for target, target_frame in split_frame.groupby("target", sort=True):
                score_rows.append(
                    {
                        "method": method,
                        "split": split,
                        "target": target,
                        "score": _score(target_frame, method),
                    }
                )
    scores = pd.DataFrame(score_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output / "predictions.csv", index=False, encoding="utf-8")
    scores.to_csv(args.output / "scores.csv", index=False, encoding="utf-8")

    print("\n多输出候选全局得分：")
    print(
        scores.loc[
            scores["split"].eq("all") & scores["target"].eq("all")
        ]
        .nlargest(20, "score")
        .to_string(index=False)
    )
    print("\n跨月份与近期最佳结果：")
    for split in ("cross_month", "recent"):
        print(
            scores.loc[
                scores["split"].eq(split) & scores["target"].eq("all")
            ]
            .nlargest(10, "score")
            .to_string(index=False)
        )
    print(f"\n实验结果已保存: {args.output}")


if __name__ == "__main__":
    main()
