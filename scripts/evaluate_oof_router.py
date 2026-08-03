"""在既有训练期 OOF 上快速评估逐目标、逐时距的动态路由策略。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    "last_value",
    "damped_trend",
    "daily_naive",
    "daily_median_3",
    "candidate_1_trial_7",
    "candidate_2_trial_4",
    "candidate_3_trial_3",
)
FUSED_CANDIDATE = "incumbent_fused_loo"


@dataclass(frozen=True)
class RouterResult:
    """单个路由器的交叉拟合预测和可部署全量模型描述。"""

    predictions: pd.Series
    recipes: dict[str, object]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default="2026-08-02_14-13-16*")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "cache" / "experiments" / "oof_router",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--methods",
        help="仅运行逗号分隔的方法名；默认运行全部方法",
    )
    return parser.parse_args()


def _baseline_directory(pattern: str) -> Path:
    matches = sorted((PROJECT_ROOT / "outputs").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"基线运行目录应唯一匹配，实际找到 {len(matches)} 个: {pattern}")
    return matches[0]


def _score(frame: pd.DataFrame, prediction: pd.Series) -> float:
    values = frame[["target", "horizon_steps", "y_true"]].copy()
    values["y_pred"] = prediction.to_numpy(dtype=float)
    rows: list[float] = []
    for _, group in values.groupby(["target", "horizon_steps"], sort=True):
        actual = group["y_true"].to_numpy(dtype=float)
        predicted = group["y_pred"].to_numpy(dtype=float)
        finite = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[finite]
        predicted = predicted[finite]
        if not len(actual):
            continue
        rows.append(
            float(
                np.mean(
                    np.abs(predicted - actual)
                    / np.maximum(np.abs(actual), 1.0e-6)
                )
            )
        )
    if not rows:
        raise ValueError("没有可计算路由得分的有限标签")
    return 100.0 * (1.0 - float(np.mean(rows)))


def _candidate_errors(frame: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    actual = frame["y_true"].to_numpy(dtype=float)
    denominator = np.maximum(np.abs(actual), 1.0e-6)
    matrix = frame[candidates].to_numpy(dtype=float)
    return np.abs(matrix - actual[:, None]) / denominator[:, None]


def _load_origin_features(frame: pd.DataFrame) -> pd.DataFrame:
    """从正式特征缓存中恢复每个 OOF 预测起点可见的因果特征。"""

    path = PROJECT_ROOT / "cache" / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少正式特征缓存: {path}")
    columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    direct_prefixes = (
        "feat_time_",
        "feat_current_generator_",
        "feat_diff_generator_",
        "feat_accel_generator_",
        "feat_ewma_generator_",
        "feat_missing__",
        "feat_outlier__",
        "feat_gas_balance_",
        "feat_holder_",
        "feat_state_",
    )
    lag_prefixes = (
        "feat_lag_generator_1_",
        "feat_lag_generator_all_",
        "feat_lag_blast_furnace_1_",
        "feat_lag_coke_oven_1_",
        "feat_lag_converter_1_",
    )
    roll_prefixes = (
        "feat_roll_generator_1_",
        "feat_roll_generator_all_",
        "feat_roll_blast_furnace_1_",
        "feat_roll_coke_oven_1_",
        "feat_roll_converter_1_",
    )
    lag_suffixes = tuple(f"_{step}" for step in (1, 4, 8, 16, 32, 96))
    roll_suffixes = tuple(
        f"_{window}_{stat}"
        for window in (4, 8, 16, 32, 96)
        for stat in ("mean", "std", "slope")
    )
    selected = [
        column
        for column in columns
        if column == "datetime"
        or column.startswith(direct_prefixes)
        or (column.startswith(lag_prefixes) and column.endswith(lag_suffixes))
        or (column.startswith(roll_prefixes) and column.endswith(roll_suffixes))
    ]
    cached = pd.read_csv(path, usecols=selected, encoding="utf-8")
    cached["datetime"] = pd.to_datetime(cached["datetime"], errors="raise")
    cached = cached.set_index("datetime")
    aligned = cached.reindex(pd.to_datetime(frame["origin"], errors="raise"))
    aligned.index = frame.index
    return aligned


def _router_features(frame: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """只构造预测时可获得的时间、工况和候选分歧特征。"""

    origin = pd.to_datetime(frame["origin"], errors="raise")
    target_time = pd.to_datetime(frame["target_datetime"], errors="raise")
    output = pd.DataFrame(index=frame.index)
    for prefix, timestamps in (("origin", origin), ("target", target_time)):
        minute = timestamps.dt.hour * 60 + timestamps.dt.minute
        output[f"{prefix}_day_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
        output[f"{prefix}_day_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
        output[f"{prefix}_week_sin"] = np.sin(
            2.0 * np.pi * timestamps.dt.dayofweek / 7.0
        )
        output[f"{prefix}_week_cos"] = np.cos(
            2.0 * np.pi * timestamps.dt.dayofweek / 7.0
        )
    prediction_matrix = frame[candidates].to_numpy(dtype=float)
    scale = np.maximum(np.abs(frame["last_value"].to_numpy(dtype=float)), 1.0)
    for index, candidate in enumerate(candidates):
        output[f"pred_{candidate}"] = prediction_matrix[:, index] / scale
        output[f"delta_{candidate}"] = (
            prediction_matrix[:, index] - prediction_matrix[:, 0]
        ) / scale
    output["candidate_std"] = np.std(prediction_matrix, axis=1) / scale
    output["candidate_range"] = (
        np.max(prediction_matrix, axis=1) - np.min(prediction_matrix, axis=1)
    ) / scale
    output["candidate_direction"] = (
        np.median(prediction_matrix, axis=1) - prediction_matrix[:, 0]
    ) / scale

    # 目标自身状态优先，煤气平衡和柜位特征作为共享工况信号。
    common_prefixes = ("feat_gas_", "feat_holder_", "feat_missing__")
    for column in frame.columns:
        name = str(column)
        if name.startswith(("feat_state_", *common_prefixes)):
            output[name] = pd.to_numeric(frame[column], errors="coerce")
    origin_features = _load_origin_features(frame)
    new_columns = origin_features.columns.difference(output.columns, sort=False)
    output = pd.concat([output, origin_features[new_columns]], axis=1)
    return output.replace([np.inf, -np.inf], np.nan)


def _impute(train: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    medians = train.median(axis=0, numeric_only=True).fillna(0.0)
    return (
        train.fillna(medians).to_numpy(dtype=np.float32),
        held.fillna(medians).to_numpy(dtype=np.float32),
    )


def _fold_groups(frame: pd.DataFrame):
    return frame.groupby("fold", sort=True)


def _fit_error_lgbm(
    train: pd.DataFrame,
    held: pd.DataFrame,
    features: pd.DataFrame,
    candidates: list[str],
    *,
    seed: int,
    num_leaves: int,
    min_child_samples: int,
    model_share: float,
    temperature: float,
    probability_blend: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    from lightgbm import LGBMRegressor

    train_x, held_x = _impute(features.loc[train.index], features.loc[held.index])
    train_errors = _candidate_errors(train, candidates)
    expected = np.empty((len(held), len(candidates)), dtype=float)
    priors = np.mean(train_errors, axis=0)
    for candidate_index in range(len(candidates)):
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=80,
            learning_rate=0.035,
            num_leaves=int(num_leaves),
            max_depth=3,
            min_child_samples=int(min_child_samples),
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=seed + candidate_index,
            n_jobs=1,
            verbosity=-1,
        )
        target = np.log1p(50.0 * train_errors[:, candidate_index])
        model.fit(train_x, target)
        modeled = np.expm1(model.predict(held_x)) / 50.0
        expected[:, candidate_index] = (
            float(model_share) * np.maximum(modeled, 0.0)
            + (1.0 - float(model_share)) * priors[candidate_index]
        )
    matrix = held[candidates].to_numpy(dtype=float)
    if probability_blend:
        logits = -expected / max(float(temperature), 1.0e-6)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        prediction = np.sum(matrix * weights, axis=1)
    else:
        winners = np.argmin(expected, axis=1)
        prediction = np.take_along_axis(matrix, winners[:, None], axis=1)[:, 0]
    return prediction, {
        "priors": priors.tolist(),
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "model_share": model_share,
        "temperature": temperature,
        "probability_blend": probability_blend,
    }


def _fit_winner_lgbm(
    train: pd.DataFrame,
    held: pd.DataFrame,
    features: pd.DataFrame,
    candidates: list[str],
    *,
    seed: int,
    probability_blend: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    from lightgbm import LGBMClassifier

    train_x, held_x = _impute(features.loc[train.index], features.loc[held.index])
    errors = _candidate_errors(train, candidates)
    ordered = np.sort(errors, axis=1)
    labels = np.argmin(errors, axis=1)
    margin = np.maximum(ordered[:, 1] - ordered[:, 0], 1.0e-4)
    sample_weight = np.clip(margin / np.median(margin), 0.25, 8.0)
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=100,
        learning_rate=0.035,
        num_leaves=7,
        max_depth=3,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(train_x, labels, sample_weight=sample_weight)
    probabilities = model.predict_proba(held_x)
    aligned = np.zeros((len(held), len(candidates)), dtype=float)
    aligned[:, np.asarray(model.classes_, dtype=int)] = probabilities
    matrix = held[candidates].to_numpy(dtype=float)
    if probability_blend:
        prediction = np.sum(matrix * aligned, axis=1)
    else:
        winners = np.argmax(aligned, axis=1)
        prediction = np.take_along_axis(matrix, winners[:, None], axis=1)[:, 0]
    return prediction, {"probability_blend": probability_blend}


def _fit_ridge_stack(
    train: pd.DataFrame,
    held: pd.DataFrame,
    features: pd.DataFrame,
    candidates: list[str],
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, object]]:
    from sklearn.linear_model import Ridge

    # 只给线性堆叠候选预测和周期项，防止高维工况特征吞掉小样本。
    selected_columns = [
        column
        for column in features.columns
        if column.startswith(("pred_", "delta_", "origin_", "target_"))
    ]
    train_x, held_x = _impute(
        features.loc[train.index, selected_columns],
        features.loc[held.index, selected_columns],
    )
    actual = train["y_true"].to_numpy(dtype=float)
    train_anchor = np.maximum(
        np.abs(train["last_value"].to_numpy(dtype=float)), 1.0
    )
    held_anchor = np.maximum(
        np.abs(held["last_value"].to_numpy(dtype=float)), 1.0
    )
    target_ratio = actual / train_anchor
    sample_weight = train_anchor / np.maximum(np.abs(actual), 1.0)
    sample_weight /= np.mean(sample_weight)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(train_x, target_ratio, sample_weight=sample_weight)
    return model.predict(held_x) * held_anchor, {
        "alpha": alpha,
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "feature_columns": selected_columns,
    }


def _fit_residual_lgbm(
    train: pd.DataFrame,
    held: pd.DataFrame,
    features: pd.DataFrame,
    *,
    seed: int,
    model_share: float,
    num_leaves: int = 7,
    min_child_samples: int = 60,
) -> tuple[np.ndarray, dict[str, object]]:
    """围绕现有留一折融合学习相对残差，限制二层模型的偏移幅度。"""

    from lightgbm import LGBMRegressor

    train_x, held_x = _impute(features.loc[train.index], features.loc[held.index])
    train_anchor = np.maximum(
        np.abs(train[FUSED_CANDIDATE].to_numpy(dtype=float)), 1.0
    )
    held_anchor = np.maximum(
        np.abs(held[FUSED_CANDIDATE].to_numpy(dtype=float)), 1.0
    )
    actual = train["y_true"].to_numpy(dtype=float)
    ratio = actual / train_anchor
    sample_weight = train_anchor / np.maximum(np.abs(actual), 1.0)
    sample_weight /= np.mean(sample_weight)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=120,
        learning_rate=0.025,
        num_leaves=int(num_leaves),
        max_depth=3,
        min_child_samples=int(min_child_samples),
        subsample=0.85,
        colsample_bytree=0.55,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(train_x, ratio, sample_weight=sample_weight)
    modeled_ratio = np.clip(model.predict(held_x), 0.70, 1.30)
    corrected_ratio = 1.0 + float(model_share) * (modeled_ratio - 1.0)
    return held_anchor * corrected_ratio, {
        "model_share": model_share,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
    }


def _cross_fit(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    candidates: list[str],
    name: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, dict[str, object]]],
) -> RouterResult:
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    recipes: dict[str, object] = {}
    combinations = list(frame.groupby(["target", "horizon_steps"], sort=True))
    for combination_index, ((target, horizon), subset) in enumerate(combinations, start=1):
        print(
            f"[{name}] {combination_index}/{len(combinations)} "
            f"{target} t+{int(horizon) * 15} 分钟",
            flush=True,
        )
        for fold, held in _fold_groups(subset):
            train = subset.loc[
                subset["fold"].ne(fold) & np.isfinite(subset["y_true"])
            ]
            values, recipe = fit_predict(train, held)
            prediction.loc[held.index] = values
            recipes[f"{target}|{int(horizon)}|{fold}"] = recipe
    if prediction.isna().any() or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError(f"{name} 的交叉拟合预测未完整覆盖 OOF")
    return RouterResult(predictions=prediction, recipes=recipes)


def _oracle(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    errors = _candidate_errors(frame, candidates)
    winners = np.argmin(errors, axis=1)
    matrix = frame[candidates].to_numpy(dtype=float)
    values = np.take_along_axis(matrix, winners[:, None], axis=1)[:, 0]
    return pd.Series(values, index=frame.index, dtype=float)


def main() -> None:
    args = _arguments()
    baseline = _baseline_directory(args.baseline_run)
    oof_path = baseline / "reports" / "high_accuracy_oof.csv"
    oof = pd.read_csv(oof_path, encoding="utf-8")
    keys = ["fold", "origin", "target_datetime", "target", "horizon_steps"]
    fused_path = baseline / "reports" / "high_accuracy_fused_leave_one_fold.csv"
    fused = pd.read_csv(fused_path, encoding="utf-8")
    oof = oof.merge(
        fused[keys + ["y_pred"]].rename(columns={"y_pred": FUSED_CANDIDATE}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    candidates = [name for name in DEFAULT_CANDIDATES if name in oof.columns]
    if len(candidates) != len(DEFAULT_CANDIDATES):
        missing = sorted(set(DEFAULT_CANDIDATES).difference(candidates))
        raise ValueError(f"OOF 缺少路由候选: {missing}")
    candidates.append(FUSED_CANDIDATE)
    features = _router_features(oof, candidates)

    methods: dict[str, Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, dict[str, object]]]] = {
        "error_lgbm_hard": lambda train, held: _fit_error_lgbm(
            train,
            held,
            features,
            candidates,
            seed=args.seed,
            num_leaves=7,
            min_child_samples=60,
            model_share=0.50,
            temperature=0.01,
            probability_blend=False,
        ),
        "error_lgbm_soft_005": lambda train, held: _fit_error_lgbm(
            train,
            held,
            features,
            candidates,
            seed=args.seed,
            num_leaves=7,
            min_child_samples=60,
            model_share=0.50,
            temperature=0.005,
            probability_blend=True,
        ),
        "error_lgbm_soft_010": lambda train, held: _fit_error_lgbm(
            train,
            held,
            features,
            candidates,
            seed=args.seed,
            num_leaves=7,
            min_child_samples=60,
            model_share=0.50,
            temperature=0.010,
            probability_blend=True,
        ),
        "winner_lgbm_hard": lambda train, held: _fit_winner_lgbm(
            train,
            held,
            features,
            candidates,
            seed=args.seed,
            probability_blend=False,
        ),
        "winner_lgbm_soft": lambda train, held: _fit_winner_lgbm(
            train,
            held,
            features,
            candidates,
            seed=args.seed,
            probability_blend=True,
        ),
        "ridge_1": lambda train, held: _fit_ridge_stack(
            train, held, features, candidates, alpha=1.0
        ),
        "ridge_10": lambda train, held: _fit_ridge_stack(
            train, held, features, candidates, alpha=10.0
        ),
        "ridge_100": lambda train, held: _fit_ridge_stack(
            train, held, features, candidates, alpha=100.0
        ),
        "residual_lgbm_025": lambda train, held: _fit_residual_lgbm(
            train, held, features, seed=args.seed, model_share=0.25
        ),
        "residual_lgbm_050": lambda train, held: _fit_residual_lgbm(
            train, held, features, seed=args.seed, model_share=0.50
        ),
        "residual_lgbm_075": lambda train, held: _fit_residual_lgbm(
            train, held, features, seed=args.seed, model_share=0.75
        ),
        "residual_lgbm_100": lambda train, held: _fit_residual_lgbm(
            train, held, features, seed=args.seed, model_share=1.00
        ),
    }
    if args.methods:
        selected_names = [value.strip() for value in args.methods.split(",") if value.strip()]
        unknown = sorted(set(selected_names).difference(methods))
        if unknown:
            raise ValueError(f"未知路由方法: {unknown}")
        methods = {name: methods[name] for name in selected_names}

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    predictions = oof[
        ["fold", "origin", "target_datetime", "target", "horizon_steps", "y_true"]
    ].copy()
    report_rows: list[dict[str, object]] = []

    references = {
        **{candidate: pd.Series(oof[candidate], index=oof.index) for candidate in candidates},
        "oracle": _oracle(oof, candidates),
    }
    all_predictions: dict[str, pd.Series] = dict(references)
    recipes: dict[str, object] = {}
    for name, method in methods.items():
        result = _cross_fit(oof, features, candidates, name, method)
        all_predictions[name] = result.predictions
        recipes[name] = result.recipes

    for name, prediction in all_predictions.items():
        predictions[name] = prediction.to_numpy(dtype=float)
        for split, mask in (
            ("all", pd.Series(True, index=oof.index)),
            ("recent", oof["fold"].astype(str).str.startswith("recent_")),
            (
                "cross_month",
                oof["fold"].astype(str).str.startswith("cross_month_"),
            ),
        ):
            report_rows.append(
                {
                    "method": name,
                    "split": split,
                    "score": _score(oof.loc[mask], prediction.loc[mask]),
                    "rows": int(mask.sum()),
                }
            )

    report = pd.DataFrame(report_rows).sort_values(
        ["split", "score"], ascending=[True, False]
    )
    report.to_csv(output / "scores.csv", index=False, encoding="utf-8")
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    (output / "recipes.json").write_text(
        json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n逐路由跨折得分：")
    print(report.to_string(index=False))
    print(f"\n实验结果已保存到: {output}")


if __name__ == "__main__":
    main()
