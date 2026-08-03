"""复核树模型参数化或时序 Mixup 候选，并将每个阶段的预测立即落盘。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gas_power.config import load_config  # noqa: E402
from gas_power.models.factory import build_model  # noqa: E402
from gas_power.pipeline import prepare_prepared_data  # noqa: E402
from gas_power.tuning import candidate_config_from_settings  # noqa: E402
from gas_power.validation import (  # noqa: E402
    RecentWindowSplitter,
    run_rolling_validation,
    splitter_from_config,
)


KEY_COLUMNS = ["fold", "origin", "target_datetime", "target", "horizon_steps"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=float, default=0.0)
    parser.add_argument(
        "--parameterization",
        choices=("direct", "component", "gas_availability"),
        help="覆盖基线候选的目标参数化方式",
    )
    parser.add_argument(
        "--strategy",
        choices=("direct", "global"),
        help="覆盖基线候选的多步训练策略",
    )
    parser.add_argument("--baseline-run", default="2026-08-02_20-02-43*")
    parser.add_argument("--trial", type=int, default=7)
    parser.add_argument(
        "--training-window-days",
        help="覆盖基线候选训练窗口；使用 all 表示全部历史",
    )
    parser.add_argument("--recent-folds", type=int, default=3)
    parser.add_argument("--cross-month-folds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare-only", action="store_true")
    return parser.parse_args()


def _baseline_directory(project_root: Path, pattern: str) -> Path:
    matches = sorted((project_root / "outputs").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"基线运行目录应唯一匹配，实际找到 {len(matches)} 个: {pattern}")
    return matches[0]


def _trial_settings(baseline: Path, trial_number: int) -> dict[str, Any]:
    candidates = pd.read_csv(baseline / "reports" / "high_accuracy_candidates.csv")
    selected = candidates.loc[candidates["trial"].astype(int).eq(int(trial_number))]
    if len(selected) != 1:
        raise ValueError(f"基线候选中未唯一找到 trial {trial_number}")
    return dict(json.loads(str(selected.iloc[0]["parameters"])))


def _prefixed(predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    output = predictions.copy()
    output["fold"] = output["fold"].map(lambda value: f"{prefix}_{int(value)}")
    return output


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["fold"] = output["fold"].astype(str)
    output["origin"] = pd.to_datetime(output["origin"], errors="raise")
    output["target_datetime"] = pd.to_datetime(
        output["target_datetime"], errors="raise"
    )
    output["target"] = output["target"].astype(str)
    output["horizon_steps"] = output["horizon_steps"].astype(int)
    return output


def _mape(frame: pd.DataFrame, prediction_column: str) -> float:
    actual = frame["y_true"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        raise ValueError(f"{prediction_column} 没有可计算 MAPE 的有效样本")
    actual = actual[valid]
    predicted = predicted[valid]
    denominator = np.maximum(np.abs(actual), 1.0e-6)
    return float(np.mean(np.abs(predicted - actual) / denominator))


def _comparison(
    baseline: Path,
    candidate: pd.DataFrame,
    trial_number: int,
) -> pd.DataFrame:
    oof_path = baseline / "reports" / "high_accuracy_oof.csv"
    columns = pd.read_csv(oof_path, nrows=0).columns
    incumbent_columns = [
        str(column) for column in columns if str(column).endswith(f"_trial_{trial_number}")
    ]
    if len(incumbent_columns) != 1:
        raise ValueError(
            f"基线 OOF 中未唯一找到 trial {trial_number}: {incumbent_columns}"
        )
    incumbent_column = incumbent_columns[0]
    incumbent = pd.read_csv(
        oof_path,
        usecols=[*KEY_COLUMNS, "y_true", incumbent_column],
    ).rename(columns={incumbent_column: "incumbent_prediction"})
    challenger = candidate[[*KEY_COLUMNS, "y_true", "y_pred"]].rename(
        columns={"y_pred": "mixup_prediction"}
    )
    incumbent = _normalize_keys(incumbent)
    challenger = _normalize_keys(challenger)
    # 单折 RecentWindowSplitter 会把最新窗口重新编号为 0，不能依赖折号对齐。
    # 验证类型与四个时间语义键可唯一定位相同的 OOF 样本。
    incumbent["validation_type"] = incumbent["fold"].str.rsplit("_", n=1).str[0]
    challenger["validation_type"] = challenger["fold"].str.rsplit("_", n=1).str[0]
    semantic_keys = ["validation_type", *KEY_COLUMNS[1:]]
    joined = incumbent.merge(
        challenger,
        on=semantic_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_incumbent", "_mixup"),
    )
    if len(joined) != len(challenger):
        raise ValueError(
            f"候选 OOF 未完整对齐基线: challenger={len(challenger)}, joined={len(joined)}"
        )
    if not np.allclose(
        joined["y_true_incumbent"],
        joined["y_true_mixup"],
        equal_nan=True,
    ):
        raise ValueError("候选与基线 OOF 的真实标签不一致")
    joined["y_true"] = joined.pop("y_true_mixup")
    joined = joined.drop(columns="y_true_incumbent")
    joined["fold"] = joined.pop("fold_incumbent")
    joined = joined.drop(columns="fold_mixup")

    rows: list[dict[str, Any]] = []
    for (validation_type, target), group in joined.groupby(
        ["validation_type", "target"], sort=True
    ):
        incumbent_mape = _mape(group, "incumbent_prediction")
        mixup_mape = _mape(group, "mixup_prediction")
        rows.append(
            {
                "validation_type": validation_type,
                "target": target,
                "incumbent_mape": incumbent_mape,
                "mixup_mape": mixup_mape,
                "mape_gain": incumbent_mape - mixup_mape,
                "score_gain": 100.0 * (incumbent_mape - mixup_mape),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _arguments()
    project_root = PROJECT_ROOT
    config = load_config(project_root / "config" / "official_preliminary.yaml")
    baseline = _baseline_directory(project_root, args.baseline_run)
    output = args.output or (
        project_root / "cache" / "experiments" / f"temporal_mixup_{args.ratio:.2f}"
    )
    output.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        recent_predictions = pd.read_csv(output / "recent_predictions.csv")
        cross_predictions = pd.read_csv(output / "cross_month_predictions.csv")
        candidate = pd.concat([recent_predictions, cross_predictions], ignore_index=True)
        report = _comparison(baseline, candidate, args.trial)
        report.to_csv(output / "comparison.csv", index=False)
        print(report.to_string(index=False))
        print(f"消融结果已保存: {output}")
        return

    prepared, features, _ = prepare_prepared_data(config)
    settings = _trial_settings(baseline, args.trial)
    if args.parameterization is not None:
        settings["parameterization"] = str(args.parameterization)
    if args.strategy is not None:
        settings["strategy"] = str(args.strategy)
    if args.training_window_days is not None:
        value = str(args.training_window_days).strip().lower()
        settings["training_window_days"] = (
            None if value in {"all", "none", "null"} else int(value)
        )
        window_label = "全历史" if settings["training_window_days"] is None else f"{value} 天"
        experiment_label = f"trial {args.trial} {window_label}"
    else:
        experiment_label = (
            f"{args.parameterization or settings.get('parameterization', 'direct')} 参数化，"
            f"{args.strategy or settings.get('strategy', 'direct')} 策略，Mixup {args.ratio:.0%}"
        )
    settings.update(
        {
            "temporal_mixup_ratio": float(args.ratio),
            "temporal_mixup_min_lambda": 0.70,
            "temporal_mixup_max_lambda": 0.90,
        }
    )
    candidate_config = candidate_config_from_settings(config, **settings)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    horizons = list(range(1, int(config.section("forecast")["short_steps"]) + 1))
    validation = config.section("validation")
    recent_config = config.raw.get("recent_validation", {})

    recent_splitter = RecentWindowSplitter(
        folds=int(args.recent_folds),
        validation_points=int(recent_config.get("validation_points", 192)),
        step_points=int(recent_config.get("step_points", 192)),
        rolling_train_points=(
            int(recent_config["rolling_train_points"])
            if recent_config.get("rolling_train_points") is not None
            else None
        ),
    )
    recent = run_rolling_validation(
        frame=prepared.model_input,
        model_factory=lambda: build_model(candidate_config),
        splitter=recent_splitter,
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=50,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
        show_progress=True,
        progress_description=f"{experiment_label} 近期窗口消融",
    )
    recent_predictions = _prefixed(recent.predictions, "recent")
    recent_predictions.to_csv(output / "recent_predictions.csv", index=False)

    cross_validation = dict(validation)
    cross_validation["folds"] = int(args.cross_month_folds)
    cross = run_rolling_validation(
        frame=prepared.model_input,
        model_factory=lambda: build_model(candidate_config),
        splitter=splitter_from_config(cross_validation),
        target_columns=targets,
        horizons=horizons,
        interval_minutes=15,
        near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
        worst_error_count=50,
        raw_targets=prepared.raw_targets,
        feature_matrix=features,
        data_source=prepared.source,
        show_progress=True,
        progress_description=f"{experiment_label} 跨月份消融",
    )
    cross_predictions = _prefixed(cross.predictions, "cross_month")
    cross_predictions.to_csv(output / "cross_month_predictions.csv", index=False)

    candidate = pd.concat([recent_predictions, cross_predictions], ignore_index=True)
    report = _comparison(baseline, candidate, args.trial)
    report.to_csv(output / "comparison.csv", index=False)
    print(report.to_string(index=False))
    print(f"消融结果已保存: {output}")


if __name__ == "__main__":
    main()
