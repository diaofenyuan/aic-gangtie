"""在月初两日验证折上复核正式树模型候选。"""

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
from gas_power.features import LeakageError  # noqa: E402
from gas_power.models.factory import build_model  # noqa: E402
from gas_power.pipeline import prepare_prepared_data  # noqa: E402
from gas_power.tuning import candidate_config_from_settings  # noqa: E402
from gas_power.validation import TimeSplit, run_rolling_validation  # noqa: E402


KEYS = ["fold", "origin", "target_datetime", "target", "horizon_steps"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default="2026-08-02_20-02-43*")
    parser.add_argument("--trials", default="7,4,3")
    parser.add_argument("--months", default="2025-04")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "cache" / "experiments" / "month_boundary_oof",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _baseline_directory(pattern: str) -> Path:
    matches = sorted((PROJECT_ROOT / "outputs").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"基线运行目录应唯一匹配，实际找到 {len(matches)} 个: {pattern}")
    return matches[0]


def _trial_settings(baseline: Path, trial_number: int) -> dict[str, Any]:
    candidates = pd.read_csv(baseline / "reports" / "high_accuracy_candidates.csv")
    selected = candidates.loc[candidates["trial"].astype(int).eq(int(trial_number))]
    if len(selected) != 1:
        raise ValueError(f"基线候选中未唯一找到 trial {trial_number}")
    return dict(json.loads(str(selected.iloc[0]["parameters"])))


class MonthBoundarySplitter:
    """训练到月底，验证紧随其后的完整两个自然日。"""

    def __init__(self, months: list[str], *, interval_minutes: int = 15):
        self.starts = [pd.Period(value, freq="M").start_time for value in months]
        self.interval_minutes = int(interval_minutes)
        if not self.starts:
            raise ValueError("至少需要一个月初验证窗口")

    def split(self, index: pd.DatetimeIndex, max_horizon: int) -> list[TimeSplit]:
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("月初验证索引必须严格递增且唯一")
        interval = pd.Timedelta(minutes=self.interval_minutes)
        splits: list[TimeSplit] = []
        for fold, start in enumerate(self.starts):
            train_end = start - interval
            origins = pd.date_range(start, periods=192, freq=interval)
            if train_end not in index or not origins.isin(index).all():
                raise ValueError(f"数据未完整覆盖月初验证窗口: {start:%Y-%m}")
            latest_target = origins.max() + int(max_horizon) * interval
            if latest_target not in index:
                raise ValueError(f"月初验证窗口缺少最远期标签: {latest_target}")
            split = TimeSplit(
                fold=fold,
                train_start=pd.Timestamp(index.min()),
                train_end=pd.Timestamp(train_end),
                validation_origins=pd.DatetimeIndex(origins),
            )
            if split.train_end >= split.validation_origins.min():
                raise LeakageError("月初验证训练区间与验证起点重叠")
            splits.append(split)
        return splits


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["origin"] = pd.to_datetime(output["origin"], errors="raise")
    output["target_datetime"] = pd.to_datetime(
        output["target_datetime"], errors="raise"
    )
    output["fold"] = "month_" + output["origin"].dt.to_period("M").astype(str)
    output["horizon_steps"] = output["horizon_steps"].astype(int)
    return output


def _score(frame: pd.DataFrame, prediction_column: str) -> float:
    mapes: list[float] = []
    for _, group in frame.groupby(["target", "horizon_steps"], sort=True):
        actual = group["y_true"].to_numpy(dtype=float)
        predicted = group[prediction_column].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted)
        mapes.append(
            float(
                np.mean(
                    np.abs(predicted[valid] - actual[valid])
                    / np.maximum(np.abs(actual[valid]), 1.0e-6)
                )
            )
        )
    return 100.0 * (1.0 - float(np.mean(mapes)))


def main() -> None:
    args = _arguments()
    trials = [int(value) for value in args.trials.split(",") if value.strip()]
    months = [value.strip() for value in args.months.split(",") if value.strip()]
    if not trials or not months:
        raise ValueError("trials 和 months 均不能为空")

    baseline = _baseline_directory(args.baseline_run)
    config = load_config(PROJECT_ROOT / "config" / "official_preliminary.yaml")
    args.output.mkdir(parents=True, exist_ok=True)
    prepared, features, _ = prepare_prepared_data(config)
    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    horizons = list(range(1, int(config.section("forecast")["short_steps"]) + 1))
    validation = config.section("validation")
    splitter = MonthBoundarySplitter(months)

    combined: pd.DataFrame | None = None
    prediction_columns: list[str] = []
    for trial_index, trial in enumerate(trials, start=1):
        path = args.output / f"trial_{trial}_predictions.csv"
        if path.exists() and not args.force:
            predictions = pd.read_csv(path, encoding="utf-8")
            print(f"复用 trial {trial} 月初 OOF: {path}", flush=True)
        else:
            settings = _trial_settings(baseline, trial)
            candidate_config = candidate_config_from_settings(config, **settings)
            artifacts = run_rolling_validation(
                frame=prepared.model_input,
                model_factory=lambda item=candidate_config: build_model(item),
                splitter=splitter,
                target_columns=targets,
                horizons=horizons,
                interval_minutes=15,
                near_zero_threshold=float(validation.get("near_zero_threshold", 1.0e-6)),
                worst_error_count=50,
                raw_targets=prepared.raw_targets,
                feature_matrix=features,
                data_source=prepared.source,
                show_progress=True,
                progress_description=(
                    f"月初 OOF 候选 {trial_index}/{len(trials)}，trial {trial}"
                ),
            )
            predictions = _normalize(artifacts.predictions)
            predictions.to_csv(path, index=False, encoding="utf-8")
        predictions = _normalize(predictions)
        column = f"trial_{trial}"
        prediction_columns.append(column)
        candidate = predictions[[*KEYS, "y_true", "y_pred"]].rename(
            columns={"y_pred": column}
        )
        if combined is None:
            combined = candidate
        else:
            combined = combined.merge(
                candidate.drop(columns="y_true"),
                on=KEYS,
                how="inner",
                validate="one_to_one",
            )

    if combined is None:
        raise RuntimeError("没有生成月初 OOF")
    rows: list[dict[str, object]] = []
    for method in prediction_columns:
        for target, group in [("all", combined), *combined.groupby("target", sort=True)]:
            rows.append(
                {
                    "method": method,
                    "target": target,
                    "score": _score(group, method),
                    "folds": group["fold"].nunique(),
                    "rows": len(group),
                }
            )
    report = pd.DataFrame(rows).sort_values(
        ["target", "score"], ascending=[True, False]
    )
    combined.to_csv(args.output / "predictions.csv", index=False, encoding="utf-8")
    report.to_csv(args.output / "scores.csv", index=False, encoding="utf-8")
    print("\n月初两日 OOF 得分：")
    print(report.to_string(index=False))
    print(f"\n实验结果已保存: {args.output}")


if __name__ == "__main__":
    main()
