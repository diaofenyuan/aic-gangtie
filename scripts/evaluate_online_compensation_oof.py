"""在训练期 OOF 上评估严格因果的实时预测误差补偿。"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["fold", "origin", "target_datetime", "target", "horizon_steps"]


@dataclass(frozen=True)
class CompensationSettings:
    """单个在线补偿候选的完整可复现参数。"""

    signal: str
    aggregation: str
    window: int
    share: float

    @property
    def name(self) -> str:
        return (
            f"{self.signal}_{self.aggregation}_w{self.window}"
            f"_s{self.share:g}"
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", default="2026-08-02_20-02-43*")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "cache"
        / "experiments"
        / "online_compensation_oof",
    )
    return parser.parse_args()


def _baseline_directory(pattern: str) -> Path:
    matches = sorted((PROJECT_ROOT / "outputs").glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"基线运行目录应唯一匹配，实际找到 {len(matches)} 个: {pattern}"
        )
    return matches[0]


def _mape(frame: pd.DataFrame, prediction: pd.Series) -> float:
    actual = frame["y_true"].to_numpy(dtype=float)
    predicted = prediction.reindex(frame.index).to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    if not finite.any():
        return float("inf")
    return float(
        np.mean(
            np.abs(predicted[finite] - actual[finite])
            / np.maximum(np.abs(actual[finite]), 1.0e-6)
        )
    )


def _competition_score(frame: pd.DataFrame, prediction: pd.Series) -> float:
    mapes = [
        _mape(group, prediction)
        for _, group in frame.groupby(["target", "horizon_steps"], sort=True)
    ]
    return 100.0 * (1.0 - float(np.mean(mapes)))


def _center(values: deque[float], aggregation: str) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    if aggregation == "median":
        return float(np.median(array))
    if aggregation == "mean":
        return float(np.mean(array))
    raise ValueError(f"未知聚合方法: {aggregation}")


def _simulate_group(
    frame: pd.DataFrame,
    settings: CompensationSettings,
) -> pd.Series:
    """逐起点回放；当前起点只能使用已经到达的真实观测。"""

    ordered = frame.sort_values(["origin", "horizon_steps"])
    output = pd.Series(np.nan, index=ordered.index, dtype=float)
    shared_absolute: deque[float] = deque(maxlen=settings.window)
    shared_relative: deque[float] = deque(maxlen=settings.window)
    horizon_absolute = {
        horizon: deque(maxlen=settings.window) for horizon in range(1, 9)
    }
    horizon_relative = {
        horizon: deque(maxlen=settings.window) for horizon in range(1, 9)
    }

    for origin in sorted(ordered["origin"].unique()):
        # target_datetime == origin 的误差在当前时刻已经兑现，属于可见历史。
        matured = ordered.loc[
            ordered["target_datetime"].eq(origin)
            & np.isfinite(ordered["y_true"].to_numpy(dtype=float))
        ]
        if not matured.empty:
            actual = float(matured["y_true"].iloc[0])
            absolute_values: list[float] = []
            relative_values: list[float] = []
            for row in matured.itertuples():
                residual = actual - float(row.base_prediction)
                relative = residual / max(abs(float(row.base_prediction)), 1.0)
                horizon = int(row.horizon_steps)
                horizon_absolute[horizon].append(residual)
                horizon_relative[horizon].append(relative)
                absolute_values.append(residual)
                relative_values.append(relative)
            shared_absolute.append(float(np.median(absolute_values)))
            shared_relative.append(float(np.median(relative_values)))

        current = ordered.loc[ordered["origin"].eq(origin)]
        shared_abs = _center(shared_absolute, settings.aggregation)
        shared_rel = _center(shared_relative, settings.aggregation)
        for row in current.itertuples():
            horizon = int(row.horizon_steps)
            base = float(row.base_prediction)
            horizon_abs = _center(
                horizon_absolute[horizon], settings.aggregation
            )
            horizon_rel = _center(
                horizon_relative[horizon], settings.aggregation
            )
            if settings.signal == "shared_absolute":
                correction = shared_abs
            elif settings.signal == "shared_relative":
                correction = base * shared_rel
            elif settings.signal == "horizon_absolute":
                correction = horizon_abs
            elif settings.signal == "horizon_relative":
                correction = base * horizon_rel
            elif settings.signal == "blended_absolute":
                correction = 0.5 * (shared_abs + horizon_abs)
            elif settings.signal == "blended_relative":
                correction = 0.5 * base * (shared_rel + horizon_rel)
            elif settings.signal == "scaled_shared_absolute":
                correction = shared_abs * horizon / 8.0
            else:
                raise ValueError(f"未知在线误差信号: {settings.signal}")
            output.loc[row.Index] = base + settings.share * correction

    if output.isna().any() or not np.isfinite(output.to_numpy(dtype=float)).all():
        raise RuntimeError(f"在线补偿未完整覆盖 OOF: {settings.name}")
    return output


def _candidate_grid() -> list[CompensationSettings]:
    return [
        CompensationSettings(signal, aggregation, window, share)
        for signal in (
            "shared_absolute",
            "shared_relative",
            "horizon_absolute",
            "horizon_relative",
            "blended_absolute",
            "blended_relative",
            "scaled_shared_absolute",
        )
        for aggregation in ("median", "mean")
        for window in (1, 2, 4, 8, 16, 32, 64)
        for share in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
    ]


def _select_cross_month_recipe(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    held_fold: str | None,
    per_column: bool,
) -> tuple[pd.Series, dict[str, str]]:
    """仅用其他跨月份折选参；近期折永不参与候选选择。"""

    cross_mask = frame["fold"].astype(str).str.startswith("cross_month_")
    selection = frame.loc[cross_mask]
    if held_fold is not None:
        selection = selection.loc[selection["fold"].ne(held_fold)]
    candidates = predictions.columns.astype(str).tolist()
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    recipe: dict[str, str] = {}
    groups = (
        frame.groupby(["target", "horizon_steps"], sort=True)
        if per_column
        else [("all", frame)]
    )
    for key, deployment in groups:
        if per_column:
            target, horizon = key
            selected_rows = selection.loc[
                selection["target"].eq(target)
                & selection["horizon_steps"].eq(horizon)
            ]
            recipe_key = f"{target}|{int(horizon)}"
        else:
            selected_rows = selection
            recipe_key = "all"
        losses = {
            candidate: _competition_score(
                selected_rows, predictions[candidate]
            )
            for candidate in candidates
        }
        winner = max(losses, key=losses.get)
        output.loc[deployment.index] = predictions.loc[deployment.index, winner]
        recipe[recipe_key] = winner
    return output, recipe


def main() -> None:
    args = _arguments()
    baseline = _baseline_directory(args.baseline_run)
    frame = pd.read_csv(
        baseline / "reports" / "high_accuracy_fused_leave_one_fold.csv",
        encoding="utf-8",
    ).rename(columns={"y_pred": "base_prediction"})
    frame["origin"] = pd.to_datetime(frame["origin"], errors="raise")
    frame["target_datetime"] = pd.to_datetime(
        frame["target_datetime"], errors="raise"
    )
    prediction_columns: dict[str, np.ndarray] = {
        "base_prediction": frame["base_prediction"].to_numpy(dtype=float)
    }

    grid = _candidate_grid()
    signal_grid = list(
        dict.fromkeys(
            (settings.signal, settings.aggregation, settings.window)
            for settings in grid
        )
    )
    groups = list(frame.groupby(["fold", "target"], sort=True))
    for index, (signal, aggregation, window) in enumerate(signal_grid, start=1):
        settings = CompensationSettings(signal, aggregation, window, 1.0)
        if index == 1 or index % 10 == 0 or index == len(signal_grid):
            print(
                f"在线误差信号: {index}/{len(signal_grid)} {settings.name}",
                flush=True,
            )
        unit_prediction = pd.Series(np.nan, index=frame.index, dtype=float)
        for _, group in groups:
            unit_prediction.loc[group.index] = _simulate_group(group, settings)
        correction = (
            unit_prediction.to_numpy(dtype=float)
            - frame["base_prediction"].to_numpy(dtype=float)
        )
        for share in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
            candidate_settings = CompensationSettings(
                signal, aggregation, window, share
            )
            prediction_columns[candidate_settings.name] = (
                frame["base_prediction"].to_numpy(dtype=float)
                + share * correction
            )
    predictions = pd.DataFrame(prediction_columns, index=frame.index)

    cross_folds = sorted(
        fold
        for fold in frame["fold"].astype(str).unique()
        if fold.startswith("cross_month_")
    )
    selected_global = pd.Series(np.nan, index=frame.index, dtype=float)
    selected_column = pd.Series(np.nan, index=frame.index, dtype=float)
    recipes: dict[str, object] = {"cross_fitted": {}}
    for held_fold in cross_folds:
        held_index = frame.index[frame["fold"].eq(held_fold)]
        global_prediction, global_recipe = _select_cross_month_recipe(
            frame,
            predictions,
            held_fold=held_fold,
            per_column=False,
        )
        column_prediction, column_recipe = _select_cross_month_recipe(
            frame,
            predictions,
            held_fold=held_fold,
            per_column=True,
        )
        selected_global.loc[held_index] = global_prediction.loc[held_index]
        selected_column.loc[held_index] = column_prediction.loc[held_index]
        recipes["cross_fitted"][held_fold] = {
            "global": global_recipe,
            "column": column_recipe,
        }

    recent_index = frame.index[
        frame["fold"].astype(str).str.startswith("recent_")
    ]
    deployment_global, deployment_global_recipe = _select_cross_month_recipe(
        frame,
        predictions,
        held_fold=None,
        per_column=False,
    )
    deployment_column, deployment_column_recipe = _select_cross_month_recipe(
        frame,
        predictions,
        held_fold=None,
        per_column=True,
    )
    selected_global.loc[recent_index] = deployment_global.loc[recent_index]
    selected_column.loc[recent_index] = deployment_column.loc[recent_index]
    recipes["deployment"] = {
        "global": deployment_global_recipe,
        "column": deployment_column_recipe,
    }
    predictions["selected_global"] = selected_global
    predictions["selected_column"] = selected_column

    score_rows: list[dict[str, object]] = []
    split_frames = {
        "all": frame,
        "cross_month": frame.loc[
            frame["fold"].astype(str).str.startswith("cross_month_")
        ],
        "recent": frame.loc[
            frame["fold"].astype(str).str.startswith("recent_")
        ],
    }
    for method in predictions.columns:
        for split, split_frame in split_frames.items():
            score_rows.append(
                {
                    "method": method,
                    "split": split,
                    "target": "all",
                    "score": _competition_score(split_frame, predictions[method]),
                }
            )
            for target, target_frame in split_frame.groupby("target", sort=True):
                score_rows.append(
                    {
                        "method": method,
                        "split": split,
                        "target": target,
                        "score": _competition_score(
                            target_frame, predictions[method]
                        ),
                    }
                )
    scores = pd.DataFrame(score_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    frame[[*KEYS, "y_true", "base_prediction"]].assign(
        selected_global=predictions["selected_global"],
        selected_column=predictions["selected_column"],
    ).to_csv(args.output / "selected_predictions.csv", index=False, encoding="utf-8")
    scores.to_csv(args.output / "scores.csv", index=False, encoding="utf-8")
    (args.output / "recipes.json").write_text(
        json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n全局得分最高的在线补偿候选：")
    print(
        scores.loc[
            scores["split"].eq("all") & scores["target"].eq("all")
        ]
        .nlargest(15, "score")
        .to_string(index=False)
    )
    print("\n严格跨月份选择结果：")
    print(
        scores.loc[
            scores["method"].isin(
                ["base_prediction", "selected_global", "selected_column"]
            )
        ]
        .sort_values(["split", "target", "score"], ascending=[True, True, False])
        .to_string(index=False)
    )
    print(f"\n实验结果已保存: {args.output}")


if __name__ == "__main__":
    main()
