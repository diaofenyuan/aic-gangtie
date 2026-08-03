"""使用历史相似轨迹做因果类比预测，并与锁定融合 OOF 比较。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["generator_1", "generator_all"]
HORIZONS = list(range(1, 9))
KEYS = ["fold", "origin", "target_datetime", "target", "horizon_steps"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--recent-folds", type=int, default=1)
    parser.add_argument("--cross-month-folds", type=int, default=1)
    parser.add_argument("--baseline-run", default="2026-08-02_20-02-43*")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _baseline_directory(pattern: str) -> Path:
    matches = sorted((PROJECT_ROOT / "outputs").glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"基线运行目录应唯一匹配，实际找到 {len(matches)} 个: {pattern}")
    return matches[0]


def _patterns(
    values: np.ndarray,
    positions: np.ndarray,
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.stack(
        [values[position - context + 1 : position + 1] for position in positions]
    )
    differences = np.diff(windows, axis=1)
    center = np.median(differences, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(differences - center), axis=1)
    scale = np.maximum(scale, 0.5)
    normalized = differences / scale[:, None, :]
    # 最近水平和一阶趋势提供工况强度，轨迹差分仍占距离主体。
    level = windows[:, -1, :] / np.asarray([200.0, 440.0])[None, :]
    trend = differences[:, -min(4, differences.shape[1]) :, :].mean(axis=1)
    trend = trend / scale
    pattern = np.concatenate(
        [normalized.reshape(len(positions), -1), level, trend], axis=1
    )
    return pattern.astype(np.float32), scale.astype(np.float32)


def _fold_predictions(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    fold: str,
    origins: pd.DatetimeIndex,
    *,
    context: int,
    neighbors: int,
) -> pd.DataFrame:
    origin_positions = index.get_indexer(origins)
    if (origin_positions < 0).any():
        raise ValueError(f"{fold} 的预测起点未在处理后数据中完整找到")
    train_end = int(origin_positions.min()) - 1
    candidate_positions = np.arange(
        context - 1,
        train_end - max(HORIZONS) + 1,
        dtype=np.int64,
    )
    if len(candidate_positions) < neighbors:
        raise ValueError(f"{fold} 的相似轨迹候选不足 {neighbors} 个")

    candidate_patterns, candidate_scale = _patterns(
        values, candidate_positions, context
    )
    query_patterns, query_scale = _patterns(values, origin_positions, context)
    search = NearestNeighbors(
        n_neighbors=int(neighbors), metric="euclidean", algorithm="brute", n_jobs=-1
    ).fit(candidate_patterns)
    distances, neighbor_indexes = search.kneighbors(query_patterns)
    distance_scale = np.maximum(np.median(distances, axis=1, keepdims=True), 1.0e-6)
    weights = np.exp(-distances / distance_scale)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)

    neighbor_positions = candidate_positions[neighbor_indexes]
    neighbor_scales = candidate_scale[neighbor_indexes]
    volatility_ratio = np.clip(
        query_scale[:, None, :] / np.maximum(neighbor_scales, 1.0e-6),
        0.5,
        2.0,
    )
    residuals = np.empty((len(origins), len(HORIZONS), len(TARGETS)), dtype=float)
    for horizon_index, horizon in enumerate(HORIZONS):
        neighbor_changes = (
            values[neighbor_positions + horizon] - values[neighbor_positions]
        )
        adjusted = neighbor_changes * volatility_ratio
        residuals[:, horizon_index, :] = np.sum(
            adjusted * weights[:, :, None], axis=1
        )
    predictions = residuals + values[origin_positions, None, :]

    rows: list[dict[str, object]] = []
    for origin_index, origin in enumerate(origins):
        for horizon_index, horizon in enumerate(HORIZONS):
            target_time = origin + pd.Timedelta(minutes=15 * horizon)
            for target_index, target in enumerate(TARGETS):
                rows.append(
                    {
                        "fold": fold,
                        "origin": origin,
                        "target_datetime": target_time,
                        "target": target,
                        "horizon_steps": horizon,
                        "analog_prediction": predictions[
                            origin_index, horizon_index, target_index
                        ],
                    }
                )
    return pd.DataFrame(rows)


def _score(frame: pd.DataFrame, column: str) -> float:
    mapes: list[float] = []
    for _, group in frame.groupby(["target", "horizon_steps"], sort=True):
        actual = group["y_true"].to_numpy(dtype=float)
        predicted = group[column].to_numpy(dtype=float)
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
    if min(args.context, args.neighbors, args.recent_folds, args.cross_month_folds) <= 0:
        raise ValueError("上下文、邻居数和验证折数必须大于 0")
    baseline = _baseline_directory(args.baseline_run)
    incumbent = pd.read_csv(
        baseline / "reports" / "high_accuracy_fused_leave_one_fold.csv",
        usecols=[*KEYS, "y_true", "y_pred"],
    ).rename(columns={"y_pred": "incumbent_prediction"})
    incumbent["origin"] = pd.to_datetime(incumbent["origin"], errors="raise")
    incumbent["target_datetime"] = pd.to_datetime(
        incumbent["target_datetime"], errors="raise"
    )

    cross_folds = sorted(
        fold for fold in incumbent["fold"].unique() if str(fold).startswith("cross_month_")
    )[: int(args.cross_month_folds)]
    recent_folds = sorted(
        fold for fold in incumbent["fold"].unique() if str(fold).startswith("recent_")
    )[-int(args.recent_folds) :]
    selected_folds = [*cross_folds, *recent_folds]
    selected = incumbent.loc[incumbent["fold"].isin(selected_folds)].copy()

    processed = pd.read_csv(
        PROJECT_ROOT / "cache" / "processed.csv",
        usecols=["datetime", *TARGETS],
        encoding="utf-8",
    )
    processed["datetime"] = pd.to_datetime(processed["datetime"], errors="raise")
    processed = processed.set_index("datetime").sort_index()
    values = processed[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("处理后目标序列仍包含非有限值")

    parts: list[pd.DataFrame] = []
    for fold in selected_folds:
        origins = pd.DatetimeIndex(
            sorted(selected.loc[selected["fold"].eq(fold), "origin"].unique())
        )
        print(f"相似轨迹 {fold}: 起点 {len(origins)}，上下文 {args.context}，邻居 {args.neighbors}")
        parts.append(
            _fold_predictions(
                values,
                processed.index,
                str(fold),
                origins,
                context=int(args.context),
                neighbors=int(args.neighbors),
            )
        )
    analog = pd.concat(parts, ignore_index=True)
    joined = selected.merge(analog, on=KEYS, how="inner", validate="one_to_one")
    if len(joined) != len(selected):
        raise ValueError("相似轨迹预测未完整对齐锁定 OOF")
    for share in (0.10, 0.20, 0.30, 0.50):
        joined[f"blend_{int(share * 100):02d}"] = (
            (1.0 - share) * joined["incumbent_prediction"]
            + share * joined["analog_prediction"]
        )

    methods = [
        "incumbent_prediction",
        "analog_prediction",
        "blend_10",
        "blend_20",
        "blend_30",
        "blend_50",
    ]
    report_rows: list[dict[str, object]] = []
    joined["validation_type"] = joined["fold"].str.rsplit("_", n=1).str[0]
    for split, group in joined.groupby("validation_type", sort=True):
        baseline_score = _score(group, "incumbent_prediction")
        for method in methods:
            score = _score(group, method)
            report_rows.append(
                {
                    "validation_type": split,
                    "method": method,
                    "score": score,
                    "score_gain": score - baseline_score,
                    "folds": group["fold"].nunique(),
                }
            )
    report = pd.DataFrame(report_rows).sort_values(
        ["validation_type", "score"], ascending=[True, False]
    )
    output = args.output or (
        PROJECT_ROOT
        / "cache"
        / "experiments"
        / f"analog_c{args.context}_k{args.neighbors}_r{args.recent_folds}_x{args.cross_month_folds}"
    )
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    report.to_csv(output / "scores.csv", index=False, encoding="utf-8")
    print("\n相似轨迹 OOF 得分：")
    print(report.to_string(index=False))
    print(f"\n实验结果已保存: {output}")


if __name__ == "__main__":
    main()
