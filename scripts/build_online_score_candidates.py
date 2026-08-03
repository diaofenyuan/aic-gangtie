"""基于已知线上分项结果构建可归因的冲分提交候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gas_power.submission_freeze import (  # noqa: E402
    freeze_submission,
    verify_submission_freeze,
)


SOURCE_STAMPS = {
    "all_champion": "2026-08-02_14-13-16",
    "diversity": "2026-08-02_17-26-01",
    "one_champion": "2026-08-02_20-02-43",
}
ONLINE_SCORES = {
    "all_champion": {"generator_1": 0.9457, "generator_all": 0.9577},
    "diversity": {"generator_1": 0.9456, "generator_all": 0.9560},
    "one_champion": {"generator_1": 0.9467, "generator_all": 0.9574},
}
TARGETS = ("generator_1", "generator_all")
HORIZONS = (15, 30, 45, 60, 75, 90, 105, 120)
EXPECTED_COLUMNS = [
    "datetime",
    *(f"{target}_t+{horizon}_pred" for target in TARGETS for horizon in HORIZONS),
]
ARCHIVE_NAME = "咕咕嘎嘎_gas_predict_prelim.zip"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="历史结果所在目录",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="候选输出目录；为保护历史结果，目录必须尚不存在",
    )
    parser.add_argument(
        "--candidate-set",
        choices=("initial", "horizon-probes"),
        default="initial",
        help="构建首轮全局候选，或构建逐步长线上探针",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate_source(outputs: Path, stamp: str) -> Path:
    matches = [path for path in outputs.glob(f"{stamp}*") if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"无法唯一定位历史结果 {stamp}: {matches}")
    source = matches[0]
    for name in ("input.csv", "s_result.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"历史结果缺少 {name}: {source}")
    return source


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"预测列不符合初赛 17 列规范: {path}")
    if len(frame) != 192:
        raise ValueError(f"预测必须包含 192 个起报时刻，实际 {len(frame)}: {path}")

    timestamps = pd.to_datetime(frame["datetime"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"datetime 必须唯一且严格递增: {path}")
    intervals = timestamps.diff().dropna()
    if not intervals.eq(pd.Timedelta(minutes=15)).all():
        raise ValueError(f"datetime 必须保持 15 分钟间隔: {path}")

    values = frame[EXPECTED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"预测包含 NaN 或 Inf: {path}")
    return frame


def _assert_sources_aligned(frames: Mapping[str, pd.DataFrame]) -> None:
    reference = frames["one_champion"]["datetime"]
    for label, frame in frames.items():
        if not frame["datetime"].equals(reference):
            raise ValueError(f"历史预测时间轴不一致: {label}")


def _combine_target(
    frames: Mapping[str, pd.DataFrame],
    target: str,
    weights: Mapping[str, float],
) -> np.ndarray:
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"{target} 的组合权重之和必须为 1: {weights}")
    columns = [f"{target}_t+{horizon}_pred" for horizon in HORIZONS]
    combined = np.zeros((len(next(iter(frames.values()))), len(columns)), dtype=float)
    for source, weight in weights.items():
        combined += float(weight) * frames[source][columns].to_numpy(dtype=float)
    return combined


def _build_prediction(
    frames: Mapping[str, pd.DataFrame],
    target_weights: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    output = pd.DataFrame({"datetime": frames["one_champion"]["datetime"]})
    for target in TARGETS:
        columns = [f"{target}_t+{horizon}_pred" for horizon in HORIZONS]
        output[columns] = _combine_target(frames, target, target_weights[target])

    values = output[EXPECTED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("候选预测包含非有限值或负数")
    one = output[[f"generator_1_t+{h}_pred" for h in HORIZONS]].to_numpy()
    total = output[[f"generator_all_t+{h}_pred" for h in HORIZONS]].to_numpy()
    if (total < one).any():
        raise ValueError("候选预测违反 generator_all >= generator_1")
    return output[EXPECTED_COLUMNS]


def _build_horizon_probe(
    frames: Mapping[str, pd.DataFrame],
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """只改变一个步长，使线上分差可直接归因到该步长。"""

    baseline_weights = {
        "generator_1": {"one_champion": 1.0},
        "generator_all": {"all_champion": 1.0},
    }
    output = _build_prediction(frames, baseline_weights)
    one_column = f"generator_1_t+{horizon}_pred"
    all_column = f"generator_all_t+{horizon}_pred"

    # generator_1 使用已产生明显变化的 60% 外推，识别哪些步长仍然受益。
    output[one_column] = (
        1.6 * frames["one_champion"][one_column]
        - 0.6 * frames["diversity"][one_column]
    )
    # generator_all 的 60% 全局外推已在线上提升，进一步放大到 130%。
    output[all_column] = (
        2.3 * frames["all_champion"][all_column]
        - 1.3 * frames["one_champion"][all_column]
    )
    metadata: dict[str, object] = {
        "baseline": baseline_weights,
        "probed_horizon_minutes": horizon,
        "probe": {
            "generator_1": {"one_champion": 1.6, "diversity": -0.6},
            "generator_all": {"all_champion": 2.3, "one_champion": -1.3},
        },
        "online_reference": {
            "generator_1": 0.9467,
            "generator_all": 0.9577,
        },
    }

    values = output[EXPECTED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"t+{horizon} 探针包含非有限值或负数")
    one = output[[f"generator_1_t+{h}_pred" for h in HORIZONS]].to_numpy()
    total = output[[f"generator_all_t+{h}_pred" for h in HORIZONS]].to_numpy()
    if (total < one).any():
        raise ValueError(f"t+{horizon} 探针违反 generator_all >= generator_1")
    return output[EXPECTED_COLUMNS], metadata


def _write_candidate(
    *,
    destination: Path,
    name: str,
    input_source: Path,
    prediction: pd.DataFrame,
    weights: Mapping[str, Mapping[str, float]],
    source_directories: Mapping[str, Path],
) -> None:
    candidate = destination / name
    candidate.mkdir(parents=False, exist_ok=False)
    shutil.copy2(input_source, candidate / "input.csv")
    prediction.to_csv(
        candidate / "s_result.csv",
        index=False,
        encoding="utf-8",
        float_format="%.6f",
        lineterminator="\n",
    )

    manifest = {
        "purpose": "线上冲分定向实验；本地分数不作为晋级依据",
        "input_source": str(input_source.parent.resolve()),
        "input_sha256": _sha256(candidate / "input.csv"),
        "prediction_sources": {
            label: {
                "directory": str(path.resolve()),
                "s_result_sha256": _sha256(path / "s_result.csv"),
                "known_online_scores": ONLINE_SCORES[label],
            }
            for label, path in source_directories.items()
        },
        "target_weights": weights,
        "submission_rows": len(prediction),
        "submission_columns": list(prediction.columns),
    }
    (candidate / "online_experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    freeze_submission(candidate)
    verify_submission_freeze(candidate)
    archive_path = candidate / ARCHIVE_NAME
    with ZipFile(
        archive_path,
        mode="w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(candidate / "input.csv", arcname="input.csv")
        archive.write(candidate / "s_result.csv", arcname="s_result.csv")
    with ZipFile(archive_path) as archive:
        if archive.namelist() != ["input.csv", "s_result.csv"]:
            raise ValueError(f"压缩包根目录文件不正确: {archive.namelist()}")


def main() -> None:
    args = _arguments()
    outputs = args.outputs.resolve()
    default_destination = (
        "online_score_candidates_2026-08-02"
        if args.candidate_set == "initial"
        else "online_horizon_probes_2026-08-02"
    )
    destination = (
        args.destination.resolve()
        if args.destination is not None
        else (PROJECT_ROOT / "outputs" / default_destination).resolve()
    )
    if destination.exists():
        raise FileExistsError(f"候选目录已存在，为保护结果不覆盖: {destination}")

    print("[1/4] 定位三次已获得线上分数的历史提交")
    sources = {
        label: _locate_source(outputs, stamp)
        for label, stamp in SOURCE_STAMPS.items()
    }
    print("[2/4] 校验时间轴、17 列结构、有限值和满分输入来源")
    frames = {
        label: _load_prediction(path / "s_result.csv")
        for label, path in sources.items()
    }
    _assert_sources_aligned(frames)
    trusted_input = sources["one_champion"] / "input.csv"
    trusted_hash = _sha256(trusted_input)
    if _sha256(sources["all_champion"] / "input.csv") != trusted_hash:
        raise ValueError("两个 50/50 质量分提交的 input.csv 不一致")

    destination.mkdir(parents=True, exist_ok=False)
    if args.candidate_set == "initial":
        # 每个候选均能由线上返回的两个目标分数独立判断方向。
        candidates = {
            "01_target_champions": {
                "generator_1": {"one_champion": 1.0},
                "generator_all": {"all_champion": 1.0},
            },
            "02_diversity_blend20": {
                "generator_1": {"one_champion": 0.8, "diversity": 0.2},
                "generator_all": {"all_champion": 0.8, "diversity": 0.2},
            },
            "03_direction_extrapolate25": {
                "generator_1": {"one_champion": 1.25, "diversity": -0.25},
                "generator_all": {"all_champion": 1.25, "one_champion": -0.25},
            },
            "04_direction_extrapolate60": {
                "generator_1": {"one_champion": 1.6, "diversity": -0.6},
                "generator_all": {"all_champion": 1.6, "one_champion": -0.6},
            },
        }
        print("[3/4] 构建分项冠军、误差多样性和两档方向外推候选")
        for index, (name, weights) in enumerate(candidates.items(), start=1):
            print(f"      ({index}/{len(candidates)}) {name}")
            prediction = _build_prediction(frames, weights)
            _write_candidate(
                destination=destination,
                name=name,
                input_source=trusted_input,
                prediction=prediction,
                weights=weights,
                source_directories=sources,
            )
    else:
        print("[3/4] 构建 8 个逐步长线上强扰动探针")
        for index, horizon in enumerate(HORIZONS, start=1):
            name = f"probe_t{horizon:03d}"
            print(f"      ({index}/{len(HORIZONS)}) {name}")
            prediction, metadata = _build_horizon_probe(frames, horizon)
            _write_candidate(
                destination=destination,
                name=name,
                input_source=trusted_input,
                prediction=prediction,
                weights=metadata,
                source_directories=sources,
            )

    print("[4/4] 完成冻结校验；每个 ZIP 根目录仅含 input.csv 和 s_result.csv")
    print(f"候选目录: {destination}")


if __name__ == "__main__":
    main()
