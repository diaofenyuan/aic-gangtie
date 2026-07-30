"""提交文件联合校验和可追踪清单生成。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from gas_power.availability import FeatureAvailabilityRegistry
from gas_power.config import ProjectConfig
from gas_power.outputs import (
    validate_forecast_frame,
    validate_optimization_frame,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_utf8_csv(path: Path) -> pd.DataFrame:
    try:
        path.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"提交文件不是 UTF-8: {path}") from exc
    return pd.read_csv(path, encoding="utf-8")


def validate_submission_bundle(
    config: ProjectConfig,
    expected_origins: pd.DatetimeIndex,
    training_metadata: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    results = config.path("results")
    short_path = results / "s_result.csv"
    long_path = results / "l_result.csv"
    opt_path = results / "opt_result.csv"
    missing = [str(path) for path in (short_path, long_path, opt_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"提交文件缺失: {missing}")

    targets = [str(value) for value in config.section("data")["roles"]["targets"]]
    forecast = config.section("forecast")
    interval = int(config.section("optimization").get("interval_minutes", 15))
    post = config.raw.get("postprocessing", {})
    capacities = post.get("target_capacity_mw", {}) if isinstance(post, Mapping) else {}
    submission = config.raw.get("submission", {})
    tolerance = float(submission.get("capacity_tolerance_mw", 1.0e-6))

    short = validate_forecast_frame(
        _read_utf8_csv(short_path),
        targets,
        list(range(1, int(forecast["short_steps"]) + 1)),
        str(config.section("data")["timestamp_format"]),
        interval,
        expected_origins=expected_origins,
        capacity_bounds=dict(capacities),
        enforce_target_consistency=True,
        capacity_tolerance=tolerance,
    )
    long = validate_forecast_frame(
        _read_utf8_csv(long_path),
        targets,
        list(range(1, int(forecast["long_steps"]) + 1)),
        str(config.section("data")["timestamp_format"]),
        interval,
        expected_origins=expected_origins,
        capacity_bounds=dict(capacities),
        enforce_target_consistency=True,
        capacity_tolerance=tolerance,
    )
    gas_columns = [
        str(config.section("optimization")["output_columns"][gas])
        for gas in config.section("optimization")["gas_types"]
    ]
    opt = validate_optimization_frame(
        _read_utf8_csv(opt_path), gas_columns, str(config.section("data")["timestamp_format"])
    )

    registry = FeatureAvailabilityRegistry.from_config(config)
    whitelist = sorted(
        set(registry.allowed_source_columns("short")).intersection(
            registry.allowed_source_columns("long")
        )
    )
    manifest = {
        "manifest_version": 1,
        "datetime_semantics": "forecast_origin",
        "code_version": {"type": "source_sha256", "value": _code_digest(config.root)},
        "config": {"path": str(config.source), "sha256": _sha256(config.source)},
        "availability": {
            "path": str(registry.path),
            "sha256": _sha256(registry.path) if registry.path and registry.path.exists() else None,
        },
        "training": {
            "start": training_metadata.get("train_start"),
            "end": training_metadata.get("train_end"),
            "rows": training_metadata.get("training_rows"),
            "model_type": training_metadata.get("model_type"),
        },
        "feature_source_whitelist": whitelist,
        "feature_columns": list(feature_columns),
        "files": {
            "s_result.csv": {"rows": len(short), "columns": len(short.columns), "sha256": _sha256(short_path)},
            "l_result.csv": {"rows": len(long), "columns": len(long.columns), "sha256": _sha256(long_path)},
            "opt_result.csv": {"rows": len(opt), "columns": len(opt.columns), "sha256": _sha256(opt_path)},
        },
        "checks": {
            "datetime_is_forecast_origin": True,
            "horizons_complete_and_ordered": True,
            "targets_not_swapped_by_consistency": True,
            "no_missing_duplicate_nan_inf_or_index_column": True,
            "utf8": True,
            "capacity_bounds": True,
        },
    }
    path = results / "submission_manifest.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
