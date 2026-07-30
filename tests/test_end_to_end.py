from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from gas_power.config import load_config
from gas_power.outputs import validate_forecast_frame, validate_optimization_frame
from gas_power.pipeline import (
    audit_data_pipeline,
    demo_pipeline,
    validate_submission_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_end_to_end_in_isolated_directory(tmp_path: Path) -> None:
    with (PROJECT_ROOT / "config" / "default.yaml").open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw["paths"] = {
        "root": str(tmp_path),
        "data": "data/synthetic",
        "cache": "cache",
        "logs": "logs",
        "models": "models",
        "results": "results",
    }
    raw["synthetic"]["periods"] = 800
    raw["validation"].update(
        {
            "folds": 2,
            "initial_train_points": 400,
            "validation_points": 16,
            "step_points": 64,
            "rolling_train_points": 400,
        }
    )
    raw["forecast"]["prediction_origins"]["count"] = 2
    raw["optimization"]["horizon_steps"] = 16
    config_path = tmp_path / "test_config.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, allow_unicode=True, sort_keys=False)

    config = load_config(config_path)
    result = demo_pipeline(config)
    assert result["validate"]["leakage_checks"] == "passed"
    assert result["optimize"]["diagnostics"]["success"] is True
    assert result["optimize"]["diagnostics"]["priority_mode"] == "lexicographic"

    results_dir = tmp_path / "results"
    short = pd.read_csv(results_dir / "s_result.csv", encoding="utf-8")
    long = pd.read_csv(results_dir / "l_result.csv", encoding="utf-8")
    opt = pd.read_csv(results_dir / "opt_result.csv", encoding="utf-8")
    targets = raw["data"]["roles"]["targets"]
    validate_forecast_frame(short, targets, list(range(1, 9)))
    validate_forecast_frame(long, targets, list(range(1, 97)))
    gas_columns = [
        raw["optimization"]["output_columns"][gas]
        for gas in raw["optimization"]["gas_types"]
    ]
    validate_optimization_frame(opt, gas_columns)
    assert (results_dir / "optimization_diagnostics.json").exists()
    assert (results_dir / "reports" / "resource_boundary_forecast.csv").exists()
    unit_plan = pd.read_csv(results_dir / "optimization_unit_plan.csv", encoding="utf-8")
    assert {"generator_1_plan_mw", "generator_all_plan_mw"}.issubset(unit_plan.columns)
    assert (results_dir / "reports" / "inference_runtime.json").exists()
    assert (tmp_path / "models" / "forecast_model.joblib").exists()

    audit = audit_data_pipeline(config)
    assert "future_generator_all_leak" in audit["future_red_fields"]
    formal_features = pd.read_csv(tmp_path / "cache" / "features.csv", encoding="utf-8")
    assert not any("future" in column for column in formal_features.columns)

    submission = validate_submission_pipeline(config)
    assert submission["checks"]["datetime_is_forecast_origin"] is True
    assert (results_dir / "submission_manifest.json").exists()
