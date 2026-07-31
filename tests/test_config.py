from __future__ import annotations

from pathlib import Path

import yaml

from gas_power.config import (
    ConfigError,
    configured_value,
    load_config,
    validate_competition_command,
)
from gas_power.data import load_original_input_frame
from gas_power.pipeline import _model_horizons


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_marks_unconfirmed_parameters() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    optimization = config.section("optimization")
    for gas_type, settings in optimization["gas_types"].items():
        conversion = settings["conversion_mw_per_volume"]
        assert configured_value(
            conversion, f"optimization.gas_types.{gas_type}.conversion_mw_per_volume"
        ) > 0
        assert "等待" in conversion["status"]
    for parameter in optimization["objective"].values():
        assert "等待" in parameter["status"]


def test_default_config_has_required_units_and_horizons() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    units = config.section("optimization")["units"]
    assert [unit["rated_mw"] for unit in units].count(50.0) == 4
    assert [unit["rated_mw"] for unit in units].count(120.0) == 2
    assert config.section("forecast")["short_steps"] == 8
    assert config.section("forecast")["long_steps"] == 96


def test_default_config_does_not_define_root_result_directories() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    paths = config.section("paths")

    assert "results" not in paths
    assert "reports" not in paths
    assert paths["outputs"] == "outputs"


def test_official_preliminary_config_uses_isolated_official_paths() -> None:
    config = load_config(PROJECT_ROOT / "config" / "official_preliminary.yaml")

    assert config.section("competition_compliance")["stage"] == "preliminary"
    assert config.path("data") == PROJECT_ROOT / "data" / "preliminary" / "train"
    assert (
        config.path("scoring_data")
        == PROJECT_ROOT / "data" / "preliminary" / "scoring"
    )
    assert config.section("data")["tables"]["load"]["path"] == "Pre_load.csv"
    assert (
        config.section("prediction_input")["table_paths"]["load"]
        == "Pre_test_load.csv"
    )
    assert config.section("forecast")["model"]["type"] == "last_value"
    assert config.section("forecast")["model"]["components"] == []
    assert config.section("forecast")["prediction_origins"]["mode"] == "scoring"
    assert _model_horizons(config) == list(range(1, 9))
    assert config.section("validation")["folds"] == 8
    assert "test_evaluation" not in config.raw


def test_official_input_keeps_all_raw_fields_with_original_names() -> None:
    config = load_config(PROJECT_ROOT / "config" / "official_preliminary.yaml")
    settings = config.section("prediction_input")

    frame = load_original_input_frame(
        config,
        config.path("scoring_data"),
        settings["table_paths"],
    )

    assert frame.shape == (192, 29)
    assert "blast_furnace_5" in frame
    assert "air_heater_5" in frame
    assert "blast_furnace_user4" in frame
    assert "blast_furnace_gas_holder_2" in frame
    assert not any(str(column).startswith("feat_") for column in frame)


def test_official_config_rejects_scoring_directory_as_training_data(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "config" / "official_preliminary.yaml"
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw["extends"] = str(PROJECT_ROOT / "config" / "default.yaml")
    raw["paths"]["data"] = raw["paths"]["scoring_data"]
    invalid = tmp_path / "invalid_official.yaml"
    with invalid.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, allow_unicode=True, sort_keys=False)

    try:
        load_config(invalid)
    except ConfigError as exc:
        assert "受限目录" in str(exc)
    else:
        raise AssertionError("评分目录被误配为训练数据时必须拒绝加载")


def test_official_preliminary_config_rejects_non_preliminary_commands() -> None:
    config = load_config(PROJECT_ROOT / "config" / "official_preliminary.yaml")

    validate_competition_command(config, "predict")
    try:
        validate_competition_command(config, "optimize")
    except ConfigError as exc:
        assert "初赛正式配置禁止" in str(exc)
    else:
        raise AssertionError("初赛正式配置不得执行优化命令")
