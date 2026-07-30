from __future__ import annotations

from pathlib import Path

from gas_power.config import configured_value, load_config


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
