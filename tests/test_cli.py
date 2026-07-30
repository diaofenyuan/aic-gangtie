from __future__ import annotations

from pathlib import Path

from gas_power.cli import build_parser


def test_cli_defaults_to_complete_prediction_task() -> None:
    args = build_parser().parse_args([])

    assert args.command == "run"
    assert args.config == Path("config/default.yaml")


def test_cli_accepts_worker_and_progress_overrides() -> None:
    args = build_parser().parse_args(
        ["--workers", "8", "--no-progress", "validate"]
    )

    assert args.command == "validate"
    assert args.workers == 8
    assert args.progress is False
