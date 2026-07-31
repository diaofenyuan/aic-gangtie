from __future__ import annotations

from pathlib import Path

from gas_power.cli import _format_console_summary, build_parser


def test_cli_defaults_to_complete_prediction_task() -> None:
    args = build_parser().parse_args([])

    assert args.command == "run"
    assert args.config == Path("config/official_preliminary.yaml")
    assert args.json_output is False


def test_cli_accepts_worker_and_progress_overrides() -> None:
    args = build_parser().parse_args(
        ["--workers", "8", "--no-progress", "validate"]
    )

    assert args.command == "validate"
    assert args.workers == 8
    assert args.progress is False


def test_cli_accepts_full_json_output() -> None:
    args = build_parser().parse_args(["--json", "run"])

    assert args.json_output is True


def test_console_summary_is_compact_and_uses_chinese_labels() -> None:
    result = {
        "train": {
            "model_type": "LastValueModel",
            "training_rows": 11521,
            "train_start": "2025-01-01 00:00:00",
            "train_end": "2025-05-01 00:00:00",
            "feature_columns": ["不应输出到摘要"],
        },
        "validate": {"folds": 2, "leakage_checks": "passed"},
        "predict": {
            "origins": 1,
            "runtime": {"total_inference_seconds": 9.278},
        },
        "submission_archive": "outputs/提交压缩包.zip",
        "result_file": "outputs/运行结果.json",
        "output_directory": "outputs/预测结果",
    }

    summary = _format_console_summary("run", result)

    assert len(summary.splitlines()) == 7
    assert "运行完成：完整预测任务" in summary
    assert "训练：最后值保持模型" in summary
    assert "泄漏检查通过" in summary
    assert "不应输出到摘要" not in summary
