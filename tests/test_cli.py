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
        "submission_archive": "outputs/teamname_gas_predict_prelim.zip",
        "result_file": "outputs/运行结果.json",
        "output_directory": "outputs/预测结果",
    }

    summary = _format_console_summary("run", result)

    assert len(summary.splitlines()) == 7
    assert "运行完成：完整预测任务" in summary
    assert "训练：最后值保持模型" in summary
    assert "泄漏检查通过" in summary
    assert "不应输出到摘要" not in summary


def test_console_summary_prints_local_score_last() -> None:
    result = {
        "validate": {
            "folds": 8,
            "leakage_checks": "passed",
            "local_score": {
                "official_score": False,
                "cross_month": {
                    "score": {
                        "final_score": 0.936443,
                        "display_scale": 1.0,
                        "score_percent": 93.6443,
                    }
                },
                "recent": {
                    "score": {
                        "final_score": 0.951728,
                        "display_scale": 1.0,
                        "score_percent": 95.1728,
                    }
                },
            },
        },
        "output_directory": "outputs/预测结果",
    }

    summary = _format_console_summary("run", result)

    assert summary.splitlines()[-2:] == [
        "本地得分：93.644300 分（跨月份训练期滚动验证，非官方榜分）",
        "近期得分：95.172800 分（近期训练期滚动验证，非官方榜分）",
    ]
