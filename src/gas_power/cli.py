"""项目命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from threadpoolctl import threadpool_limits

from gas_power.config import ProjectConfig, load_config, validate_competition_command
from gas_power.logging_utils import configure_logging, set_random_seed
from gas_power.output_session import OutputSession
from gas_power.pipeline import (
    audit_data_pipeline,
    backtest_pipeline,
    benchmark_pipeline,
    demo_pipeline,
    discover_relations_pipeline,
    environment_pipeline,
    generate_synthetic_pipeline,
    optimize_pipeline,
    predict_pipeline,
    run_task_pipeline,
    train_pipeline,
    tune_pipeline,
    validate_pipeline,
    validate_submission_pipeline,
)
from gas_power.runtime import configure_runtime


Pipeline = Callable[[ProjectConfig], dict[str, Any]]

COMMAND_LABELS = {
    "generate-synthetic": "生成合成数据",
    "doctor": "检查高精度训练环境",
    "train": "训练模型",
    "tune": "高精度搜索与门控",
    "validate": "滚动验证",
    "audit-data": "数据审计",
    "benchmark": "基线评测",
    "discover-relations": "关系发现",
    "backtest": "扩展回测",
    "predict": "生成预测结果",
    "optimize": "发电优化",
    "validate-submission": "校验提交文件",
    "run": "完整预测任务",
    "demo": "合成数据演示",
}

MODEL_LABELS = {
    "WeightedEnsembleModel": "加权融合模型",
    "LastValueModel": "最后值保持模型",
}


def _result_section(
    command: str,
    result: dict[str, Any],
    section_name: str,
) -> dict[str, Any] | None:
    """兼容单阶段命令和完整任务的嵌套返回结构。"""

    if command == section_name:
        return result
    section = result.get(section_name)
    return section if isinstance(section, dict) else None


def _format_console_summary(command: str, result: dict[str, Any]) -> str:
    """生成适合人工阅读的简短中文运行摘要。"""

    lines = [f"运行完成：{COMMAND_LABELS.get(command, command)}"]
    train = _result_section(command, result, "train")
    if train is not None:
        model_type = str(train.get("model_type", "未知模型"))
        model_label = MODEL_LABELS.get(model_type, model_type)
        rows = int(train.get("training_rows", 0))
        lines.append(
            f"训练：{model_label}，{rows:,} 行，"
            f"{train.get('train_start', '未知')} 至 {train.get('train_end', '未知')}"
        )

    validation = _result_section(command, result, "validate")
    if validation is not None:
        leakage_label = {
            "passed": "通过",
            "failed": "未通过",
        }.get(str(validation.get("leakage_checks", "")), "未知")
        lines.append(
            f"验证：{int(validation.get('folds', 0))} 折，泄漏检查{leakage_label}"
        )

    prediction = _result_section(command, result, "predict")
    if prediction is not None:
        runtime = prediction.get("runtime", {})
        seconds = (
            float(runtime.get("total_inference_seconds", 0.0))
            if isinstance(runtime, dict)
            else 0.0
        )
        lines.append(
            f"预测：{int(prediction.get('origins', 0))} 个起点，推理耗时 {seconds:.2f} 秒"
        )

    archive_path = result.get("submission_archive")
    if archive_path:
        lines.append(f"提交压缩包：{archive_path}")
    result_file = result.get("result_file")
    if result_file:
        lines.append(f"完整结果：{result_file}")
    output_directory = result.get("output_directory")
    if output_directory:
        lines.append(f"输出目录：{output_directory}")

    if validation is not None:
        local_score = validation.get("local_score")
        if isinstance(local_score, dict):
            for section_name, label, protocol in (
                ("cross_month", "本地得分", "跨月份训练期滚动验证"),
                ("recent", "近期得分", "近期训练期滚动验证"),
            ):
                section = local_score.get(section_name)
                score = section.get("score") if isinstance(section, dict) else None
                if not isinstance(score, dict):
                    continue
                score_percent = score.get("score_percent")
                if score_percent is None and "final_score" in score:
                    display_scale = float(score.get("display_scale", 1.0))
                    score_percent = float(score["final_score"]) / display_scale * 100.0
                if score_percent is not None:
                    lines.append(
                        f"{label}：{float(score_percent):.6f} 分"
                        f"（{protocol}，非官方榜分）"
                    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="煤气发电量预测与发电优化")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/official_preliminary.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"并行工作线程数（默认使用配置值，当前机器有 {os.cpu_count() or 1} 个逻辑处理器）",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="显示或关闭 tqdm 进度条",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="在终端输出完整结果 JSON；默认仅显示中文摘要",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("generate-synthetic", help="生成仅用于流程测试的合成数据")
    subparsers.add_parser("doctor", help="检查 LightGBM、CatBoost、Optuna 和 PyTorch CUDA")
    subparsers.add_parser("train", help="预处理、构建特征并训练模型")
    subparsers.add_parser("tune", help="运行 Optuna 粗筛、完整时间折复核和 OOF 融合")
    subparsers.add_parser("validate", help="执行时间滚动验证和泄漏检查")
    subparsers.add_parser("audit-data", help="执行时间语义和数据泄漏审计")
    subparsers.add_parser("benchmark", help="在统一时间折上比较强基线矩阵")
    subparsers.add_parser("discover-relations", help="发现确定性关系但不自动加入模型")
    subparsers.add_parser("backtest", help="执行扩展、滚动和连续两天回测")
    subparsers.add_parser("predict", help="生成并校验短周期和长周期结果")
    subparsers.add_parser("optimize", help="运行 HiGHS 调度并生成优化结果")
    subparsers.add_parser("validate-submission", help="联合校验提交文件并生成追踪清单")
    subparsers.add_parser("run", help="使用 data 中的现有数据执行完整预测任务")
    subparsers.add_parser("demo", help="依次运行合成数据到优化的完整流程")
    parser.set_defaults(command="run")
    return parser


def _pipelines() -> dict[str, Pipeline]:
    return {
        "generate-synthetic": generate_synthetic_pipeline,
        "doctor": environment_pipeline,
        "train": train_pipeline,
        "tune": tune_pipeline,
        "validate": validate_pipeline,
        "audit-data": audit_data_pipeline,
        "benchmark": benchmark_pipeline,
        "discover-relations": discover_relations_pipeline,
        "backtest": backtest_pipeline,
        "predict": predict_pipeline,
        "optimize": optimize_pipeline,
        "validate-submission": validate_submission_pipeline,
        "run": run_task_pipeline,
        "demo": demo_pipeline,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    validate_competition_command(config, str(args.command))
    workers, _ = configure_runtime(config, args.workers, args.progress)
    output_session = OutputSession.start(config)
    config.ensure_runtime_dirs()
    set_random_seed(config.seed)
    logging_config = config.section("logging")
    logger = configure_logging(
        config.path("logs"),
        level=str(logging_config.get("level", "INFO")),
        filename=str(logging_config.get("filename", "pipeline.log")),
    )
    pipeline = _pipelines()[str(args.command)]
    command = str(args.command)
    command_label = COMMAND_LABELS.get(command, command)
    logger.info("开始：%s（%d 个工作线程）", command_label, workers)
    try:
        with threadpool_limits(limits=workers):
            result = pipeline(config)
            archive_path = output_session.create_submission_archive(
                required=command in {"run", "demo", "predict"}
            )
            if archive_path is not None:
                result["submission_archive"] = str(archive_path)
    except Exception:
        logger.exception("失败：%s", command_label)
        final_directory, _ = output_session.finalize()
        logger.info("未完成结果已归档：%s", final_directory)
        raise
    final_directory, completed_at = output_session.finalize()
    result = output_session.relocate_result_paths(result, final_directory)
    result["output_directory"] = str(final_directory)
    result["completed_at"] = completed_at.isoformat()
    result_file = final_directory / "运行结果.json"
    result["result_file"] = str(result_file)
    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("完成：%s", command_label)
    if args.json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(_format_console_summary(command, result))


if __name__ == "__main__":
    main()
