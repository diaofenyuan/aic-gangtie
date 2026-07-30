"""项目命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from gas_power.config import ProjectConfig, load_config
from gas_power.logging_utils import configure_logging, set_random_seed
from gas_power.output_session import OutputSession
from gas_power.pipeline import (
    audit_data_pipeline,
    backtest_pipeline,
    benchmark_pipeline,
    demo_pipeline,
    discover_relations_pipeline,
    generate_synthetic_pipeline,
    optimize_pipeline,
    predict_pipeline,
    run_task_pipeline,
    train_pipeline,
    validate_pipeline,
    validate_submission_pipeline,
)
from gas_power.runtime import configure_runtime


Pipeline = Callable[[ProjectConfig], dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="煤气发电量预测与发电优化")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
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
    subparsers = parser.add_subparsers(dest="command", required=False)
    subparsers.add_parser("generate-synthetic", help="生成仅用于流程测试的合成数据")
    subparsers.add_parser("train", help="预处理、构建特征并训练模型")
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
        "train": train_pipeline,
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
    workers, show_progress = configure_runtime(config, args.workers, args.progress)
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
    logger.info("开始命令: %s，工作线程: %d", args.command, workers)
    try:
        with threadpool_limits(limits=workers):
            with tqdm(
                total=1,
                desc=f"任务 {args.command}",
                unit="项",
                dynamic_ncols=True,
                disable=not show_progress,
            ) as task_progress:
                result = pipeline(config)
                archive_path = output_session.create_submission_archive(
                    required=str(args.command) in {"run", "demo", "predict"}
                )
                if archive_path is not None:
                    result["submission_archive"] = str(archive_path)
                task_progress.update(1)
    except Exception:
        logger.exception("命令失败: %s", args.command)
        final_directory, _ = output_session.finalize()
        logger.info("未完成结果已归档: %s", final_directory)
        raise
    final_directory, completed_at = output_session.finalize()
    result = output_session.relocate_result_paths(result, final_directory)
    result["output_directory"] = str(final_directory)
    result["completed_at"] = completed_at.isoformat()
    logger.info("命令完成: %s", args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
