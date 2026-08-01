"""为每次命令创建独立的结果目录，并在结束时按完成时间归档。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from gas_power.config import ProjectConfig
from gas_power.submission_freeze import FREEZE_MANIFEST, verify_submission_freeze


@dataclass
class OutputSession:
    """管理单次运行的临时目录和最终结果目录。"""

    output_root: Path
    staging_directory: Path
    archive_name: str

    @classmethod
    def start(cls, config: ProjectConfig) -> "OutputSession":
        output_root = config.path("outputs", "outputs")
        output_root.mkdir(parents=True, exist_ok=True)
        staging_directory = output_root / f".running-{uuid4().hex}"
        staging_directory.mkdir(parents=False, exist_ok=False)

        submission = config.raw.get("submission", {})
        archive_name = (
            str(submission.get("archive_name", "teamname_gas_predict_prelim.zip"))
            if isinstance(submission, dict)
            else "teamname_gas_predict_prelim.zip"
        )
        if Path(archive_name).name != archive_name or not archive_name.lower().endswith(
            ".zip"
        ):
            raise ValueError("submission.archive_name 必须是不含目录的 .zip 文件名")

        paths = config.section("paths")
        paths["results"] = str(staging_directory)
        paths["reports"] = str(staging_directory / "reports")
        return cls(
            output_root=output_root,
            staging_directory=staging_directory,
            archive_name=archive_name,
        )

    def finalize(self) -> tuple[Path, datetime]:
        """用运行结束时刻生成最终目录名，并原子移动本次结果。"""

        while True:
            completed_at = datetime.now().astimezone()
            timestamp = completed_at.strftime("%Y-%m-%d_%H-%M-%S")
            final_directory = self.output_root / f"{timestamp}预测结果"
            if not final_directory.exists():
                break
            # 同一秒已有结果时等待进入下一秒，目录名仍只保留结束时间。
            sleep(0.01)
        self.staging_directory.rename(final_directory)
        return final_directory, completed_at

    def create_submission_archive(self, *, required: bool = False) -> Path | None:
        """把初赛要求的 input.csv 和 s_result.csv 写入压缩包根目录。"""

        submission_files = ("input.csv", "s_result.csv")
        missing = [
            name
            for name in submission_files
            if not (self.staging_directory / name).is_file()
        ]
        if missing:
            if required:
                raise FileNotFoundError(
                    f"无法生成提交压缩包，缺少文件: {', '.join(missing)}"
                )
            return None

        # 打包前校验冻结清单，保证提交文件在预测完成后未被改写。
        if (self.staging_directory / FREEZE_MANIFEST).is_file():
            verify_submission_freeze(self.staging_directory)

        archive_path = self.staging_directory / self.archive_name
        with ZipFile(
            archive_path,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in submission_files:
                archive.write(self.staging_directory / name, arcname=name)
        return archive_path

    def relocate_result_paths(self, value: Any, final_directory: Path) -> Any:
        """把返回结果中的临时路径替换为最终归档路径。"""

        staging = str(self.staging_directory)
        final = str(final_directory)
        if isinstance(value, str):
            return final + value[len(staging) :] if value.startswith(staging) else value
        if isinstance(value, dict):
            return {
                key: self.relocate_result_paths(item, final_directory)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.relocate_result_paths(item, final_directory) for item in value]
        if isinstance(value, tuple):
            return tuple(
                self.relocate_result_paths(item, final_directory) for item in value
            )
        return value
