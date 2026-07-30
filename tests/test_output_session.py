from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from gas_power.config import ProjectConfig
from gas_power.output_session import OutputSession


def test_output_session_uses_completion_time_and_literal_suffix(tmp_path: Path) -> None:
    config = ProjectConfig(
        raw={"paths": {"root": str(tmp_path), "outputs": "outputs"}},
        source=tmp_path / "config.yaml",
        root=tmp_path,
    )
    session = OutputSession.start(config)
    config.ensure_runtime_dirs()
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "reports").exists()
    result_file = session.staging_directory / "s_result.csv"
    result_file.write_text("datetime,prediction\n", encoding="utf-8")
    input_file = session.staging_directory / "input.csv"
    input_file.write_text("datetime,feature\n", encoding="utf-8")
    archive_path = session.create_submission_archive(required=True)

    assert archive_path is not None
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["input.csv", "s_result.csv"]
        assert archive.read("input.csv") == input_file.read_bytes()
        assert archive.read("s_result.csv") == result_file.read_bytes()

    before = datetime.now().astimezone()
    final_directory, completed_at = session.finalize()
    after = datetime.now().astimezone()

    assert before <= completed_at <= after
    assert final_directory.parent == tmp_path / "outputs"
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}预测结果",
        final_directory.name,
    )
    assert (final_directory / "s_result.csv").exists()
    assert (final_directory / "提交压缩包.zip").exists()

    relocated = session.relocate_result_paths(
        {"prediction": str(result_file)}, final_directory
    )
    assert relocated["prediction"] == str(final_directory / "s_result.csv")
