from __future__ import annotations

from pathlib import Path

import pytest

from gas_power.submission_freeze import (
    SubmissionFreezeError,
    freeze_submission,
    verify_submission_freeze,
)


def _write_submission(directory: Path) -> None:
    (directory / "input.csv").write_text(
        "datetime,feature\n2025-05-01 00:00:00,1\n",
        encoding="utf-8",
    )
    (directory / "s_result.csv").write_text(
        "datetime,prediction\n2025-05-01 00:00:00,2\n",
        encoding="utf-8",
    )


def test_submission_freeze_detects_changes_after_freeze(tmp_path: Path) -> None:
    _write_submission(tmp_path)
    frozen = freeze_submission(tmp_path)

    verified = verify_submission_freeze(tmp_path)
    assert verified["files"] == frozen["files"]

    with (tmp_path / "s_result.csv").open("a", encoding="utf-8") as stream:
        stream.write("2025-05-01 00:15:00,3\n")

    with pytest.raises(SubmissionFreezeError, match="发生变化"):
        verify_submission_freeze(tmp_path)
