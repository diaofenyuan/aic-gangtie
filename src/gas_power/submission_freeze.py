"""冻结并校验初赛提交文件。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUBMISSION_FILES = ("input.csv", "s_result.csv")
FREEZE_MANIFEST = "submission_freeze.json"


class SubmissionFreezeError(ValueError):
    """提交文件与冻结清单不一致。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_submission(directory: Path) -> dict[str, Any]:
    """记录两个提交文件的 SHA-256 和字节数。"""

    root = Path(directory)
    files: dict[str, dict[str, Any]] = {}
    for name in SUBMISSION_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"无法冻结提交，缺少文件: {path}")
        files[name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    payload: dict[str, Any] = {
        "status": "FROZEN_SUBMISSION",
        "algorithm": "sha256",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path = root / FREEZE_MANIFEST
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_submission_freeze(directory: Path) -> dict[str, Any]:
    """校验当前提交文件仍与冻结清单逐字节一致。"""

    root = Path(directory)
    manifest_path = root / FREEZE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"提交冻结清单不存在: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SubmissionFreezeError(f"提交冻结清单无效: {manifest_path}") from exc

    if payload.get("status") != "FROZEN_SUBMISSION":
        raise SubmissionFreezeError("提交冻结清单状态无效")
    if payload.get("algorithm") != "sha256":
        raise SubmissionFreezeError("提交冻结清单仅允许 sha256 算法")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != set(SUBMISSION_FILES):
        raise SubmissionFreezeError("提交冻结清单中的文件集合不正确")

    for name in SUBMISSION_FILES:
        path = root / name
        expected = files.get(name)
        if not path.is_file():
            raise SubmissionFreezeError(f"冻结后的提交文件缺失: {name}")
        if not isinstance(expected, dict):
            raise SubmissionFreezeError(f"提交冻结记录无效: {name}")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_size != int(expected.get("size_bytes", -1)):
            raise SubmissionFreezeError(f"冻结后的提交文件大小发生变化: {name}")
        if actual_hash != str(expected.get("sha256", "")):
            raise SubmissionFreezeError(f"冻结后的提交文件哈希发生变化: {name}")
    return payload
