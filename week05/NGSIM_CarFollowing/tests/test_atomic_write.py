"""Tests for the staged-write helpers that all pipeline stages rely on."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
    replace_file_atomically,
    write_csv_atomic,
    write_json_atomic,
)


def _stage(tmp_path: Path, name: str, content: str = "fresh content") -> Path:
    staged = tmp_path / name
    staged.write_text(content, encoding="utf-8")
    return staged


def test_replace_file_atomically_moves_staged_content(tmp_path: Path) -> None:
    destination = tmp_path / "result.txt"
    destination.write_text("stale content", encoding="utf-8")
    staged = _stage(tmp_path, "staged.txt")

    strategy = replace_file_atomically(staged, destination)

    assert strategy == "replace"
    assert destination.read_text(encoding="utf-8") == "fresh content"
    assert not staged.exists()


def test_transient_permission_error_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.txt"
    destination.write_text("stale content", encoding="utf-8")
    staged = _stage(tmp_path, "staged.txt")

    real_replace = os.replace
    calls: list[int] = []

    def flaky_replace(source: str, target: str) -> None:
        calls.append(1)
        if len(calls) <= 2:
            error = PermissionError(13, "Access is denied")
            error.winerror = 5  # type: ignore[attr-defined]
            raise error
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr("common.time.sleep", lambda _seconds: None)

    strategy = replace_file_atomically(staged, destination)

    assert strategy == "replace"
    assert len(calls) == 3
    assert destination.read_text(encoding="utf-8") == "fresh content"
    assert not staged.exists()


def test_permanent_lock_falls_back_to_in_place_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.txt"
    destination.write_text("stale content", encoding="utf-8")
    staged = _stage(tmp_path, "staged.txt")

    def always_locked(_source: str, _target: str) -> None:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(os, "replace", always_locked)
    monkeypatch.setattr("common.time.sleep", lambda _seconds: None)

    strategy = replace_file_atomically(staged, destination)

    assert strategy == "in_place_copy"
    assert destination.read_text(encoding="utf-8") == "fresh content"
    assert not staged.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_non_retryable_errors_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.txt"
    staged = _stage(tmp_path, "staged.txt")

    def broken_replace(_source: str, _target: str) -> None:
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(os, "replace", broken_replace)

    with pytest.raises(OSError):
        replace_file_atomically(staged, destination)


def test_write_helpers_replace_existing_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("old,columns\n1,2\n", encoding="utf-8")
    json_path = tmp_path / "summary.json"
    json_path.write_text('{"old": true}', encoding="utf-8")

    write_csv_atomic(pd.DataFrame({"a": [1.5], "b": [2.0]}), csv_path)
    write_json_atomic({"fresh": True, "nan": float("nan")}, json_path)

    rewritten = pd.read_csv(csv_path)
    assert list(rewritten.columns) == ["a", "b"]
    assert rewritten.iloc[0]["a"] == pytest.approx(1.5)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["fresh"] is True
    assert payload["nan"] is None
    assert not list(tmp_path.glob(".*.tmp"))
