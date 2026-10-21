"""codex_runner: gated, read-only, and never reaches a real Codex CLI. A fake `codex` on PATH stands in for it."""
import asyncio, os, stat
from pathlib import Path

import pytest

from app import facility


def _fake_codex(bin_dir: Path) -> None:
    exe = bin_dir / "codex"
    exe.write_text('#!/bin/sh\necho "ARGS:$*"\ncat\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FACILITY_ENABLE_PROVIDER_EXEC", raising=False)
    with pytest.raises(facility.ExecutionDisabled):
        asyncio.run(facility.codex_runner(tmp_path, "hi", False, "", 5))


def test_write_tasks_are_refused_even_when_exec_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    monkeypatch.setenv("FACILITY_ENABLE_WRITE_EXEC", "1")
    with pytest.raises(facility.ExecutionDisabled, match="read-only"):
        asyncio.run(facility.codex_runner(tmp_path, "hi", True, "", 5))


def test_read_only_run_uses_read_only_sandbox_and_returns_text(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_codex(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    res = asyncio.run(facility.codex_runner(tmp_path, "hello", False, "gpt-x", 10))
    assert "--sandbox read-only" in res["text"]
    assert "-m gpt-x" in res["text"]
    assert res["text"].endswith("hello")
    assert res["exit_code"] == 0


def test_run_task_gates_codex_before_the_fixture_scan(monkeypatch):
    monkeypatch.delenv("FACILITY_ENABLE_PROVIDER_EXEC", raising=False)
    task = {"id": "t1", "title": "t", "prompt": "p", "fixture": "readonly_probe", "permissions": "read_only",
            "allowed_paths": [], "verify": ["python", "-m", "pytest", "check_noop.py"], "timeout_s": 30}
    with pytest.raises(facility.ExecutionDisabled):
        asyncio.run(facility.run_task(task, runner=facility.codex_runner))


def test_task_runner_field_is_validated():
    base = {"id": "t1", "title": "t", "prompt": "p", "fixture": "readonly_probe", "permissions": "read_only",
            "verify": ["python", "-m", "pytest", "check_noop.py"]}
    facility.validate_task({**base, "runner": "chatgpt"})
    with pytest.raises(facility.TaskError, match="runner must be"):
        facility.validate_task({**base, "runner": "gemini"})


def test_codex_runner_cannot_be_requested_for_write_tasks():
    base = {"id": "t1", "title": "t", "prompt": "p", "fixture": "buggy_calc", "permissions": "write",
            "allowed_paths": ["calc.py"], "verify": ["python", "-m", "pytest", "test_calc.py"], "runner": "chatgpt"}
    with pytest.raises(facility.TaskError, match="read-only"):
        facility.validate_task(base)
