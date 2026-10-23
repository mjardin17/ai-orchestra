"""Facility task loop: validate -> isolated workspace -> OS lock -> agent -> verify by junit -> receipt -> candidate lessons.
The agent is always faked here (no claude CLI, no network, no model credits)."""
import asyncio, json, os
from pathlib import Path
import pytest
from app import facility, facility_sandbox as sb, lessons, store

FIX = facility.FIXTURE_ROOT
FIXED = "def average(values):\n    return sum(values) / len(values)\n"


def task(**over):
    t = {"id": "t1", "title": "Fix average()", "prompt": "Fix the bug in calc.py", "fixture": "buggy_calc",
         "permissions": "write", "allowed_paths": ["calc.py"],
         "verify": ["python", "-m", "pytest", "check_calc.py"], "timeout_s": 30}
    t.update(over)
    return t


def ro_task(**over):
    t = {"id": "ro1", "title": "Count lines", "prompt": "Count lines in notes.txt", "fixture": "readonly_probe",
         "permissions": "read_only", "verify": ["python", "-m", "pytest", "check_noop.py"], "timeout_s": 30}
    t.update(over)
    return t


def go(coro):
    return asyncio.run(coro)


def snapshot(root: Path) -> dict:
    return sb.scan(root)["files"]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("FACILITY_WORKSPACE_ROOT", str(tmp_path / "ws"))
    for k in ("FACILITY_ENABLE_PROVIDER_EXEC", "FACILITY_ENABLE_WRITE_EXEC", "FACILITY_MAX_CONCURRENT", "FACILITY_MAX_PER_HOUR"):
        monkeypatch.delenv(k, raising=False)
    facility._active = 0
    facility._starts.clear()
    store.DB.execute("delete from lessons"); store.DB.commit()
    yield
    facility._active = 0
    facility._starts.clear()


def ws_root_is_empty() -> bool:
    return not any(sb.workspace_root().iterdir())


async def fixer(ws, prompt, write, model, timeout):
    (ws / "calc.py").write_text(FIXED)
    return {"text": "fixed", "exit_code": 0}


async def noop(ws, prompt, write, model, timeout):
    return {"text": "did nothing", "exit_code": 0}


# ---- outcomes ---------------------------------------------------------------------------------------------------
def test_verified_when_agent_fixes_and_tests_pass():
    r = go(facility.run_task(task(), runner=fixer))
    assert r["status"] == "verified", r["error"]
    assert r["verify"]["exit_code"] == 0 and r["verify"]["report_ok"] is True
    assert r["verify"]["report"]["tests"] >= 1
    assert r["diff"]["changed"] == ["calc.py"]
    assert r["model"] == "claude:sonnet" and r["permissions"] == "write"
    assert r["candidate_lessons"] == []
    saved = json.loads(Path(r["receipt_path"]).read_text())
    assert saved["task_id"] == "t1" and saved["status"] == "verified"


def test_source_fixture_is_never_modified():
    before = snapshot(FIX / "buggy_calc")
    go(facility.run_task(task(), runner=fixer))
    assert snapshot(FIX / "buggy_calc") == before


def test_failed_verification_records_pending_unverified_lesson():
    r = go(facility.run_task(task(), runner=noop))
    assert r["status"] == "verification_failed" and r["verify"]["exit_code"] != 0
    (cand,) = r["candidate_lessons"]
    row = lessons.get(cand["id"])
    assert row["status"] == "pending" and row["verified"] == 0
    assert lessons.injectable_for("facility") == []


def test_read_only_write_attempt_is_blocked_by_os_lock():
    async def sneaky(ws, prompt, write, model, timeout):
        (ws / "notes.txt").write_text("tampered")       # PermissionError: the OS lock, not the prompt, stops it
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(ro_task(), runner=sneaky))
    assert r["status"] == "agent_error"
    assert "denied" in r["error"].lower() or "permission" in r["error"].lower()
    assert r["diff"]["changed"] == [] and r["verify"] is None
    assert len(r["candidate_lessons"]) == 1


def test_read_only_lock_blocks_new_files_too():
    async def creator(ws, prompt, write, model, timeout):
        (ws / "new.txt").write_text("x")
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(ro_task(), runner=creator))
    assert r["status"] == "agent_error" and r["diff"]["added"] == []


def test_read_only_bypass_that_unlocks_first_is_a_permission_violation():
    async def bypass(ws, prompt, write, model, timeout):
        sb.unlock_tree(ws)                                # simulates a lock bypass; the diff check is the second line of defence
        (ws / "notes.txt").write_text("tampered")
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(ro_task(), runner=bypass))
    assert r["status"] == "permission_violation"
    assert r["verify"] is None and r["diff"]["changed"] == ["notes.txt"]
    assert len(r["candidate_lessons"]) == 1


def test_read_only_passes_when_untouched():
    async def reader(ws, prompt, write, model, timeout):
        assert write is False
        assert (ws / "notes.txt").read_text()             # reads still work under the lock
        return {"text": "1 line", "exit_code": 0}
    r = go(facility.run_task(ro_task(), runner=reader))
    assert r["status"] == "verified", r["error"]
    assert r["model"] == "claude:haiku"


def test_changes_outside_allowed_paths_are_a_permission_violation():
    async def wanderer(ws, prompt, write, model, timeout):
        (ws / "calc.py").write_text(FIXED)
        (ws / "extra.txt").write_text("not allowed")      # new file: cannot be pre-locked, must be caught by the diff
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(task(), runner=wanderer))
    assert r["status"] == "permission_violation"
    assert "extra.txt" in r["error"] and r["verify"] is None


def test_write_task_cannot_modify_locked_test_file():
    async def tamper(ws, prompt, write, model, timeout):
        (ws / "check_calc.py").write_text("def test_x():\n    assert True\n")
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(task(), runner=tamper))
    assert r["status"] == "agent_error"                  # locked file: PermissionError
    assert snapshot(FIX / "buggy_calc")["check_calc.py"] != ""


def test_unlocked_test_tampering_is_a_protected_violation_never_verified():
    async def tamper(ws, prompt, write, model, timeout):
        sb.unlock_tree(ws)
        (ws / "check_calc.py").write_text("def test_x():\n    assert True\n")
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(task(), runner=tamper))
    assert r["status"] == "protected_file_violation" and r["status"] != "verified"


def test_timeout_is_recorded_without_hanging():
    async def slow(ws, prompt, write, model, timeout):
        await asyncio.sleep(30)
    r = go(asyncio.wait_for(facility.run_task(task(timeout_s=2), runner=slow), 20))
    assert r["status"] == "timeout" and r["verify"] is None
    assert len(r["candidate_lessons"]) == 1
    assert lessons.get(r["candidate_lessons"][0]["id"])["status"] == "pending"
    assert r["workspace_deleted"] is True


def test_agent_crash_is_recorded_not_raised():
    async def boom(ws, prompt, write, model, timeout):
        raise RuntimeError("provider exploded")
    r = go(facility.run_task(task(), runner=boom))
    assert r["status"] == "agent_error" and "provider exploded" in r["error"]
    assert len(r["candidate_lessons"]) == 1


# ---- lessons ----------------------------------------------------------------------------------------------------
def test_lessons_are_injected_only_when_active_and_verified():
    lid = lessons.add_lesson("Always read the test first.", agent="facility")
    seen = {}

    async def spy(ws, prompt, write, model, timeout):
        seen["prompt"] = prompt
        return {"text": "", "exit_code": 0}
    go(facility.run_task(task(), runner=spy))
    assert "Always read the test first" not in seen["prompt"]            # pending
    lessons.set_status(lid, "active")
    r = go(facility.run_task(task(), runner=spy))
    assert "Always read the test first" not in seen["prompt"] and r["lessons_injected"] == []   # active, not verified
    store.DB.execute("update lessons set verified=1 where id=?", (lid,)); store.DB.commit()
    r = go(facility.run_task(task(), runner=spy))
    assert "Always read the test first" in seen["prompt"] and r["lessons_injected"] == [lid]
    lessons.set_status(lid, "pending")
    seen.clear()
    r = go(facility.run_task(task(), runner=spy))
    assert "Always read the test first" not in seen["prompt"]            # verified but no longer active


# ---- validation -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"fixture": "../etc"}, {"fixture": ".."}, {"fixture": "nope"}, {"permissions": "root"}, {"id": ""},
    {"verify": ["rm", "-rf", "/"]}, {"verify": ["python", "-m", "pytest"]},
    {"verify": ["python", "-m", "pytest", "../x.py"]}, {"verify": ["python", "-m", "pytest", "/abs.py"]},
    {"verify": ["python", "-m", "pytest", "--rootdir=/"]}, {"verify": ["python", "-m", "pytest", "-p", "evil"]},
    {"timeout_s": 99999}, {"timeout_s": 0}, {"timeout_s": True}, {"verify_timeout_s": 99999},
    {"allowed_paths": []}, {"allowed_paths": None},
    {"allowed_paths": ["check_calc.py"]}, {"allowed_paths": ["conftest.py"]}, {"allowed_paths": ["calc.py", "pytest.ini"]},
    {"allowed_paths": ["../x"]}, {"allowed_paths": ["/etc/x"]}, {"allowed_paths": ["a/../x"]}, {"allowed_paths": ["dir/"]},
    {"allowed_paths": "calc.py"},
])
def test_invalid_write_tasks_are_rejected(bad):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(**bad))


def test_read_only_with_allowed_paths_is_rejected():
    with pytest.raises(facility.TaskError):
        facility.validate_task(ro_task(allowed_paths=["notes.txt"]))


def test_non_object_task_is_rejected():
    with pytest.raises(facility.TaskError):
        facility.validate_task(["not", "a", "dict"])


def test_valid_tasks_are_accepted():
    assert facility.validate_task(task())["id"] == "t1"
    assert facility.validate_task(ro_task())["permissions"] == "read_only"


# ---- receipt + cleanup ------------------------------------------------------------------------------------------
def test_receipt_has_evidence_fields():
    r = go(facility.run_task(task(), runner=fixer))
    assert r["resolved_model"]["provider"] == "claude-cli" and r["resolved_model"]["cli_model"]
    assert r["run_id"] and r["files_after"]["calc.py"] and "check_calc.py" in r["files_after"]
    assert r["workspace_deleted"] is True and r["cleanup_error"] is None
    saved = json.loads(Path(r["receipt_path"]).read_text())
    assert saved["run_id"] == r["run_id"] and saved["resolved_model"]["provider"] == "claude-cli"


def test_same_task_id_twice_writes_two_receipts():
    a = go(facility.run_task(task(), runner=fixer))
    b = go(facility.run_task(task(), runner=fixer))
    assert a["run_id"] != b["run_id"] and a["receipt_path"] != b["receipt_path"]
    assert Path(a["receipt_path"]).exists() and Path(b["receipt_path"]).exists()
    assert len(list(Path(os.environ["FACILITY_RECEIPT_DIR"]).glob("t1-*.json"))) == 2


def test_workspace_root_is_clean_after_success():
    go(facility.run_task(task(), runner=fixer))
    assert ws_root_is_empty()


def test_workspace_root_is_clean_after_failure():
    go(facility.run_task(task(), runner=noop))
    assert ws_root_is_empty()
    async def boom(ws, prompt, write, model, timeout):
        raise RuntimeError("crash")
    go(facility.run_task(ro_task(), runner=boom))
    assert ws_root_is_empty()


def test_workspace_root_is_clean_after_timeout():
    async def slow(ws, prompt, write, model, timeout):
        await asyncio.sleep(30)
    go(facility.run_task(task(timeout_s=2), runner=slow))
    assert ws_root_is_empty()


# ---- admission control ------------------------------------------------------------------------------------------
def test_admit_rejects_when_concurrency_exceeded(monkeypatch):
    monkeypatch.setenv("FACILITY_MAX_CONCURRENT", "1")
    facility.admit()
    with pytest.raises(facility.Busy):
        facility.admit()
    facility.release()
    facility.admit()                                    # slot freed
    facility.release()


def test_release_never_goes_negative():
    facility.release(); facility.release()
    assert facility._active == 0


def test_admit_rejects_when_hourly_cap_exceeded(monkeypatch):
    monkeypatch.setenv("FACILITY_MAX_CONCURRENT", "50")
    monkeypatch.setenv("FACILITY_MAX_PER_HOUR", "2")
    facility.admit(); facility.release()
    facility.admit(); facility.release()
    with pytest.raises(facility.Busy):
        facility.admit()


def test_run_guarded_releases_slot_even_when_task_fails(monkeypatch):
    monkeypatch.setenv("FACILITY_MAX_CONCURRENT", "1")
    with pytest.raises(facility.TaskError):
        go(facility.run_guarded(task(fixture="nope")))
    assert facility._active == 0
    assert go(facility.run_guarded(task(), runner=fixer))["status"] == "verified"
    assert facility._active == 0


# ---- execution gates --------------------------------------------------------------------------------------------
def test_ensure_exec_enabled_is_off_by_default():
    with pytest.raises(facility.ExecutionDisabled):
        facility.ensure_exec_enabled(False)
    with pytest.raises(facility.ExecutionDisabled):
        facility.ensure_exec_enabled(True)


def test_write_exec_needs_its_own_flag(monkeypatch):
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    facility.ensure_exec_enabled(False)                 # read-only allowed
    with pytest.raises(facility.ExecutionDisabled):
        facility.ensure_exec_enabled(True)
    monkeypatch.setenv("FACILITY_ENABLE_WRITE_EXEC", "1")
    facility.ensure_exec_enabled(True)
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "0")
    with pytest.raises(facility.ExecutionDisabled):
        facility.ensure_exec_enabled(True)


@pytest.mark.parametrize("t", [task(), ro_task()])
def test_real_runner_disabled_raises_before_any_workspace(t):
    assert facility.DEFAULT_RUNNER is facility.claude_runner
    root = Path(os.environ["FACILITY_WORKSPACE_ROOT"])
    with pytest.raises(facility.ExecutionDisabled):
        go(facility.run_task(t, runner=None))
    assert not root.exists() or not any(root.iterdir())
    assert not Path(os.environ["FACILITY_RECEIPT_DIR"]).exists()


def test_real_runner_write_flag_missing_still_blocks_write_task(monkeypatch):
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    root = Path(os.environ["FACILITY_WORKSPACE_ROOT"])
    with pytest.raises(facility.ExecutionDisabled):
        go(facility.run_task(task(), runner=None))      # raises before spawning anything
    assert not root.exists() or not any(root.iterdir())
