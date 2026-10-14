"""Facility task loop: validate -> isolated workspace -> agent -> verify by command -> receipt -> candidate lessons.
The agent is faked here, so these tests spend no model credits."""
import asyncio, json
import pytest
from fastapi.testclient import TestClient
from app import facility, lessons, store
from app.main import app

FIX = facility.FIXTURE_ROOT


def task(**over):
    t = {"id": "t1", "title": "Fix average()", "prompt": "Fix the bug in calc.py", "fixture": "buggy_calc",
         "permissions": "write", "verify": ["python", "-m", "pytest", "check_calc.py"], "timeout_s": 30}
    t.update(over)
    return t


def go(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_RECEIPT_DIR", str(tmp_path / "receipts"))
    store.DB.execute("delete from lessons"); store.DB.commit()


async def fixer(ws, prompt, write, model, timeout):
    (ws / "calc.py").write_text("def average(values):\n    return sum(values) / len(values)\n")
    return {"text": "fixed", "exit_code": 0}


async def noop(ws, prompt, write, model, timeout):
    return {"text": "did nothing", "exit_code": 0}


def test_verified_when_agent_fixes_and_tests_pass():
    r = go(facility.run_task(task(), runner=fixer))
    assert r["status"] == "verified"
    assert r["verify"]["exit_code"] == 0
    assert r["diff"]["changed"] == ["calc.py"]
    assert r["model"] == "claude:sonnet" and r["permissions"] == "write"
    assert r["candidate_lessons"] == []
    saved = json.loads(open(r["receipt_path"]).read())
    assert saved["task_id"] == "t1" and saved["status"] == "verified"


def test_source_fixture_is_never_modified():
    before = facility.snapshot(FIX / "buggy_calc")
    go(facility.run_task(task(), runner=fixer))
    assert facility.snapshot(FIX / "buggy_calc") == before


def test_failed_verification_records_pending_unverified_lesson():
    r = go(facility.run_task(task(), runner=noop))
    assert r["status"] == "verification_failed" and r["verify"]["exit_code"] != 0
    (cand,) = r["candidate_lessons"]
    row = lessons.get(cand["id"])
    assert row["status"] == "pending" and row["verified"] == 0
    assert lessons.injectable_for("facility") == []


def test_read_only_write_attempt_is_blocked():
    async def sneaky(ws, prompt, write, model, timeout):
        (ws / "notes.txt").write_text("tampered")
        return {"text": "ok", "exit_code": 0}
    r = go(facility.run_task(task(fixture="readonly_probe", permissions="read_only",
                                  verify=["python", "-m", "pytest", "check_noop.py"]), runner=sneaky))
    assert r["status"] == "permission_violation"
    assert r["verify"] is None and r["diff"]["changed"] == ["notes.txt"]


def test_read_only_passes_when_untouched():
    async def reader(ws, prompt, write, model, timeout):
        assert write is False
        return {"text": "1 line", "exit_code": 0}
    r = go(facility.run_task(task(fixture="readonly_probe", permissions="read_only",
                                  verify=["python", "-m", "pytest", "check_noop.py"]), runner=reader))
    assert r["status"] == "verified" and r["model"] == "claude:haiku"


def test_timeout_is_recorded_and_agent_is_cancelled():
    async def slow(ws, prompt, write, model, timeout):
        await asyncio.sleep(10)
    r = go(facility.run_task(task(timeout_s=1), runner=slow))
    assert r["status"] == "timeout" and r["verify"] is None
    assert len(r["candidate_lessons"]) == 1


def test_agent_crash_is_recorded_not_raised():
    async def boom(ws, prompt, write, model, timeout):
        raise RuntimeError("provider exploded")
    r = go(facility.run_task(task(), runner=boom))
    assert r["status"] == "agent_error" and "provider exploded" in r["error"]


def test_verified_active_lessons_are_injected_into_the_prompt():
    lid = lessons.add_lesson("Always read the test first.", agent="facility")
    lessons.set_status(lid, "active")
    seen = {}

    async def spy(ws, prompt, write, model, timeout):
        seen["prompt"] = prompt
        return {"text": "", "exit_code": 0}
    r = go(facility.run_task(task(), runner=spy))
    assert "Always read the test first" not in seen["prompt"]       # approved but unverified: not injected
    assert r["lessons_injected"] == []
    store.DB.execute("update lessons set verified=1 where id=?", (lid,)); store.DB.commit()
    r = go(facility.run_task(task(), runner=spy))
    assert "Always read the test first" in seen["prompt"] and r["lessons_injected"] == [lid]


@pytest.mark.parametrize("bad", [
    {"fixture": "../etc"}, {"fixture": "nope"}, {"permissions": "root"}, {"id": ""},
    {"verify": ["rm", "-rf", "/"]}, {"verify": ["python", "-m", "pytest", "../x.py"]},
    {"verify": ["python", "-m", "pytest", "/abs.py"]}, {"verify": ["python", "-m", "pytest", "--rootdir=/"]},
    {"timeout_s": 99999},
])
def test_invalid_tasks_are_rejected(bad):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(**bad))


def test_api_endpoint_runs_with_injected_runner(monkeypatch):
    monkeypatch.setattr(facility, "DEFAULT_RUNNER", fixer)
    c = TestClient(app)
    r = c.post("/api/tasks/run", json=task())
    assert r.status_code == 200 and r.json()["status"] == "verified"
    assert c.post("/api/tasks/run", json=task(fixture="../x")).status_code == 400
