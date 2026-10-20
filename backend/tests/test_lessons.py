import pytest
from app import lessons, store


def verify(lid):
    """Stand-in for the future verification pipeline (P6): marks a lesson verified directly in the database."""
    store.DB.execute("update lessons set verified=1 where id=?", (lid,))
    store.DB.commit()


@pytest.fixture(autouse=True)
def clean():
    store.DB.execute("delete from lessons")
    store.DB.commit()


def test_record_failure_creates_pending_lesson():
    lid = lessons.record_failure("research", "codex:gpt", "HTTP 401 token expired", "find comps")
    row = lessons.get(lid)
    assert row["status"] == "pending"
    assert row["seen_count"] == 1
    assert "401" in row["rule"]


def test_same_failure_is_deduplicated_and_counted():
    a = lessons.record_failure("research", "codex:gpt", "HTTP 401 token expired at 12:01", "x")
    b = lessons.record_failure("research", "codex:gpt", "HTTP 401 token expired at 13:44", "y")
    assert a == b
    assert lessons.get(a)["seen_count"] == 2


def test_repeated_failure_is_counted_but_never_promoted():
    """A repeated failure is a failure mode, not a verified rule (the old behaviour made it ACTIVE after 3 repeats)."""
    for _ in range(25):
        lid = lessons.record_failure("coder", "claude:haiku", "invalid JSON from model", "z")
    row = lessons.get(lid)
    assert row["seen_count"] == 25
    assert row["status"] == "pending" and row["verified"] == 0 and row["kind"] == "failure_mode"
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_pending_lessons_are_not_injected():
    lessons.record_failure("coder", "claude:haiku", "bad output", "z")
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_verified_approved_lesson_is_injected_for_matching_agent_only():
    lid = lessons.record_failure("coder", "claude:haiku", "bad output", "z")
    lessons.set_status(lid, "active")
    verify(lid)
    assert "bad output" in lessons.with_lessons("SYSTEM", "coder")
    assert lessons.with_lessons("SYSTEM", "writer") == "SYSTEM"


def test_global_lesson_applies_to_every_agent():
    lid = lessons.add_lesson("Never retry a rejected refresh token; flag it for reconnect.", agent="*")
    lessons.set_status(lid, "active")
    verify(lid)
    assert "refresh token" in lessons.with_lessons("SYSTEM", "writer")


def test_dismissed_lesson_is_not_injected_and_not_reactivated():
    lid = lessons.record_failure("coder", "m", "oops", "z")
    lessons.set_status(lid, "dismissed")
    for _ in range(5):
        lessons.record_failure("coder", "m", "oops", "z")
    assert lessons.get(lid)["status"] == "dismissed"
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_injection_is_capped_and_ordered_by_frequency():
    for i in range(lessons.MAX_INJECTED + 3):
        lid = lessons.add_lesson(f"rule number {i}", agent="*")
        lessons.set_status(lid, "active")
        verify(lid)
    top = lessons.add_lesson("most common mistake", agent="*")
    lessons.set_status(top, "active")
    verify(top)
    store.DB.execute("update lessons set seen_count=99 where id=?", (top,))
    store.DB.commit()
    out = lessons.with_lessons("SYSTEM", "writer")
    assert out.count("\n- ") == lessons.MAX_INJECTED
    assert "most common mistake" in out


def test_record_failure_never_raises_on_garbage():
    assert lessons.record_failure(None, None, None, None) is not None


def test_approved_but_unverified_lesson_is_stored_but_never_injected():
    lid = lessons.record_failure("coder", "claude:haiku", "bad output", "z")
    assert lessons.set_status(lid, "active")
    row = lessons.get(lid)
    assert row["status"] == "active" and row["verified"] == 0 and row["approved_at"]   # approval recorded, verification NOT claimed
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"
    assert lessons.injectable_for("coder") == []
    assert [x["id"] for x in lessons.active_for("coder")] == [lid]                       # still stored and listable
    assert lessons.get(lid)["rule"]                                                      # history preserved


def test_verified_but_not_approved_lesson_is_not_injected_either():
    lid = lessons.record_failure("coder", "claude:haiku", "bad output", "z")
    verify(lid)
    assert lessons.get(lid)["status"] == "pending"
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_dismissed_verified_lesson_is_not_injected():
    lid = lessons.add_lesson("rule", "*")
    verify(lid)
    lessons.set_status(lid, "dismissed")
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_only_verified_and_approved_lessons_reach_the_prompt_when_mixed():
    ok = lessons.add_lesson("VERIFIED-RULE", "*")
    unverified = lessons.add_lesson("UNVERIFIED-APPROVED", "*")
    pending = lessons.add_lesson("PENDING-ONLY", "*")
    for lid in (ok, unverified):
        lessons.set_status(lid, "active")
    verify(ok)
    out = lessons.with_lessons("SYSTEM", "coder")
    assert "VERIFIED-RULE" in out and "Verified rules" in out
    assert "UNVERIFIED-APPROVED" not in out and "PENDING-ONLY" not in out and "Unverified" not in out


def test_repeating_and_approving_never_flips_verified():
    lid = lessons.record_failure("coder", "m", "same failure", "z")
    for _ in range(10):
        lessons.record_failure("coder", "m", "same failure", "z")
    lessons.set_status(lid, "active")
    assert lessons.get(lid)["verified"] == 0 and lessons.get(lid)["seen_count"] == 11


def test_council_solve_loop_never_shows_an_agent_an_unverified_lesson(monkeypatch):
    """Drives the real orchestrator.solve() and captures the system prompt the model actually receives."""
    import asyncio
    from app import models, orchestrator
    bad = lessons.add_lesson("APPROVED-BUT-UNVERIFIED", "*")
    good = lessons.add_lesson("APPROVED-AND-VERIFIED", "*")
    for lid in (bad, good):
        lessons.set_status(lid, "active")
    verify(good)
    seen = []

    async def fake_run(m, system, prompt, tools=False):
        seen.append(system)
        return "answer"
    monkeypatch.setattr(models, "run_strict", fake_run)
    pool = [{"id": "claude:haiku", "label": "H", "kind": "claude", "tier": "fast", "power": 4, "cost": 1}]

    async def go():
        return [e async for e in orchestrator.solve("writer", "writer", {"difficulty": {}}, pool, "SYS", "p", "req", None)]
    asyncio.run(go())
    assert seen and "APPROVED-AND-VERIFIED" in seen[0] and "APPROVED-BUT-UNVERIFIED" not in seen[0]


def test_legacy_active_lessons_read_back_as_unverified(tmp_path):
    """A database created before the new columns existed: its ACTIVE lessons must come back unverified and stay readable."""
    import subprocess, sys, sqlite3, os
    db = tmp_path / "orchestra.db"
    c = sqlite3.connect(db)
    c.executescript("""create table lessons(id text primary key, signature text unique, agent text, model text, rule text,
        status text default 'pending', seen_count int default 1, created real, last_seen real);
        insert into lessons values('old1','sig','coder','m','old auto-promoted rule','active',3,1,1);""")
    c.commit(); c.close()
    code = ("from app import lessons; r = lessons.get('old1'); "
            "print(r['status'], r['verified'], r['provenance']); print('old auto-promoted rule' in lessons.with_lessons('S','coder')); print(lessons.get('old1')['rule'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(__import__("pathlib").Path(__file__).resolve().parents[1])))
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == "active 0 legacy_unverified"
    assert lines[1] == "False"                      # legacy ACTIVE lessons are preserved but not injected
    assert lines[2] == "old auto-promoted rule"
