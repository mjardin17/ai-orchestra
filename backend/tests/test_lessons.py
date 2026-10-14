import pytest
from app import lessons, store


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


def test_lesson_auto_activates_after_repeats():
    for _ in range(lessons.PROMOTE_AT):
        lid = lessons.record_failure("coder", "claude:haiku", "invalid JSON from model", "z")
    assert lessons.get(lid)["status"] == "active"


def test_pending_lessons_are_not_injected():
    lessons.record_failure("coder", "claude:haiku", "bad output", "z")
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_approved_lesson_is_injected_for_matching_agent_only():
    lid = lessons.record_failure("coder", "claude:haiku", "bad output", "z")
    lessons.set_status(lid, "active")
    assert "bad output" in lessons.with_lessons("SYSTEM", "coder")
    assert lessons.with_lessons("SYSTEM", "writer") == "SYSTEM"


def test_global_lesson_applies_to_every_agent():
    lid = lessons.add_lesson("Never retry a rejected refresh token; flag it for reconnect.", agent="*")
    lessons.set_status(lid, "active")
    assert "refresh token" in lessons.with_lessons("SYSTEM", "writer")


def test_dismissed_lesson_is_not_injected_and_not_reactivated():
    lid = lessons.record_failure("coder", "m", "oops", "z")
    lessons.set_status(lid, "dismissed")
    for _ in range(lessons.PROMOTE_AT + 1):
        lessons.record_failure("coder", "m", "oops", "z")
    assert lessons.get(lid)["status"] == "dismissed"
    assert lessons.with_lessons("SYSTEM", "coder") == "SYSTEM"


def test_injection_is_capped_and_ordered_by_frequency():
    for i in range(lessons.MAX_INJECTED + 3):
        lid = lessons.add_lesson(f"rule number {i}", agent="*")
        lessons.set_status(lid, "active")
    top = lessons.add_lesson("most common mistake", agent="*")
    lessons.set_status(top, "active")
    store.DB.execute("update lessons set seen_count=99 where id=?", (top,))
    store.DB.commit()
    out = lessons.with_lessons("SYSTEM", "writer")
    assert out.count("\n- ") == lessons.MAX_INJECTED
    assert "most common mistake" in out


def test_record_failure_never_raises_on_garbage():
    assert lessons.record_failure(None, None, None, None) is not None
