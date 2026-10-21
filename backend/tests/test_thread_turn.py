"""Thread turns: fake runners only (no provider call), in-memory thread store, and the HTTP token gate."""
import asyncio, sqlite3

import pytest
from fastapi.testclient import TestClient

from app import facility, thread_turn, threads
from app.main import app

TOKEN = "test-token-not-a-real-secret-0004"


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)  # the HTTP test runs the app in another thread
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    create table threads(id text primary key, name text, created real);
    create table thread_participants(thread_id text, participant text, active int default 1, joined_seq int,
        primary key(thread_id, participant));
    create table thread_messages(id text primary key, thread_id text, seq int, role text, participant text, content text, created real);
    """)
    monkeypatch.setattr(threads, "DB", conn)
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    return conn


def fake(text):
    async def run(ws, prompt, write, model, timeout):
        return {"text": text, "exit_code": 0, "prompt": prompt}
    return run


def failing(ws, prompt, write, model, timeout):
    async def run(ws, prompt, write, model, timeout):
        raise RuntimeError("quota exceeded")
    return run(ws, prompt, write, model, timeout)


def test_every_active_participant_answers_and_later_ones_see_earlier_answers(db):
    tid = threads.create_thread("plan")
    threads.set_participant(tid, "claude", True)
    threads.set_participant(tid, "chatgpt", True)
    results = asyncio.run(thread_turn.run_turn(tid, "hello", runners={"claude": fake("A"), "chatgpt": fake("B")}))
    assert [r["ok"] for r in results] == [True, True]
    seen = {m["participant"]: m["content"] for m in threads.transcript(tid) if m["role"] == "assistant"}
    assert seen == {"claude": "A", "chatgpt": "B"}


def test_one_failing_model_does_not_stop_the_others(db):
    tid = threads.create_thread("plan")
    threads.set_participant(tid, "claude", True)
    threads.set_participant(tid, "chatgpt", True)
    results = asyncio.run(thread_turn.run_turn(tid, "hi", runners={"claude": failing, "chatgpt": fake("B")}))
    by = {r["participant"]: r for r in results}
    assert by["claude"]["ok"] is False and "quota" in by["claude"]["error"]
    assert by["chatgpt"]["ok"] is True
    assert [m["participant"] for m in threads.transcript(tid) if m["role"] == "assistant"] == ["chatgpt"]


def test_disabled_provider_writes_nothing(db, monkeypatch):
    monkeypatch.delenv("FACILITY_ENABLE_PROVIDER_EXEC")
    tid = threads.create_thread("plan")
    threads.set_participant(tid, "claude", True)
    with pytest.raises(facility.ExecutionDisabled):
        asyncio.run(thread_turn.run_turn(tid, "hi", runners={"claude": fake("A")}))
    assert threads.transcript(tid) == []


def test_turn_needs_a_participant_and_content(db):
    tid = threads.create_thread("plan")
    with pytest.raises(threads.ThreadError, match="no active participants"):
        asyncio.run(thread_turn.run_turn(tid, "hi", runners={}))
    threads.set_participant(tid, "claude", True)
    with pytest.raises(threads.ThreadError, match="content is required"):
        asyncio.run(thread_turn.run_turn(tid, "   ", runners={}))


def test_http_routes_require_the_token(db, monkeypatch):
    monkeypatch.setenv("FACILITY_API_TOKEN", TOKEN)
    client = TestClient(app)
    assert client.post("/api/threads", json={"name": "x"}).status_code == 401
    assert client.post("/api/threads", json={"name": "x"}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    h = {"Authorization": f"Bearer {TOKEN}"}
    tid = client.post("/api/threads", json={"name": "x"}, headers=h).json()["id"]
    r = client.post(f"/api/threads/{tid}/participants", json={"participant": "claude", "active": True}, headers=h)
    assert r.json() == {"active": ["claude"]}
    assert client.post(f"/api/threads/{tid}/turn", json={"content": 5}, headers=h).status_code == 400
    assert client.post("/api/threads/nope/turn", json={"content": "hi"}, headers=h).status_code == 400


def test_get_thread_needs_the_token_and_returns_the_transcript(db, monkeypatch):
    monkeypatch.setenv("FACILITY_API_TOKEN", TOKEN)
    client = TestClient(app)
    h = {"Authorization": f"Bearer {TOKEN}"}
    tid = client.post("/api/threads", json={"name": "x"}, headers=h).json()["id"]
    threads.add_message(tid, "user", "hello")
    assert client.get(f"/api/threads/{tid}").status_code == 401
    body = client.get(f"/api/threads/{tid}", headers=h).json()
    assert body["messages"][0]["content"] == "hello" and body["active"] == []


def test_threads_page_is_served_and_renders_text_not_html():
    client = TestClient(app)
    res = client.get("/threads")
    assert res.status_code == 200 and "Facility threads" in res.text
    assert "innerHTML" not in res.text  # model output is rendered as text only
