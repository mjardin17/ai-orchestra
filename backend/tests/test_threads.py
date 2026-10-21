"""Shared-thread store: full transcript for every participant, join and leave between turns, validation."""
import sqlite3

import pytest

from app import threads


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    monkeypatch.setattr(threads, "DB", conn)
    conn.executescript("""
    create table threads(id text primary key, name text, created real);
    create table thread_participants(thread_id text, participant text, active int default 1, joined_seq int,
        primary key(thread_id, participant));
    create table thread_messages(id text primary key, thread_id text, seq int, role text, participant text, content text, created real);
    """)
    return conn


def test_late_joiner_sees_the_whole_transcript(db):
    tid = threads.create_thread("plan")
    threads.set_participant(tid, "claude", True)
    threads.add_message(tid, "user", "first question")
    threads.add_message(tid, "assistant", "claude's answer", "claude")
    threads.set_participant(tid, "chatgpt", True)
    threads.add_message(tid, "user", "follow-up")
    prompt = threads.prompt_for(tid, "chatgpt")
    assert "first question" in prompt and "claude's answer" in prompt and "follow-up" in prompt
    assert prompt.endswith("Answer the latest User message as Assistant (chatgpt).")


def test_leaving_stops_answers_but_keeps_history(db):
    tid = threads.create_thread("plan")
    threads.set_participant(tid, "claude", True)
    threads.set_participant(tid, "chatgpt", True)
    threads.set_participant(tid, "claude", False)
    assert threads.active_participants(tid) == ["chatgpt"]
    threads.add_message(tid, "user", "q")
    threads.add_message(tid, "assistant", "gpt says hi", "chatgpt")
    threads.set_participant(tid, "claude", True)           # rejoins
    assert threads.active_participants(tid) == ["claude", "chatgpt"]
    assert [m["participant"] for m in threads.transcript(tid) if m["role"] == "assistant"] == ["chatgpt"]


def test_validation(db):
    tid = threads.create_thread("plan")
    with pytest.raises(threads.ThreadError, match="participant must be"):
        threads.set_participant(tid, "gemini", True)
    with pytest.raises(threads.ThreadError, match="name a participant"):
        threads.add_message(tid, "assistant", "x")
    with pytest.raises(threads.ThreadError, match="no participant"):
        threads.add_message(tid, "user", "x", "claude")
    with pytest.raises(threads.ThreadError, match="unknown thread"):
        threads.transcript("nope")
    with pytest.raises(threads.ThreadError, match="required"):
        threads.create_thread("  ")


def test_prompt_needs_a_user_message_to_answer(db):
    tid = threads.create_thread("plan")
    with pytest.raises(threads.ThreadError, match="has none yet"):
        threads.prompt_for(tid, "chatgpt")
    threads.add_message(tid, "user", "q")
    threads.add_message(tid, "assistant", "a", "claude")
    assert "Answer the latest User message as Assistant (chatgpt)." in threads.prompt_for(tid, "chatgpt")
