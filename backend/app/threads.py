"""Shared Facility threads: one transcript that several models can take part in, and models can join or leave between turns.

Every participant sees the whole transcript, including turns from before it joined. Leaving only stops a participant
from answering the next turn: its earlier messages stay in the history. Storage only; no runner is called here."""
import sqlite3, time, uuid

from . import store

DB: sqlite3.Connection = store.DB
PARTICIPANTS = ("claude", "chatgpt")     # runner names, matching facility.RUNNERS
ROLES = ("user", "assistant")

DB.executescript("""
create table if not exists threads(id text primary key, name text, created real);
create table if not exists thread_participants(
    thread_id text, participant text, active int default 1, joined_seq int,
    primary key(thread_id, participant));
create table if not exists thread_messages(
    id text primary key, thread_id text, seq int, role text, participant text, content text, created real);
create index if not exists thread_messages_by_thread on thread_messages(thread_id, seq);
""")


class ThreadError(ValueError):
    """Unknown thread, unknown participant, or a request that would break the transcript."""


def _require_thread(thread_id: str) -> None:
    if DB.execute("select 1 from threads where id=?", (thread_id,)).fetchone() is None:
        raise ThreadError(f"unknown thread: {thread_id}")


def _next_seq(thread_id: str) -> int:
    row = DB.execute("select coalesce(max(seq), 0) from thread_messages where thread_id=?", (thread_id,)).fetchone()
    return row[0] + 1


def create_thread(name: str) -> str:
    if not name.strip():
        raise ThreadError("thread name is required")
    tid = uuid.uuid4().hex[:12]
    DB.execute("insert into threads(id, name, created) values(?,?,?)", (tid, name, time.time()))
    DB.commit()
    return tid


def set_participant(thread_id: str, participant: str, active: bool) -> None:
    """Add a participant (or bring a former one back) when active is True; remove it from the next turn when False."""
    _require_thread(thread_id)
    if participant not in PARTICIPANTS:
        raise ThreadError(f"participant must be one of {PARTICIPANTS}")
    row = DB.execute("select 1 from thread_participants where thread_id=? and participant=?", (thread_id, participant)).fetchone()
    if row is None:
        # joined_seq says where in the transcript this participant arrived (informational: everyone sees the full transcript);
        # the order of active_participants() is first-join order, from the row's rowid
        DB.execute("insert into thread_participants(thread_id, participant, active, joined_seq) values(?,?,?,?)",
                   (thread_id, participant, int(active), _next_seq(thread_id)))
    else:
        DB.execute("update thread_participants set active=? where thread_id=? and participant=?",
                   (int(active), thread_id, participant))
    DB.commit()


def active_participants(thread_id: str) -> list:
    _require_thread(thread_id)
    rows = DB.execute("select participant from thread_participants where thread_id=? and active=1 order by rowid",
                      (thread_id,)).fetchall()
    return [r["participant"] if hasattr(r, "keys") else r[0] for r in rows]


def add_message(thread_id: str, role: str, content: str, participant: str | None = None) -> int:
    """Append one turn. role 'user' has no participant; role 'assistant' must name the participant that wrote it."""
    _require_thread(thread_id)
    if role not in ROLES:
        raise ThreadError(f"role must be one of {ROLES}")
    if role == "assistant" and participant not in PARTICIPANTS:
        raise ThreadError("assistant messages must name a participant")
    if role == "user" and participant is not None:
        raise ThreadError("user messages have no participant")
    seq = _next_seq(thread_id)
    DB.execute("insert into thread_messages(id, thread_id, seq, role, participant, content, created) values(?,?,?,?,?,?,?)",
               (uuid.uuid4().hex, thread_id, seq, role, participant, content, time.time()))
    DB.commit()
    return seq


def transcript(thread_id: str) -> list:
    _require_thread(thread_id)
    rows = DB.execute("select seq, role, participant, content from thread_messages where thread_id=? order by seq",
                      (thread_id,)).fetchall()
    return [dict(r) for r in rows]


def prompt_for(thread_id: str, participant: str) -> str:
    """The full transcript as one prompt for `participant`, ending with the turn it should answer. Nothing is cut off."""
    msgs = transcript(thread_id)
    if not msgs or msgs[-1]["role"] != "user":
        raise ThreadError("a participant answers after a user message")
    lines = []
    for m in msgs:
        speaker = "User" if m["role"] == "user" else f"Assistant ({m['participant']})"
        lines.append(f"{speaker}: {m['content']}")
    return "\n\n".join(lines) + f"\n\nAnswer as Assistant ({participant})."
