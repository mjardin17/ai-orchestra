"""Lessons ledger: agents (Claude, Codex, free models) record their failures here as FAILURE MODES.
A repeated failure is an observation, not a verified rule: repeating never changes a lesson's status. A lesson becomes
active only when you approve it, and approval is NOT verification: verified stays 0 until a verification pipeline (reproduce
the fix, test it on held-out tasks, bind a regression test) marks it, and that pipeline does not exist yet. Agents are shown
only lessons that are both approved and verified, so today none are injected. Everything else stays stored (history is
never deleted). Dismissed lessons stay dismissed."""
import hashlib, re, sqlite3, time, uuid
from . import store

MAX_INJECTED = 5    # most rules shown to one agent, most-repeated first
STATUSES = ("pending", "active", "dismissed")

store.DB.executescript("""
create table if not exists lessons(
  id text primary key, signature text unique, agent text, model text, rule text,
  status text default 'pending', seen_count int default 1, created real, last_seen real);
""")
for _col in (  # additive columns; rows that existed before them read back as unverified legacy lessons
        "verified int default 0",                       # 1 only after the (future) verification pipeline passes
        "provenance text default 'legacy_unverified'",  # observed_failure | manual | legacy_unverified
        "kind text default 'failure_mode'",
        "approved_at real"):
    try:
        store.DB.execute(f"alter table lessons add column {_col}")
    except sqlite3.OperationalError:
        pass
store.DB.commit()


def _signature(agent, model, error) -> str:
    norm = re.sub(r"\d+|0x[0-9a-f]+", "#", str(error or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()[:160]
    return hashlib.sha1(f"{agent}|{model}|{norm}".encode()).hexdigest()[:16]


def get(lid):
    r = store.DB.execute("select * from lessons where id=?", (lid,)).fetchone()
    return dict(r) if r else None


def record_failure(agent, model, error, request=None) -> str:
    """Log one failure. Same failure again just bumps its count. Never raises: logging must not break a run."""
    agent, model, error = str(agent or "*"), str(model or "?"), str(error or "unknown error")
    sig, now = _signature(agent, model, error), time.time()
    row = store.DB.execute("select id, seen_count from lessons where signature=?", (sig,)).fetchone()
    if row:  # same failure again: count it. Frequency never promotes a lesson.
        store.DB.execute("update lessons set seen_count=?, last_seen=? where id=?", (row["seen_count"] + 1, now, row["id"]))
        store.DB.commit()
        return row["id"]
    lid = uuid.uuid4().hex[:8]
    rule = f"On '{agent}' jobs, {model} failed with: {' '.join(error.split())[:200]}. Avoid repeating this or handle it explicitly."
    store.DB.execute(
        "insert into lessons(id, signature, agent, model, rule, status, seen_count, created, last_seen, verified, provenance, kind)"
        " values(?,?,?,?,?,?,?,?,?,?,?,?)", (lid, sig, agent, model, rule, "pending", 1, now, now, 0, "observed_failure", "failure_mode"))
    store.DB.commit()
    return lid


def add_lesson(rule, agent="*") -> str:
    """A rule you write yourself (e.g. a mistake from outside the orchestra). Starts pending until approved."""
    lid, now = uuid.uuid4().hex[:8], time.time()
    store.DB.execute(
        "insert into lessons(id, signature, agent, model, rule, status, seen_count, created, last_seen, verified, provenance, kind)"
        " values(?,?,?,?,?,?,?,?,?,?,?,?)",
        (lid, f"manual:{lid}", agent or "*", "manual", rule.strip()[:500], "pending", 1, now, now, 0, "manual", "manual"))
    store.DB.commit()
    return lid


def set_status(lid, status) -> bool:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    cur = store.DB.execute("update lessons set status=?, approved_at=? where id=?",
                           (status, time.time() if status == "active" else None, lid))   # approval is not verification
    store.DB.commit()
    return cur.rowcount > 0


def list_lessons(status=None):
    q, args = "select * from lessons", ()
    if status:
        q, args = q + " where status=?", (status,)
    return [dict(r) for r in store.DB.execute(q + " order by seen_count desc, last_seen desc", args)]


def active_for(agent):
    """Approved (status active) lessons for this agent, verified or not. NOT what gets injected; see injectable_for."""
    return [dict(r) for r in store.DB.execute(
        "select * from lessons where status='active' and (agent=? or agent='*') "
        "order by seen_count desc, last_seen desc", (agent,))]


def injectable_for(agent):
    """The only lessons an agent may be shown: approved (status active) AND verified (verified=1).
    Observed, pending, dismissed, and approved-but-unverified lessons are stored but never injected."""
    return [dict(r) for r in store.DB.execute(
        "select * from lessons where status='active' and verified=1 and (agent=? or agent='*') "
        "order by seen_count desc, last_seen desc limit ?", (agent, MAX_INJECTED))]


def with_lessons(system: str, agent: str) -> str:
    """Append the verified, approved rules for this agent to its system prompt (unchanged if there are none)."""
    rules = injectable_for(agent)
    if not rules:
        return system
    return system + "\n\nVerified rules (follow them):" + "".join(f"\n- {r['rule']}" for r in rules)
