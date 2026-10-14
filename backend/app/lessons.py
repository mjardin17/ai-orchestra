"""Lessons ledger: agents (Claude, Codex, free models) record their failures here, repeated failures become
rules, and active rules are injected into every later agent prompt so the same mistake isn't made twice.
Nothing is promoted silently by a single failure: a lesson goes pending -> active after PROMOTE_AT repeats,
or when you approve it. Dismissed lessons stay dismissed."""
import hashlib, re, time, uuid
from . import store

PROMOTE_AT = 3      # repeats before a pending lesson turns active on its own
MAX_INJECTED = 5    # most rules shown to one agent, most-repeated first
STATUSES = ("pending", "active", "dismissed")

store.DB.executescript("""
create table if not exists lessons(
  id text primary key, signature text unique, agent text, model text, rule text,
  status text default 'pending', seen_count int default 1, created real, last_seen real);
""")


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
    row = store.DB.execute("select id, status, seen_count from lessons where signature=?", (sig,)).fetchone()
    if row:
        count = row["seen_count"] + 1
        status = "active" if row["status"] == "pending" and count >= PROMOTE_AT else row["status"]
        store.DB.execute("update lessons set seen_count=?, status=?, last_seen=? where id=?", (count, status, now, row["id"]))
        store.DB.commit()
        return row["id"]
    lid = uuid.uuid4().hex[:8]
    rule = f"On '{agent}' jobs, {model} failed with: {' '.join(error.split())[:200]}. Avoid repeating this or handle it explicitly."
    store.DB.execute("insert into lessons values(?,?,?,?,?,?,?,?,?)", (lid, sig, agent, model, rule, "pending", 1, now, now))
    store.DB.commit()
    return lid


def add_lesson(rule, agent="*") -> str:
    """A rule you write yourself (e.g. a mistake from outside the orchestra). Starts pending until approved."""
    lid, now = uuid.uuid4().hex[:8], time.time()
    store.DB.execute("insert into lessons values(?,?,?,?,?,?,?,?,?)",
                     (lid, f"manual:{lid}", agent or "*", "manual", rule.strip()[:500], "pending", 1, now, now))
    store.DB.commit()
    return lid


def set_status(lid, status) -> bool:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    cur = store.DB.execute("update lessons set status=? where id=?", (status, lid))
    store.DB.commit()
    return cur.rowcount > 0


def list_lessons(status=None):
    q, args = "select * from lessons", ()
    if status:
        q, args = q + " where status=?", (status,)
    return [dict(r) for r in store.DB.execute(q + " order by seen_count desc, last_seen desc", args)]


def active_for(agent):
    return [dict(r) for r in store.DB.execute(
        "select * from lessons where status='active' and (agent=? or agent='*') "
        "order by seen_count desc, last_seen desc limit ?", (agent, MAX_INJECTED))]


def with_lessons(system: str, agent: str) -> str:
    """Append the active rules for this agent to its system prompt (unchanged if there are none)."""
    rules = active_for(agent)
    if not rules:
        return system
    bullets = "".join(f"\n- {r['rule']}" for r in rules)
    return f"{system}\n\nKnown pitfalls learned from earlier runs (follow them):{bullets}"
