"""Benchmark data model (additive; nothing existing is altered except one nullable column on arena_runs).

Benchmark -> Task -> Episode -> Attempt -> TraceEvent, plus Evaluation rows for deterministic / judge / human scores.
  * trace_events and evaluations are append-only (database triggers refuse UPDATE and DELETE).
  * Big prompts are stored once in `blobs`, content-addressed by sha256, and referenced from trace events.
  * Tool calls appear in a trace ONLY if the provider returned them. Nothing is fabricated.
  * hidden_answer_key lives on the task row and is never returned unless include_hidden=True is asked for explicitly,
    and it is never part of any prompt this module builds."""
import hashlib, json, sqlite3, time, uuid
from . import store

HUMAN_CRITERIA = ("groundedness", "citation_validity", "instruction_following", "overall_quality")
TRACE_KINDS = ("prompt", "output", "tool_call", "tool_result", "error", "latency", "tokens", "model_version",
               "parameters", "evidence_sha")
MARKET_BENCHMARK_ID = "market-intelligence"
ADHOC_BENCHMARK_ID = "adhoc-arena"
LEGACY_BENCHMARK_ID = "legacy-arena"

store.DB.executescript("""
create table if not exists benchmarks(
  id text primary key, name text, description text, version text, meta text default '{}', created real);
create table if not exists bench_tasks(
  id text primary key, benchmark_id text, name text, instructions text, evidence_provider text, evidence_sha256 text,
  allowed_tools text default '[]', expected_output_schema text, scorers text default '[]', hidden_answer_key text, created real, meta text default '{}', status text default 'approved', approved_at real);
create table if not exists episodes(
  id text primary key, task_id text, arena_run_id text, status text default 'running', evidence_sha256 text,
  seed integer, judge text, winner_attempt_id text, created real);
create table if not exists attempts(
  id text primary key, episode_id text, contestant_id text, agent text, provider text, model text, model_version text,
  parameters text default '{}', status text, error_kind text, error_message text, output text, latency real,
  usage text default '{}', anon_label integer, created real);
create table if not exists trace_events(
  id integer primary key autoincrement, attempt_id text, episode_id text, seq integer, kind text, data text, created real);
create table if not exists evaluations(
  id integer primary key autoincrement, episode_id text, attempt_id text, evaluator text, scorer text, score real,
  max_score real, passed int, applicable int default 1, details text default '{}', created real);
create table if not exists blobs(sha256 text primary key, text text);
create index if not exists idx_attempts_episode on attempts(episode_id);
create index if not exists idx_trace_attempt on trace_events(attempt_id);
create index if not exists idx_eval_attempt on evaluations(attempt_id);
create trigger if not exists trace_events_no_update before update on trace_events
  begin select raise(abort, 'trace_events is append-only'); end;
create trigger if not exists trace_events_no_delete before delete on trace_events
  begin select raise(abort, 'trace_events is append-only'); end;
create trigger if not exists evaluations_no_update before update on evaluations
  begin select raise(abort, 'evaluations is append-only'); end;
create trigger if not exists evaluations_no_delete before delete on evaluations
  begin select raise(abort, 'evaluations is append-only'); end;
""")
try:  # link from the old Arena table to its episode; old rows keep working without it
    store.DB.execute("alter table arena_runs add column episode_id text")
except sqlite3.OperationalError:
    pass


for _col in ("meta text default '{}'", "status text default 'approved'", "approved_at real"):
    try:  # tasks created before these columns existed read back with defaults
        store.DB.execute(f"alter table bench_tasks add column {_col}")
    except sqlite3.OperationalError:
        pass

STATUS_APPROVED = "approved"
STATUS_AWAITING = "awaiting_task_definition_approval"


def _id() -> str:
    return uuid.uuid4().hex[:10]


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def put_blob(text: str) -> str:
    sha = sha256_text(text)
    store.DB.execute("insert or ignore into blobs values(?,?)", (sha, text or ""))
    return sha


def get_blob(sha: str):
    r = store.DB.execute("select text from blobs where sha256=?", (sha,)).fetchone()
    return r[0] if r else None


# ---- benchmarks and tasks ------------------------------------------------------------------------------------------
def add_benchmark(bid, name, description="", version="1", meta=None) -> str:
    store.DB.execute("insert or ignore into benchmarks values(?,?,?,?,?,?)",
                     (bid, name, description, version, json.dumps(meta or {}), time.time()))
    store.DB.commit()
    return bid


def get_benchmark(bid):
    r = store.DB.execute("select * from benchmarks where id=?", (bid,)).fetchone()
    return r and {**dict(r), "meta": json.loads(r["meta"] or "{}")}


def list_benchmarks():
    rows = store.DB.execute("select id from benchmarks order by created").fetchall()
    return [get_benchmark(r["id"]) for r in rows]


def add_task(tid, benchmark_id, name, instructions, evidence_provider=None, evidence_sha256=None, allowed_tools=None,
             expected_output_schema=None, scorers=None, hidden_answer_key=None, meta=None, status=STATUS_APPROVED) -> str:
    """Tasks are immutable once registered (insert-or-ignore). Change a task by registering a new id."""
    store.DB.execute("insert or ignore into bench_tasks values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        tid, benchmark_id, name, instructions, evidence_provider, evidence_sha256, json.dumps(allowed_tools or []),
        json.dumps(expected_output_schema) if expected_output_schema is not None else None, json.dumps(scorers or []),
        json.dumps(hidden_answer_key) if hidden_answer_key is not None else None, time.time(), json.dumps(meta or {}), status, time.time() if status == STATUS_APPROVED else None))
    store.DB.commit()
    return tid


def set_task_status(tid, status):
    """Lifecycle only (the task's definition stays immutable). Approving records when."""
    store.DB.execute("update bench_tasks set status=?, approved_at=? where id=?",
                     (status, time.time() if status == STATUS_APPROVED else None, tid))
    store.DB.commit()


def approve_task(tid):
    """Mark a task's definition as final and runnable. Only a person should call this."""
    set_task_status(tid, STATUS_APPROVED)


def _task_row(r, include_hidden):
    if not r:
        return None
    d = dict(r)
    for k in ("allowed_tools", "expected_output_schema", "scorers", "hidden_answer_key"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    d["meta"] = json.loads(d.get("meta") or "{}")
    d["allowed_tools"] = d["allowed_tools"] or []
    d["scorers"] = d["scorers"] or []
    if not include_hidden:
        d.pop("hidden_answer_key", None)
    return d


def get_task(tid, include_hidden=False):
    return _task_row(store.DB.execute("select * from bench_tasks where id=?", (tid,)).fetchone(), include_hidden)


def list_tasks(benchmark_id=None):
    q, args = "select * from bench_tasks", ()
    if benchmark_id:
        q, args = q + " where benchmark_id=?", (benchmark_id,)
    return [_task_row(r, False) for r in store.DB.execute(q + " order by created", args)]


OUTPUT_FORMAT_NOTE = (
    "Respond with ONE JSON object and nothing else (no prose before or after it) that matches this JSON Schema. "
    "Every claim must cite the evidence rows it comes from; do not cite a value you cannot see in the evidence.\n")


def build_task_prompt(task: dict) -> str:
    """What a contestant sees for a task: the instructions and the required output format. Never the answer key."""
    parts = [task["instructions"].strip()]
    if task.get("expected_output_schema"):
        parts.append(OUTPUT_FORMAT_NOTE + json.dumps(task["expected_output_schema"], indent=1))
    return "\n\n".join(parts)


def ensure_adhoc_task(task_text: str, evidence_sha256=None, evidence_provider=None) -> str:
    """Free-form Arena prompts become tasks of an 'Ad-hoc Arena' benchmark (same prompt + evidence -> same task)."""
    add_benchmark(ADHOC_BENCHMARK_ID, "Ad-hoc Arena prompts", "Free-form prompts run from the Arena tab; no scorers.")
    tid = "adhoc-" + sha256_text(task_text + "|" + (evidence_sha256 or ""))[:12]
    add_task(tid, ADHOC_BENCHMARK_ID, task_text.strip().splitlines()[0][:80] if task_text.strip() else "(empty)", task_text,
             evidence_provider=evidence_provider, evidence_sha256=evidence_sha256)
    return tid


# ---- episodes, attempts, traces ------------------------------------------------------------------------------------
def create_episode(task_id, evidence_sha256=None, arena_run_id=None, seed=None) -> str:
    eid = _id()
    store.DB.execute("insert into episodes(id, task_id, arena_run_id, status, evidence_sha256, seed, created) values(?,?,?,?,?,?,?)",
                     (eid, task_id, arena_run_id, "running", evidence_sha256, seed, time.time()))
    if arena_run_id:
        store.DB.execute("update arena_runs set episode_id=? where id=?", (eid, arena_run_id))
    store.DB.commit()
    return eid


def set_episode_status(eid, status):
    store.DB.execute("update episodes set status=? where id=?", (status, eid))
    store.DB.commit()


def set_episode_judge(eid, judge: dict):
    store.DB.execute("update episodes set judge=? where id=?", (json.dumps(judge), eid))
    store.DB.commit()


def get_episode(eid):
    r = store.DB.execute("select * from episodes where id=?", (eid,)).fetchone()
    return r and {**dict(r), "judge": json.loads(r["judge"]) if r["judge"] else None}


def episode_for_run(run_id):
    r = store.DB.execute("select id from episodes where arena_run_id=?", (run_id,)).fetchone()
    return r[0] if r else None


def list_episodes(limit=50):
    rows = store.DB.execute("select id from episodes order by created desc limit ?", (limit,)).fetchall()
    return [get_episode(r["id"]) for r in rows]


def append_event(attempt_id, episode_id, kind, data) -> int:
    if kind not in TRACE_KINDS:
        raise ValueError(f"unknown trace event kind: {kind}")
    seq = store.DB.execute("select coalesce(max(seq), 0) + 1 from trace_events where attempt_id=?", (attempt_id,)).fetchone()[0]
    store.DB.execute("insert into trace_events(attempt_id, episode_id, seq, kind, data, created) values(?,?,?,?,?,?)",
                     (attempt_id, episode_id, seq, kind, json.dumps(data, default=str), time.time()))
    return seq


def record_attempt(episode_id, contestant: dict, result: dict, system: str, prompt: str, evidence_sha256=None, agent=None) -> str:
    """Store one contestant's attempt and its trace. `result` is a providers.RunResult.to_dict()."""
    aid = _id()
    err = result.get("error") or {}
    store.DB.execute(
        "insert into attempts(id, episode_id, contestant_id, agent, provider, model, model_version, parameters, status,"
        " error_kind, error_message, output, latency, usage, created) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, episode_id, contestant["id"], agent, result.get("provider"), result.get("model"), result.get("model_version"),
         json.dumps(result.get("parameters") or {}), "ok" if result.get("ok") else "error", err.get("kind"), err.get("message"),
         result.get("text") if result.get("ok") else None, result.get("latency"), json.dumps(result.get("usage") or {}), time.time()))
    append_event(aid, episode_id, "prompt", {"system_sha256": put_blob(system), "prompt_sha256": put_blob(prompt),
                                             "prompt_chars": len(prompt)})
    if evidence_sha256:
        append_event(aid, episode_id, "evidence_sha", {"sha256": evidence_sha256})
    append_event(aid, episode_id, "parameters", result.get("parameters") or {})
    if result.get("model_version"):
        append_event(aid, episode_id, "model_version", {"model_version": result["model_version"]})
    for call in result.get("tool_calls") or []:       # only what the provider returned
        append_event(aid, episode_id, "tool_call", call)
    if result.get("usage"):
        append_event(aid, episode_id, "tokens", result["usage"])
    append_event(aid, episode_id, "latency", {"seconds": result.get("latency")})
    if result.get("ok"):
        append_event(aid, episode_id, "output", {"text_sha256": put_blob(result["text"]), "chars": len(result["text"])})
    else:
        append_event(aid, episode_id, "error", err)
    store.DB.commit()
    return aid


def get_attempt(aid):
    r = store.DB.execute("select * from attempts where id=?", (aid,)).fetchone()
    return r and _attempt(r)


def _attempt(r):
    d = dict(r)
    d["parameters"], d["usage"] = json.loads(d["parameters"] or "{}"), json.loads(d["usage"] or "{}")
    return d


def attempts_for(episode_id):
    return [_attempt(r) for r in store.DB.execute("select * from attempts where episode_id=? order by rowid", (episode_id,))]


def trace_for(attempt_id):
    return [{**dict(r), "data": json.loads(r["data"])} for r in
            store.DB.execute("select * from trace_events where attempt_id=? order by seq", (attempt_id,))]


def set_anon_labels(mapping: dict):
    """{label(int): attempt_id}. Internal only: judges never see this."""
    for label, aid in mapping.items():
        store.DB.execute("update attempts set anon_label=? where id=?", (int(label), aid))
    store.DB.commit()


def set_winner_for_run(run_id, contestant_id):
    eid = episode_for_run(run_id)
    if not eid:
        return
    r = store.DB.execute("select id from attempts where episode_id=? and contestant_id=?", (eid, contestant_id)).fetchone()
    store.DB.execute("update episodes set winner_attempt_id=? where id=?", (r[0] if r else None, eid))
    store.DB.commit()


# ---- evaluations ---------------------------------------------------------------------------------------------------
def add_evaluation(episode_id, attempt_id, evaluator, scorer, score, max_score=1.0, passed=None, applicable=True, details=None):
    if evaluator not in ("deterministic", "judge", "human"):
        raise ValueError("evaluator must be deterministic, judge or human")
    store.DB.execute(
        "insert into evaluations(episode_id, attempt_id, evaluator, scorer, score, max_score, passed, applicable, details, created)"
        " values(?,?,?,?,?,?,?,?,?,?)",
        (episode_id, attempt_id, evaluator, scorer, score, max_score, None if passed is None else int(passed),
         int(applicable), json.dumps(details or {}, default=str), time.time()))
    store.DB.commit()


def evaluations_for(attempt_id, evaluator=None):
    q, args = "select * from evaluations where attempt_id=?", [attempt_id]
    if evaluator:
        q, args = q + " and evaluator=?", args + [evaluator]
    return [{**dict(r), "details": json.loads(r["details"] or "{}")} for r in store.DB.execute(q + " order by id", args)]


def add_human_score(attempt_id, criterion, score, rater="josh", notes=""):
    """Rubric scoring by a person: 1 (worst) to 5 (best) on one criterion. Re-scoring adds a new row; the latest wins."""
    if criterion not in HUMAN_CRITERIA:
        raise ValueError(f"criterion must be one of {HUMAN_CRITERIA}")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
        raise ValueError("score must be a number from 1 to 5")
    att = get_attempt(attempt_id)
    if not att:
        raise KeyError("attempt not found")
    add_evaluation(att["episode_id"], attempt_id, "human", criterion, float(score), 5.0, None, True,
                   {"rater": rater, "notes": notes})


def human_scores(attempt_id) -> dict:
    """{criterion: {'score', 'rater', 'notes'}} using the most recent score per criterion."""
    out = {}
    for e in evaluations_for(attempt_id, "human"):
        out[e["scorer"]] = {"score": e["score"], "rater": e["details"].get("rater"), "notes": e["details"].get("notes", "")}
    return out


def scorecard(episode_id) -> dict:
    """Everything needed to trace a score back to its source: benchmark, task, episode, contestant, provider, model,
    version, parameters, evidence hash, deterministic / judge / human scores, latency and usage."""
    ep = get_episode(episode_id)
    if not ep:
        raise KeyError("episode not found")
    task = get_task(ep["task_id"])
    bench = get_benchmark(task["benchmark_id"]) if task else None
    rows = []
    for a in attempts_for(episode_id):
        det = [e for e in evaluations_for(a["id"], "deterministic")]
        judge = {e["scorer"]: {"score": e["score"], "max": e["max_score"], "details": e["details"]}
                 for e in evaluations_for(a["id"], "judge")}
        usage = a["usage"]
        rows.append({
            "attempt_id": a["id"], "contestant": a["contestant_id"], "agent": a["agent"], "provider": a["provider"],
            "model": a["model"], "model_version": a["model_version"], "parameters": a["parameters"],
            "evidence_sha256": ep["evidence_sha256"], "status": a["status"],
            "error": {"kind": a["error_kind"], "message": a["error_message"]} if a["status"] == "error" else None,
            "latency": a["latency"], "usage": usage, "cost": usage.get("cost"),   # cost only if the provider reported one
            "deterministic_scores": [{"scorer": e["scorer"], "score": e["score"], "passed": None if e["passed"] is None else bool(e["passed"]),
                                      "applicable": bool(e["applicable"]), "details": e["details"]} for e in det],
            "judge_scores": judge, "human_scores": human_scores(a["id"]),
            "winner": ep["winner_attempt_id"] == a["id"],
        })
    return {"benchmark": {"id": bench["id"], "name": bench["name"], "version": bench["version"]} if bench else None,
            "task": {"id": task["id"], "name": task["name"], "evidence_sha256": task["evidence_sha256"]} if task else None,
            "episode": {"id": ep["id"], "arena_run_id": ep["arena_run_id"], "status": ep["status"], "seed": ep["seed"],
                        "evidence_sha256": ep["evidence_sha256"], "judge": ep["judge"]},
            "attempts": rows}


# ---- old Arena records ---------------------------------------------------------------------------------------------
def link_legacy_arena_runs() -> dict:
    """Give every pre-existing arena_runs row an episode (idempotent). Old rows are left as they were; only
    arena_runs.episode_id is filled in. No usage, latency or tool traces are invented for them."""
    add_benchmark(LEGACY_BENCHMARK_ID, "Legacy Arena runs", "Arena comparisons recorded before the benchmark data model existed.")
    linked = skipped = 0
    for r in store.DB.execute("select id, prompt, results, winner, created from arena_runs where episode_id is null").fetchall():
        try:
            results = json.loads(r["results"])
            tid = "legacy-" + sha256_text(r["prompt"])[:12]
            add_task(tid, LEGACY_BENCHMARK_ID, (r["prompt"] or "").strip().splitlines()[0][:80] if (r["prompt"] or "").strip() else "(empty)", r["prompt"])
            eid = create_episode(tid, None, r["id"], None)
            for x in results:
                prov, _, model = str(x["id"]).partition(":")
                failed = str(x.get("text", "")).startswith("(failed:") or bool(x.get("error"))
                res = {"provider": prov, "model": model or None, "ok": not failed, "text": x.get("text", ""), "parameters": {"legacy": True},
                       "error": {"kind": x.get("error") or "provider_error", "message": x.get("text", "")} if failed else None}
                record_attempt(eid, {"id": x["id"]}, res, "", r["prompt"] or "")
            win = store.DB.execute("select id from attempts where episode_id=? and contestant_id=?", (eid, r["winner"])).fetchone()
            store.DB.execute("update episodes set status='complete', winner_attempt_id=? where id=?", (win[0] if win else None, eid))
            store.DB.commit()
            linked += 1
        except (ValueError, KeyError, TypeError):
            skipped += 1
    return {"linked": linked, "skipped": skipped}


# ---- Benchmark #1: Market Intelligence -----------------------------------------------------------------------------
MARKET_TASK_INSTRUCTIONS = (
    "Using ONLY the market-intelligence evidence supplied below, report up to 10 findings about demand, competition, "
    "trends and IP risk in the book and merchandise data. Each finding must be a single factual claim that the evidence "
    "supports. Cite the exact evidence rows (row_id, field and the value as written). Where a finding states a count, "
    "sum, average, minimum, maximum or number of distinct values, also give it as a numeric claim so it can be recomputed "
    "from the files. Do not invent figures or sources.")
MARKET_OUTPUT_SCHEMA = {
    "type": "object", "required": ["findings"],
    "properties": {"findings": {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": ["claim", "citations"],
        "properties": {
            "claim": {"type": "string"},
            "citations": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["row_id", "field", "value"],
                "properties": {"row_id": {"type": "string", "description": "e.g. dataset1_books.csv:2"},
                               "field": {"type": "string"}, "value": {"type": "string"}}}},
            "numeric": {"type": "array", "items": {
                "type": "object", "required": ["file", "op", "result"],
                "properties": {"description": {"type": "string"},
                               "file": {"type": "string", "description": "dataset1_books.csv, dataset2_merch.csv, merch_listing_counts.csv or book_trends.csv"},
                               "op": {"enum": ["count", "sum", "mean", "min", "max", "count_distinct"]},
                               "column": {"type": ["string", "null"]},
                               "where": {"type": ["object", "null"], "description": "column -> value equality filters"},
                               "result": {"type": "number"}}}}}}}}}
MARKET_SCORERS = [{"name": "schema", "kind": "deterministic"}, {"name": "citation", "kind": "deterministic"},
                  {"name": "numeric", "kind": "deterministic"}, {"name": "blind_judge", "kind": "judge"}]


def register_market_benchmark() -> dict:
    """Register Benchmark #1 (idempotent) using the real files in MARKET_DATA_DIR and record the exact evidence hash.
    Registration only: nothing is executed. If the files change, a NEW task id (hash-suffixed) is added; old tasks stay."""
    from . import market_data
    ev = market_data.build_evidence()
    add_benchmark(MARKET_BENCHMARK_ID, "Market Intelligence", "Benchmark #1: evidence-grounded analysis of the real book and merch datasets.", "1")
    tid = f"market-intel-{ev['sha256'][:12]}"
    prior = get_task(tid)
    existed = prior is not None
    add_task(tid, MARKET_BENCHMARK_ID, "Evidence-grounded market analysis", MARKET_TASK_INSTRUCTIONS, "market_data",
             ev["sha256"], [], MARKET_OUTPUT_SCHEMA, MARKET_SCORERS, None,
             {"source": ev["source"], "files": {k: {"file": v["file"], "records": v["records"], "sha256": v["sha256"],
                                                    "omitted_columns": v["omitted_columns"]} for k, v in ev["files"].items()}},
             STATUS_AWAITING)
    if get_task(tid)["approved_at"] is None:       # never explicitly approved by a person: the wording is still a placeholder
        set_task_status(tid, STATUS_AWAITING)
    return {"benchmark_id": MARKET_BENCHMARK_ID, "task_id": tid, "evidence_sha256": ev["sha256"], "source": ev["source"],
            "already_registered": existed, "status": get_task(tid)["status"], "files": {k: v["records"] for k, v in ev["files"].items()}}
