"""Benchmark data model: clean creation, append-only traces, hidden keys, legacy Arena compatibility, Benchmark #1
registration, and an end-to-end Arena episode with deterministic + judge + human scores and full provenance."""
import asyncio, hashlib, json, os, sqlite3, subprocess, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app import arena, bench, judge, market_data, models, providers, scoring, store
from app.main import app

c = TestClient(app)
BACKEND = Path(__file__).resolve().parents[1]
REAL = Path(r"C:\Users\jjard\Downloads")
needs_real = pytest.mark.skipif(not (REAL / "dataset1_books.csv").exists(), reason="real datasets not present")


def uid(prefix):
    return prefix + os.urandom(4).hex()


# ---- clean database creation / migration ------------------------------------------------------------------------
def test_clean_database_gets_every_table_and_the_arena_link_column(tmp_path):
    code = ("import sqlite3; from app import bench, lessons, providers; import app.store as s; "
            "t = {r[0] for r in s.DB.execute(\"select name from sqlite_master where type='table'\")}; "
            "cols = [r[1] for r in s.DB.execute('pragma table_info(arena_runs)')]; "
            "trig = {r[0] for r in s.DB.execute(\"select name from sqlite_master where type='trigger'\")}; "
            "print(sorted(t)); print(cols); print(sorted(trig))")
    for _ in range(2):      # second run proves the migration is idempotent on an existing database
        out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                             env=dict(os.environ, PYTHONPATH=str(BACKEND)))
        assert out.returncode == 0, out.stderr
    tables, cols, triggers = (eval(line) for line in out.stdout.strip().splitlines())
    for t in ("benchmarks", "bench_tasks", "episodes", "attempts", "trace_events", "evaluations", "blobs", "provider_health",
              "arena_runs", "lessons"):
        assert t in tables
    assert "episode_id" in cols and cols[:5] == ["id", "prompt", "results", "winner", "created"]
    assert {"trace_events_no_update", "trace_events_no_delete", "evaluations_no_update", "evaluations_no_delete"} <= set(triggers)


def test_old_arena_database_without_episode_column_is_upgraded(tmp_path):
    db = sqlite3.connect(tmp_path / "orchestra.db")
    db.executescript("create table arena_runs(id text primary key, prompt text, results text, winner text, created real);"
                     "insert into arena_runs values('old','p','[]','x',1);")
    db.commit(); db.close()
    out = subprocess.run([sys.executable, "-c", "from app import bench, store; print([tuple(r) for r in store.DB.execute('select id, episode_id from arena_runs').fetchall()])"],
                         cwd=tmp_path, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(BACKEND)))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[('old', None)]"


def test_bench_tasks_table_from_an_earlier_shape_is_migrated(tmp_path):
    db = sqlite3.connect(tmp_path / "orchestra.db")
    db.executescript("create table bench_tasks(id text primary key, benchmark_id text, name text, instructions text, evidence_provider text,"
                     " evidence_sha256 text, allowed_tools text default '[]', expected_output_schema text, scorers text default '[]',"
                     " hidden_answer_key text, created real);"
                     "insert into bench_tasks(id, benchmark_id, name, instructions) values('t1','b','n','i');")
    db.commit(); db.close()
    code = ("from app import bench; bench.add_benchmark('b','B'); bench.add_task('t2','b','n','i',meta={'k':1}); "
            "print(bench.get_task('t1')['meta'], bench.get_task('t2')['meta'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(BACKEND)))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "{} {'k': 1}"


# ---- append-only evidence ---------------------------------------------------------------------------------------
def make_attempt(result=None, evidence_sha="e" * 64):
    tid = bench.ensure_adhoc_task(uid("task "), None)
    eid = bench.create_episode(tid, evidence_sha)
    res = result or {"provider": "p", "model": "m", "ok": True, "text": "hello", "usage": {"total_tokens": 3}, "latency": 0.5,
                     "model_version": "v1", "parameters": {"max_tokens": 10}}
    return eid, bench.record_attempt(eid, {"id": "p:m"}, res, "SYSTEM", "PROMPT TEXT", evidence_sha)


def test_trace_events_and_evaluations_are_append_only():
    eid, aid = make_attempt()
    bench.add_evaluation(eid, aid, "deterministic", "schema", 1.0, 1.0, True)
    for sql in ("update trace_events set kind='x'", "delete from trace_events", "update evaluations set score=0", "delete from evaluations"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store.DB.execute(sql)
    assert bench.trace_for(aid) and bench.evaluations_for(aid)


def test_trace_records_what_happened_and_never_invents_tool_calls():
    eid, aid = make_attempt()
    kinds = [e["kind"] for e in bench.trace_for(aid)]
    assert kinds == ["prompt", "evidence_sha", "parameters", "model_version", "tokens", "latency", "output"]
    assert "tool_call" not in kinds and "tool_result" not in kinds
    prompt_ev = bench.trace_for(aid)[0]["data"]
    assert bench.get_blob(prompt_ev["prompt_sha256"]) == "PROMPT TEXT" and bench.get_blob(prompt_ev["system_sha256"]) == "SYSTEM"
    assert bench.trace_for(aid)[1]["data"]["sha256"] == "e" * 64


def test_provider_tool_calls_are_traced_only_when_returned():
    _, aid = make_attempt({"provider": "p", "model": "m", "ok": True, "text": "x", "tool_calls": [{"name": "lookup"}], "parameters": {}})
    calls = [e for e in bench.trace_for(aid) if e["kind"] == "tool_call"]
    assert [e["data"] for e in calls] == [{"name": "lookup"}]
    assert not any(e["kind"] == "tool_result" for e in bench.trace_for(aid))


def test_failed_attempt_has_an_error_event_and_no_output():
    eid, aid = make_attempt({"provider": "p", "model": "m", "ok": False, "error": {"kind": "timeout", "message": "slow"}, "parameters": {}})
    att = bench.get_attempt(aid)
    assert att["status"] == "error" and att["error_kind"] == "timeout" and att["output"] is None
    kinds = [e["kind"] for e in bench.trace_for(aid)]
    assert "error" in kinds and "output" not in kinds


def test_unknown_trace_kind_is_rejected():
    _, aid = make_attempt()
    with pytest.raises(ValueError):
        bench.append_event(aid, "x", "telepathy", {})


# ---- hidden answer keys -----------------------------------------------------------------------------------------
def test_hidden_answer_key_is_never_returned_or_prompted():
    tid = uid("keyed-")
    bench.add_benchmark("keyed-bench", "Keyed")
    bench.add_task(tid, "keyed-bench", "t", "What is the answer?", scorers=[{"name": "exact_match", "kind": "deterministic"}],
                   hidden_answer_key={"expected": "SECRET-ANSWER-42"})
    assert "hidden_answer_key" not in bench.get_task(tid)
    assert bench.get_task(tid, include_hidden=True)["hidden_answer_key"] == {"expected": "SECRET-ANSWER-42"}
    assert "SECRET-ANSWER-42" not in json.dumps(c.get("/api/bench/tasks").json())
    assert "SECRET-ANSWER-42" not in bench.build_task_prompt(bench.get_task(tid))


def test_tasks_are_immutable_once_registered():
    tid = uid("imm-")
    bench.add_benchmark("imm-bench", "Imm")
    bench.add_task(tid, "imm-bench", "t", "original")
    bench.add_task(tid, "imm-bench", "t", "changed")
    assert bench.get_task(tid)["instructions"] == "original"


# ---- legacy Arena compatibility ---------------------------------------------------------------------------------
def test_legacy_arena_runs_are_linked_without_changing_them():
    store.DB.execute("delete from arena_runs")
    store.DB.commit()
    old = [{"id": "claude:haiku", "label": "Claude Haiku", "text": "good answer"}, {"id": "groq:x", "label": "X", "text": "(failed: groq: HTTP 500)"}]
    store.save_arena("legacy01", "Legacy prompt\nsecond line", old, "claude:haiku")
    before = [r for r in store.list_arena(full=True) if r["id"] == "legacy01"]
    store.DB.execute("insert into arena_runs(id, prompt, results, winner, created) values('legacybad','p','{not json','x',1)")
    store.DB.commit()
    out = bench.link_legacy_arena_runs()
    assert out == {"linked": 1, "skipped": 1}
    assert bench.link_legacy_arena_runs() == {"linked": 0, "skipped": 1}          # idempotent
    eid = bench.episode_for_run("legacy01")
    atts = {a["contestant_id"]: a for a in bench.attempts_for(eid)}
    assert atts["claude:haiku"]["status"] == "ok" and atts["groq:x"]["status"] == "error"
    assert atts["claude:haiku"]["provider"] == "claude" and atts["claude:haiku"]["usage"] == {}
    assert not any(e["kind"] in ("tool_call", "tokens", "model_version") for e in bench.trace_for(atts["claude:haiku"]["id"]))   # nothing invented
    assert bench.get_episode(eid)["winner_attempt_id"] == atts["claude:haiku"]["id"]
    store.DB.execute("delete from arena_runs where id='legacybad'")      # the existing list_arena cannot read a corrupt row
    store.DB.commit()
    after = [r for r in store.list_arena(full=True) if r["id"] == "legacy01"]
    assert after == before                                                         # the old record reads exactly as before
    assert store.DB.execute("select episode_id from arena_runs where id='legacy01'").fetchone()[0] == eid


def test_export_skips_failed_attempts_and_still_reads_old_records():
    store.DB.execute("delete from arena_runs")
    store.DB.commit()
    store.save_arena("exp1", "Q?", [{"id": "a", "label": "A", "text": "good"}, {"id": "b", "label": "B", "text": "worse"},
                                    {"id": "c", "label": "C", "text": "(failed: timeout: slow)", "error": "timeout"}], "a")
    lines = [json.loads(l) for l in arena.export_jsonl().splitlines()]
    assert len(lines) == 1 and lines[0] == {"prompt": "Q?", "chosen": "good", "chosen_model": "A", "rejected": "worse", "rejected_model": "B"}
    r = c.get("/api/arena/export")
    assert r.status_code == 200 and "good" in r.text


def test_winner_override_still_works_and_updates_the_episode():
    store.DB.execute("delete from arena_runs")
    store.DB.commit()
    store.save_arena("win1", "Q", [{"id": "a", "label": "A", "text": "1"}, {"id": "b", "label": "B", "text": "2"}], "a")
    bench.link_legacy_arena_runs()
    assert c.put("/api/arena/winner", json={"run_id": "win1", "winner_id": "b"}).status_code == 200
    assert store.list_arena()[0]["winner"] == "b"
    eid = bench.episode_for_run("win1")
    assert bench.get_attempt(bench.get_episode(eid)["winner_attempt_id"])["contestant_id"] == "b"


# ---- Benchmark #1 registration ----------------------------------------------------------------------------------
def test_market_registration_records_the_exact_evidence_hash(tiny_market):
    out = bench.register_market_benchmark()
    assert out["evidence_sha256"] == market_data.build_evidence()["sha256"] and out["already_registered"] is False
    task = bench.get_task(out["task_id"])
    assert task["evidence_sha256"] == out["evidence_sha256"] and task["evidence_provider"] == "market_data"
    assert task["expected_output_schema"] == bench.MARKET_OUTPUT_SCHEMA
    assert [s["name"] for s in task["scorers"]] == ["schema", "citation", "numeric", "blind_judge"]
    files = task["meta"]["files"]
    assert task["meta"]["source"] == str(tiny_market)
    assert files["books"]["sha256"] == hashlib.sha256((tiny_market / "dataset1_books.csv").read_bytes()).hexdigest()
    again = bench.register_market_benchmark()
    assert again["already_registered"] is True and again["task_id"] == out["task_id"]


def test_market_task_is_registered_as_awaiting_approval_and_cannot_run(tiny_market, monkeypatch):
    out = bench.register_market_benchmark()
    assert out["status"] == bench.STATUS_AWAITING == "awaiting_task_definition_approval"
    assert bench.get_task(out["task_id"])["status"] == bench.STATUS_AWAITING
    called = []

    async def never(mm, spec):
        called.append(mm["id"])
        return providers.RunResult("p", "m", True, text="x")
    pool = [{"id": f"p{i}:m", "label": f"M{i}", "kind": "free", "provider": f"p{i}", "model": "m", "power": 3} for i in range(3)]
    monkeypatch.setattr(models, "enabled_pool", lambda: pool)
    monkeypatch.setattr(providers, "run", never)
    events = run_task(out["task_id"], ["p0:m", "p1:m"])
    assert events[-1]["type"] == "error" and "awaiting_task_definition_approval" in events[-1]["message"]
    assert called == []                                   # no contestant, no judge, no spend
    assert not any(e["type"] == "episode" for e in events)


def test_approval_is_explicit_and_not_downgraded_by_re_registering(tiny_market):
    tid = bench.register_market_benchmark()["task_id"]
    assert bench.register_market_benchmark()["status"] == bench.STATUS_AWAITING          # re-registering never approves
    bench.approve_task(tid)
    again = bench.register_market_benchmark()
    assert again["status"] == bench.STATUS_APPROVED and bench.get_task(tid)["approved_at"]


def test_adhoc_and_legacy_tasks_stay_runnable():
    assert bench.get_task(bench.ensure_adhoc_task("hello " + uid("")))["status"] == bench.STATUS_APPROVED


def test_task_status_migrates_onto_an_older_table(tmp_path):
    db = sqlite3.connect(tmp_path / "orchestra.db")
    db.executescript("create table bench_tasks(id text primary key, benchmark_id text, name text, instructions text, evidence_provider text,"
                     " evidence_sha256 text, allowed_tools text default '[]', expected_output_schema text, scorers text default '[]',"
                     " hidden_answer_key text, created real, meta text default '{}');"
                     "insert into bench_tasks(id, benchmark_id, name, instructions) values('t1','b','n','i');")
    db.commit(); db.close()
    code = "from app import bench; bench.add_benchmark('b','B'); bench.add_task('t2','b','n','i',status=bench.STATUS_AWAITING); print(bench.get_task('t1')['status'], bench.get_task('t2')['status'])"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(BACKEND)))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["approved", "awaiting_task_definition_approval"]


def test_changed_data_registers_a_new_task_and_keeps_the_old_one(tiny_market):
    first = bench.register_market_benchmark()["task_id"]
    with open(tiny_market / "dataset1_books.csv", "a", encoding="utf-8") as f:
        f.write("D,w,$40.00,Cat2,http://u4\n")
    second = bench.register_market_benchmark()["task_id"]
    assert first != second and bench.get_task(first) and bench.get_task(second)


@needs_real
def test_real_registration_matches_the_real_files(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_DIR", str(REAL))
    out = bench.register_market_benchmark()
    ev = market_data.build_evidence()
    assert out["evidence_sha256"] == ev["sha256"] and out["source"] == str(REAL)
    assert out["files"] == {"books": 887, "merch": 902, "merch_counts": 102, "trends": 6}
    meta = bench.get_task(out["task_id"])["meta"]["files"]
    for key, name in market_data.FILES.items():
        assert meta[key]["sha256"] == hashlib.sha256((REAL / name).read_bytes()).hexdigest()


# ---- end to end: a registered task through the Arena ------------------------------------------------------------
def approved_market_task():
    """Register Benchmark #1's task for the current (tiny) evidence and approve it, as a person eventually will."""
    tid = bench.register_market_benchmark()["task_id"]
    bench.approve_task(tid)
    return tid


def good_answer():
    return json.dumps({"findings": [{"claim": "Cat1 has two books", "citations": [
        {"row_id": "dataset1_books.csv:2", "field": "title", "value": "A"}, {"row_id": "dataset1_books.csv:3", "field": "price", "value": "$20.00"}],
        "numeric": [{"file": "dataset1_books.csv", "op": "count", "where": {"list_category": "Cat1"}, "result": 2},
                    {"file": "dataset1_books.csv", "op": "mean", "column": "price", "where": {"list_category": "Cat1"}, "result": 15}]}]})


def half_wrong_answer():
    return json.dumps({"findings": [{"claim": "x", "citations": [
        {"row_id": "dataset1_books.csv:2", "field": "title", "value": "A"}, {"row_id": "dataset1_books.csv:3", "field": "title", "value": "WRONG"}],
        "numeric": [{"file": "dataset1_books.csv", "op": "count", "result": 99}]}]})


def e2e_pool(monkeypatch, texts):
    pool = [{"id": f"p{i}:m{i}", "label": f"Model {i}", "kind": "free", "provider": f"p{i}", "model": f"m{i}", "power": 3} for i in range(4)]
    monkeypatch.setattr(models, "enabled_pool", lambda: pool)
    verdict = json.dumps({"scores": {str(i): {cr: 3 + (i % 3) for cr in judge.CRITERIA} for i in (1, 2, 3)}, "ranking": [1, 2, 3],
                          "reasoning": {"1": "best grounded"}})
    seen = []

    async def fake(mm, spec):
        seen.append((mm["id"], spec))
        if spec.system == judge.JUDGE_SYSTEM:
            return providers.RunResult("p3", "m3", True, text=verdict, model_version="judge-v")
        return providers.RunResult(mm["provider"], mm["model"], True, text=texts[mm["id"]], model_version="ver-" + mm["id"],
                                   usage={"total_tokens": 11}, latency=0.25, parameters={"max_tokens": spec.max_tokens})
    monkeypatch.setattr(providers, "run", fake)
    return seen


def run_task(task_id, ids):
    async def go():
        return [e async for e in arena.run("", ids, task_id=task_id)]
    return asyncio.run(go())


def test_task_episode_end_to_end_with_provenance(tiny_market, monkeypatch):
    tid = approved_market_task()
    seen = e2e_pool(monkeypatch, {"p0:m0": good_answer(), "p1:m1": half_wrong_answer(), "p2:m2": "I refuse to use JSON"})
    events = run_task(tid, ["p0:m0", "p1:m1", "p2:m2"])
    final = next(e for e in events if e["type"] == "arena_final")
    sha = market_data.build_evidence()["sha256"]
    assert next(e for e in events if e["type"] == "evidence")["sha256"] == sha
    # contestants got the task prompt + schema + evidence, identically
    contestant_specs = {(s.system, s.prompt) for mid, s in seen if s.system != judge.JUDGE_SYSTEM}
    assert len(contestant_specs) == 1
    prompt = next(iter(contestant_specs))[1]
    assert "JSON Schema" in prompt and "dataset1_books.csv:2" in prompt
    card = bench.scorecard(final["episode_id"])
    assert card["benchmark"]["name"] == "Market Intelligence" and card["task"]["id"] == tid
    assert card["episode"]["evidence_sha256"] == sha and card["task"]["evidence_sha256"] == sha
    by = {a["contestant"]: a for a in card["attempts"]}
    for a in by.values():
        assert a["provider"] and a["model"] and a["model_version"] == "ver-" + a["contestant"] and a["parameters"]["max_tokens"] == arena.ARENA_MAX_TOKENS
        assert a["evidence_sha256"] == sha and a["latency"] == 0.25 and a["usage"] == {"total_tokens": 11} and a["cost"] is None
        assert set(a["judge_scores"]) == set(judge.CRITERIA) and a["human_scores"] == {}
    det = {k: {d["scorer"]: d for d in v["deterministic_scores"]} for k, v in by.items()}
    assert det["p0:m0"]["schema"]["passed"] and det["p0:m0"]["citation"]["score"] == 1.0 and det["p0:m0"]["numeric"]["score"] == 1.0
    assert det["p1:m1"]["citation"]["score"] == 0.5 and det["p1:m1"]["numeric"]["score"] == 0.0
    assert det["p2:m2"]["schema"]["passed"] is False and det["p2:m2"]["citation"]["score"] == 0.0
    # judge is a model that was not competing, and saw nothing but anonymous submissions
    assert card["episode"]["judge"]["judge_model"] == "p3:m3" and card["episode"]["judge"]["conflict"] == "none"
    judge_prompt = next(s.prompt for mid, s in seen if s.system == judge.JUDGE_SYSTEM)
    assert "Submission 1:" in judge_prompt and "p0:m0" not in judge_prompt and "Model 0" not in judge_prompt
    assert "hidden_answer_key" not in json.dumps(card)


def test_human_scoring_foundation_and_mark_best(tiny_market, monkeypatch):
    tid = approved_market_task()
    e2e_pool(monkeypatch, {"p0:m0": good_answer(), "p1:m1": good_answer(), "p2:m2": good_answer()})
    final = next(e for e in run_task(tid, ["p0:m0", "p1:m1"]) if e["type"] == "arena_final")
    card = bench.scorecard(final["episode_id"])
    aid = card["attempts"][0]["attempt_id"]
    for crit in bench.HUMAN_CRITERIA:
        r = c.put(f"/api/bench/attempts/{aid}/human-score", json={"criterion": crit, "score": 4, "rater": "josh", "notes": "n"})
        assert r.status_code == 200
    c.put(f"/api/bench/attempts/{aid}/human-score", json={"criterion": "groundedness", "score": 2})     # re-score: latest wins
    hs = c.get(f"/api/bench/episodes/{final['episode_id']}/scorecard").json()["attempts"][0]["human_scores"]
    assert set(hs) == set(bench.HUMAN_CRITERIA) and hs["groundedness"]["score"] == 2 and hs["overall_quality"]["rater"] == "josh"
    for bad in ({"criterion": "vibes", "score": 3}, {"criterion": "groundedness", "score": 0}, {"criterion": "groundedness", "score": 6}):
        assert c.put(f"/api/bench/attempts/{aid}/human-score", json=bad).status_code == 400
    assert c.put("/api/bench/attempts/nope/human-score", json={"criterion": "groundedness", "score": 3}).status_code == 404
    # the original Mark best path still works and is reflected on the episode
    run_id = bench.get_episode(final["episode_id"])["arena_run_id"]
    assert c.put("/api/arena/winner", json={"run_id": run_id, "winner_id": "p1:m1"}).status_code == 200
    assert next(a for a in bench.scorecard(final["episode_id"])["attempts"] if a["winner"])["contestant"] == "p1:m1"


def test_failed_contestant_is_an_explicit_attempt_error_not_a_substitute(tiny_market, monkeypatch):
    tid = approved_market_task()
    texts = {"p0:m0": good_answer(), "p1:m1": good_answer(), "p2:m2": ""}
    pool = [{"id": f"p{i}:m{i}", "label": f"M{i}", "kind": "free", "provider": f"p{i}", "model": f"m{i}", "power": 3} for i in range(4)]
    monkeypatch.setattr(models, "enabled_pool", lambda: pool)
    calls = []

    async def fake(mm, spec):
        calls.append(mm["id"])
        if spec.system == judge.JUDGE_SYSTEM:
            return providers.RunResult("p3", "m3", True, text=json.dumps({"scores": {str(i): {cr: 4 for cr in judge.CRITERIA} for i in (1, 2)}, "ranking": [1, 2]}))
        if mm["id"] == "p2:m2":
            return providers.RunResult("p2", "m2", False, error={"kind": "context_exceeded", "message": "too long"})
        return providers.RunResult(mm["provider"], mm["model"], True, text=texts[mm["id"]])
    monkeypatch.setattr(providers, "run", fake)
    events = run_task(tid, ["p0:m0", "p1:m1", "p2:m2"])
    assert calls.count("p2:m2") == 1 and sorted(set(calls)) == ["p0:m0", "p1:m1", "p2:m2", "p3:m3"]     # nobody stepped in for p2
    failed = next(e for e in events if e["type"] == "model_done" and e["id"] == "p2:m2")
    assert failed["error"]["kind"] == "context_exceeded" and failed["text"].startswith("(failed: context_exceeded")
    final = next(e for e in events if e["type"] == "arena_final")
    card = bench.scorecard(final["episode_id"])
    bad = next(a for a in card["attempts"] if a["contestant"] == "p2:m2")
    assert bad["status"] == "error" and bad["error"]["kind"] == "context_exceeded" and bad["deterministic_scores"] == [] and bad["judge_scores"] == {}
    assert final["suggested"][-1] == "p2:m2"


def test_evidence_changed_since_registration_stops_the_run(tiny_market, monkeypatch):
    tid = approved_market_task()
    with open(tiny_market / "dataset1_books.csv", "a", encoding="utf-8") as f:
        f.write("D,w,$40.00,Cat2,http://u4\n")
    calls = []
    e2e_pool(monkeypatch, {})
    events = run_task(tid, ["p0:m0", "p1:m1"])
    assert events[-1]["type"] == "error" and "changed" in events[-1]["message"] and calls == []


def test_hidden_key_scores_but_never_reaches_a_contestant(monkeypatch):
    tid = uid("hk-")
    bench.add_benchmark("hk-bench", "HK")
    bench.add_task(tid, "hk-bench", "t", "Say the magic word.", scorers=[{"name": "exact_match", "kind": "deterministic"}],
                   hidden_answer_key={"expected": "XYZZY-HIDDEN"})
    seen = e2e_pool(monkeypatch, {"p0:m0": "xyzzy-hidden", "p1:m1": "nope", "p2:m2": "x"})
    final = next(e for e in run_task(tid, ["p0:m0", "p1:m1"]) if e["type"] == "arena_final")
    assert all("XYZZY-HIDDEN" not in s.prompt and "XYZZY-HIDDEN" not in s.system for _, s in seen)
    det = {a["contestant"]: a["deterministic_scores"][0]["passed"] for a in bench.scorecard(final["episode_id"])["attempts"]}
    assert det == {"p0:m0": True, "p1:m1": False}


def test_freeform_arena_runs_still_work_and_become_adhoc_episodes(monkeypatch):
    e2e_pool(monkeypatch, {"p0:m0": "a", "p1:m1": "b", "p2:m2": "c"})

    async def go():
        return [e async for e in arena.run("Plain question, no evidence", ["p0:m0", "p1:m1"])]
    events = asyncio.run(go())
    final = next(e for e in events if e["type"] == "arena_final")
    card = bench.scorecard(final["episode_id"])
    assert card["benchmark"]["id"] == "adhoc-arena" and card["episode"]["evidence_sha256"] is None
    assert store.list_arena()[0]["id"] == bench.get_episode(final["episode_id"])["arena_run_id"]
    assert [d["scorer"] for a in card["attempts"] for d in a["deterministic_scores"]] == []       # ad-hoc tasks have no scorers


# ---- endpoints --------------------------------------------------------------------------------------------------
def test_bench_endpoints(tiny_market):
    reg = c.post("/api/bench/register-market").json()
    assert reg["task_id"].startswith("market-intel-")
    assert any(b["id"] == "market-intelligence" for b in c.get("/api/bench/benchmarks").json())
    tasks = c.get("/api/bench/tasks", params={"benchmark_id": "market-intelligence"}).json()
    assert tasks and all("hidden_answer_key" not in t for t in tasks)
    assert next(t for t in tasks if t["id"] == reg["task_id"])["status"] == "awaiting_task_definition_approval"
    assert c.get("/api/bench/episodes/nope/scorecard").status_code == 404
    assert c.get("/api/bench/attempts/nope/trace").status_code == 404
    _, aid = make_attempt()
    assert [e["kind"] for e in c.get(f"/api/bench/attempts/{aid}/trace").json()][0] == "prompt"
    assert set(c.post("/api/bench/link-legacy").json()) == {"linked", "skipped"}
