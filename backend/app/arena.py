"""Side-by-side model comparison. Run one task on several chosen models independently, score the answers
(deterministic checks first, then a blind judge), and let you override the suggested best. Every comparison is stored
as a benchmark episode (see bench.py) and can still be exported as training-style pairs (weak vs strong output).

Independence: each contestant gets the same system prompt, the same task text and the same evidence, in its own call.
Nobody sees another contestant's answer before submitting; the judge sees the answers only after all are in, and only
as anonymous "Submission N" text."""
import asyncio, json, os, random, uuid
from . import bench, judge, market_data, models, providers, scoring, store
from .prompts import DIRECT

ARENA_MAX_TOKENS = int(os.getenv("ARENA_MAX_TOKENS", "4096"))   # output limit for each contestant and the judge
ARENA_TIMEOUT = float(os.getenv("ARENA_TIMEOUT", "300"))        # seconds per call

MARKET_SYSTEM = (DIRECT + " You are given real market-intelligence evidence. Base every claim only on that evidence and cite the row ID "
                 "(for example dataset1_books.csv:2) for each fact you use. Do not invent figures, sales numbers or sources; "
                 "if the evidence cannot support a claim, say so. Work independently.")


def build_prompt(task: str, evidence_text: str) -> str:
    """The one prompt every participating model receives: task first, then the identical evidence block."""
    return f"{task}\n\n{evidence_text}"


def _failed_text(err):
    return f"(failed: {err['kind']}: {err['message']})"


async def run(prompt: str, model_ids: list, market: bool = False, task_id: str = None):
    run_id = uuid.uuid4().hex[:8]
    yield {"type": "arena_run", "id": run_id}
    system, task_text, evidence, task = DIRECT, prompt, None, None
    if task_id:
        task = bench.get_task(task_id)
        if not task:
            yield {"type": "error", "message": f"Unknown task: {task_id}"}
            return
        if task["status"] != bench.STATUS_APPROVED:
            yield {"type": "error", "message": f"Task {task_id} is {task['status']}: its definition is not final, so it cannot be run yet."}
            return
        task_text, market = bench.build_task_prompt(task), task["evidence_provider"] == "market_data"
    elif not (prompt or "").strip():
        yield {"type": "error", "message": "A prompt or a task is required."}
        return
    if market:
        try:
            evidence = market_data.build_evidence()
        except Exception as e:  # missing dir/files or over budget: stop rather than run on partial or sample data
            yield {"type": "error", "message": f"Market evidence unavailable: {e}"}
            return
        if task and task["evidence_sha256"] != evidence["sha256"]:
            yield {"type": "error", "message": "The evidence files changed since this task was registered; register the benchmark again."}
            return
        system = MARKET_SYSTEM
        yield {"type": "evidence", "source": evidence["source"], "sha256": evidence["sha256"], "chars": evidence["chars"],
               "est_tokens": evidence["est_tokens"], "files": {k: v["records"] for k, v in evidence["files"].items()}}
    full_prompt = build_prompt(task_text, evidence["text"]) if evidence else task_text

    pool = {m["id"]: m for m in models.enabled_pool()}
    chosen = [pool[i] for i in model_ids if i in pool]
    if len(chosen) < 2:
        yield {"type": "error", "message": "Pick at least 2 available models to compare."}
        return

    ev_sha = evidence["sha256"] if evidence else None
    if not task:
        task = bench.get_task(bench.ensure_adhoc_task(task_text, ev_sha, "market_data" if evidence else None))
    seed = random.SystemRandom().randrange(2**31)
    episode_id = bench.create_episode(task["id"], ev_sha, run_id, seed)
    yield {"type": "episode", "id": episode_id, "task_id": task["id"]}

    spec = providers.RunSpec(system, full_prompt, max_tokens=ARENA_MAX_TOKENS, timeout=ARENA_TIMEOUT)
    q, results, attempt_ids = asyncio.Queue(), {}, {}

    async def work(m):  # one independent call per contestant; no shared state
        res = await providers.run(m, spec)
        results[m["id"]] = res
        await q.put(m)

    tasks = [asyncio.create_task(work(m)) for m in chosen]
    for _ in tasks:
        m = await q.get()
        res = results[m["id"]]
        attempt_ids[m["id"]] = bench.record_attempt(episode_id, m, res.to_dict(), system, full_prompt, ev_sha)
        yield {"type": "model_done", "id": m["id"], "label": m["label"], "text": res.text if res.ok else _failed_text(res.error),
               "attempt_id": attempt_ids[m["id"]], **({"error": res.error} if not res.ok else {})}
    await asyncio.gather(*tasks, return_exceptions=True)

    ok = [m for m in chosen if results[m["id"]].ok]
    ok_attempts = [bench.get_attempt(attempt_ids[m["id"]]) for m in ok]

    # 1) deterministic scoring (no model involved)
    if task.get("scorers"):
        answer_key = (bench.get_task(task["id"], include_hidden=True) or {}).get("hidden_answer_key")   # scorers only, never prompts
        index = scoring.EVIDENCE_PROVIDERS[task["evidence_provider"]]() if task.get("evidence_provider") in scoring.EVIDENCE_PROVIDERS else None
        for m, a in zip(ok, ok_attempts):
            scored = scoring.evaluate_attempt(episode_id, a, task, index, answer_key)
            yield {"type": "deterministic", "id": m["id"], "scores": [
                {"scorer": s.scorer, "score": s.score, "passed": s.passed, "applicable": s.applicable} for s in scored]}

    # 2) blind judge: anonymous, shuffled, never told who wrote what
    contestants = [{"id": m["id"], "label": m["label"], "provider": providers.provider_of(m), "model": m.get("model")} for m in chosen]
    verdict = await judge.run_judge(task_text, evidence["text"] if evidence else None, ok_attempts, contestants,
                                    list(pool.values()), seed, store.get_setting("judge_model", "") or None, ARENA_MAX_TOKENS, ARENA_TIMEOUT)
    bench.set_episode_judge(episode_id, verdict)
    ranking = [m["id"] for m in ok] + [m["id"] for m in chosen if not results[m["id"]].ok]
    if verdict["status"] == "ok":
        bench.set_anon_labels({int(k): v for k, v in verdict["mapping"].items()})
        for label, aid in verdict["mapping"].items():
            for crit, val in verdict["scores"][label].items():
                bench.add_evaluation(episode_id, aid, "judge", crit, val, 5.0, None, True, {
                    "reasoning": verdict["reasoning"].get(label), "submission": int(label), "judge_model": verdict["judge_model"],
                    "judge_model_version": verdict["judge_model_version"], "conflict": verdict["conflict"], "seed": seed})
        cid = {attempt_ids[m["id"]]: m["id"] for m in ok}
        ranking = [cid[verdict["mapping"][str(l)]] for l in verdict["ranking"]] + [m["id"] for m in chosen if not results[m["id"]].ok]
    yield {"type": "judge", "status": verdict["status"], "conflict": verdict.get("conflict"), "judge_model": verdict.get("judge_model"),
           **({"error": verdict["error"]} if verdict.get("error") else {}), **({"reason": verdict["reason"]} if verdict.get("reason") else {})}

    saved = f"{task_text}\n\n[market evidence: {evidence['source']} sha256={evidence['sha256']}]" if evidence else task_text
    store.save_arena(run_id, saved, [
        {"id": m["id"], "label": m["label"], "text": results[m["id"]].text if results[m["id"]].ok else _failed_text(results[m["id"]].error),
         **({"error": results[m["id"]].error["kind"]} if not results[m["id"]].ok else {})} for m in chosen], ranking[0])
    store.DB.execute("update arena_runs set episode_id=? where id=?", (episode_id, run_id))
    store.DB.commit()
    bench.set_episode_status(episode_id, "complete")
    yield {"type": "arena_final", "suggested": ranking, "episode_id": episode_id, "judge_status": verdict["status"]}


def set_winner(run_id, winner_id):
    store.set_arena_winner(run_id, winner_id)
    bench.set_winner_for_run(run_id, winner_id)


def export_jsonl():
    """One line per comparison: the request, the suggested-best (or your picked) output as 'chosen',
    and every other output as 'rejected'. This is the shape most fine-tuning/DPO tools expect.
    Failed attempts are never exported as training text."""
    lines = []

    def usable(x):
        return not x.get("error") and not str(x.get("text", "")).startswith("(failed:")
    for r in store.list_arena(full=True):
        results = {x["id"]: x for x in r["results"]}
        winner = r.get("winner")
        if winner not in results or not usable(results[winner]):
            continue
        chosen = results[winner]
        for other_id, other in results.items():
            if other_id == winner or not usable(other):
                continue
            lines.append(json.dumps({"prompt": r["prompt"], "chosen": chosen["text"], "chosen_model": chosen["label"],
                                     "rejected": other["text"], "rejected_model": other["label"]}))
    return "\n".join(lines)
