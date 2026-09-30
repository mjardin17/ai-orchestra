"""Side-by-side model comparison. Run one prompt on several chosen models, judge picks a suggested best,
you can override it. Every result is stored and can be exported as training-style pairs (weak vs strong output)
for later use if you ever want to fine-tune your own model outside this app."""
import asyncio, json, uuid
from . import models, store
from .prompts import DIRECT

JUDGE = """You are comparing answers from several models to the same request. Rank them best to worst for
accuracy, completeness and clarity. Reply with only a JSON array of the option letters, best first, e.g. ["B","A","C"]."""


async def run(prompt: str, model_ids: list):
    run_id = uuid.uuid4().hex[:8]
    yield {"type": "arena_run", "id": run_id}
    pool = {m["id"]: m for m in models.enabled_pool()}
    chosen = [pool[i] for i in model_ids if i in pool]
    if len(chosen) < 2:
        yield {"type": "error", "message": "Pick at least 2 available models to compare."}
        return

    q, results = asyncio.Queue(), {}

    async def work(m):
        try:
            results[m["id"]] = await models.run_strict(m, DIRECT, prompt)
        except Exception as e:
            results[m["id"]] = f"(failed: {e})"
        await q.put({"type": "model_done", "id": m["id"], "label": m["label"], "text": results[m["id"]]})

    tasks = [asyncio.create_task(work(m)) for m in chosen]
    for _ in tasks:
        yield await q.get()
    await asyncio.gather(*tasks, return_exceptions=True)

    letters = [chr(65 + i) for i in range(len(chosen))]
    block = "\n\n".join(f"Option {L} ({m['label']}):\n{results[m['id']]}" for L, m in zip(letters, chosen))
    ranking = [m["id"] for m in chosen]
    try:
        from .claude_client import run_agent
        verdict = await run_agent(JUDGE, f"Request: {prompt}\n\n{block}", "main")
        order = json.loads(verdict[verdict.index("["):verdict.rindex("]") + 1])
        idx = {L: m["id"] for L, m in zip(letters, chosen)}
        ranking = [idx[L] for L in order if L in idx] or ranking
    except Exception:
        pass  # judge failed to parse: leave the run order as the suggested ranking

    store.save_arena(run_id, prompt, [{"id": m["id"], "label": m["label"], "text": results[m["id"]]} for m in chosen], ranking[0])
    yield {"type": "arena_final", "suggested": ranking}


def set_winner(run_id, winner_id):
    store.set_arena_winner(run_id, winner_id)


def export_jsonl():
    """One line per comparison: the request, the suggested-best (or your picked) output as 'chosen',
    and every other output as 'rejected'. This is the shape most fine-tuning/DPO tools expect."""
    lines = []
    for r in store.list_arena(full=True):
        results = {x["id"]: x for x in r["results"]}
        winner = r.get("winner")
        if winner not in results:
            continue
        chosen = results[winner]
        for other_id, other in results.items():
            if other_id == winner:
                continue
            lines.append(json.dumps({"prompt": r["prompt"], "chosen": chosen["text"], "chosen_model": chosen["label"],
                                     "rejected": other["text"], "rejected_model": other["label"]}))
    return "\n".join(lines)
