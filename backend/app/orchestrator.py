import asyncio, json, os, re, uuid
from . import brain, lessons, models, skill_selector, store
from .claude_client import run_agent
from .prompts import CHECK, DIRECT, PROMPTS, REVISE, SUPERVISOR, SYNTH

LIMIT = asyncio.Semaphore(3)  # cap parallel calls to protect plan and free-tier limits
DEFAULT_STAGES = [["research"], ["writer"]]
NEED = {"easy": 2, "medium": 3, "hard": 4}  # minimum capability a model needs for each difficulty
MAX_ATTEMPTS = 4


def parse_route(text: str) -> dict:
    fallback = {"mode": "team", "stages": DEFAULT_STAGES, "priority": "medium", "difficulty": {}, "models": {},
                "reason": "Router reply was not valid JSON; used the default route."}
    try:
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        diff = {k: v for k, v in (d.get("difficulty") or {}).items() if v in NEED}
        base = {"reason": d.get("reason", ""), "priority": d.get("priority", "medium"), "difficulty": diff, "models": {}}
        if d.get("mode") == "direct":
            return {**base, "mode": "direct", "stages": []}
        seen, stages = set(), []
        for st in d.get("stages", [])[:3]:
            names = [a for a in st if a in PROMPTS and a not in seen]
            seen.update(names)
            if names:
                stages.append(names)
        # fact-check always runs last, alone, so it has something to check
        stages = [[a for a in st if a != "factcheck"] for st in stages]
        stages = [st for st in stages if st]
        if "factcheck" in seen and stages:
            stages.append(["factcheck"])
        return {**base, "mode": "team", "stages": stages or DEFAULT_STAGES}
    except Exception:
        return fallback


def ladder_for(key: str, route: dict, pool: list, tools: bool = False):
    """Models to try for a job, cheapest capable first. Returns (ladder, quality_check_allowed)."""
    ids = {m["id"]: m for m in pool}
    allow = store.get_setting("allow_private", "0") == "1"
    claude_only = route.get("private") and not allow
    pin = json.loads(store.get_setting("pins", "{}") or "{}").get(key)
    if pin in ids and not (claude_only and ids[pin]["kind"] != "claude"):
        return [ids[pin], models.find("claude:haiku")], False  # you chose it: no second-guessing, Haiku only on failure
    need = NEED.get(route.get("difficulty", {}).get(key, "medium"), 3)
    cands = [m for m in pool if not (claude_only and m["kind"] != "claude") and not (tools and m["kind"] != "claude")]
    ladder = sorted((m for m in cands if m["power"] >= need), key=lambda m: (m["cost"], m["power"]))
    if not ladder:  # nothing capable enough is enabled: use the strongest thing available
        ladder = sorted(cands, key=lambda m: -m["power"])[:1] or [models.find("claude:sonnet")]
    return ladder, need > NEED["easy"]


async def passes(request: str, text: str, hard: bool) -> bool:
    try:
        v = await run_agent(CHECK, f"Task:\n{request[:3000]}\n\nOutput to judge:\n{text[:6000]}", "main" if hard else "fast")
        return not v.strip().upper().startswith("ESCALATE")
    except Exception:
        return True


async def solve(agent, key, route, pool, system, prompt, request, run_id, tools=False):
    """The council's loop: cheapest capable model first, escalate on failure or a failed quality check."""
    ladder, checkable = ladder_for(key, route, pool, tools)
    system = lessons.with_lessons(system, key)  # rules learned from earlier failures
    hard = route.get("difficulty", {}).get(key) == "hard"
    tries, note, escalated, best, err = 0, "Working", False, None, None
    while ladder and tries < MAX_ATTEMPTS:
        m = ladder.pop(0)
        tries += 1
        yield {"type": "agent_start", "agent": agent, "model": m["label"], "note": note}
        try:
            text = await models.run_strict(m, system, prompt, tools)
        except Exception as e:
            lessons.record_failure(key, m["id"], e, request)
            err, note, escalated = e, "Retrying", True
            continue
        if checkable and ladder and tries < MAX_ATTEMPTS and not await passes(request, text, hard):
            best = best or text
            lessons.record_failure(key, m["id"], "output judged insufficient by the quality check", request)
            ladder = [x for x in ladder if x["power"] > m["power"]]
            if ladder:
                note, escalated = "Escalated", True
                continue
        store.record_use(m["id"], m["label"], escalated, run_id, m.get("power", 3))
        yield {"type": "result", "text": text, "label": m["label"] + (" (escalated)" if escalated else "")}
        return
    if best:
        store.record_use(None, "cheaper model (escalation failed)", True, run_id)
        yield {"type": "result", "text": best, "label": "cheaper model (escalation failed)"}
        return
    raise RuntimeError(f"all models failed: {err}")


async def step(agent, key, route, pool, system, prompt, request, run_id, tools=False, show_output=True):
    text = label = None
    async for ev in solve(agent, key, route, pool, system, prompt, request, run_id, tools):
        if ev["type"] == "result":
            text, label = ev["text"], ev["label"]
        else:
            yield ev
    yield {"type": "agent_done", "agent": agent, "output": text if show_output else None, "model": label}
    yield {"type": "_result", "text": text}


def plan_models(route: dict, pool: list) -> dict:
    names = ["direct"] if route["mode"] == "direct" else [a for st in route["stages"] for a in st] + ["synthesis"]
    return {n: ladder_for(n, route, pool, tools=(n == "research"))[0][0]["id"] for n in names}


def notes_from(outputs: dict) -> str:
    return "\n\n".join(f"## {n}\n{o}" for n, o in outputs.items())


async def orchestrate(prompt: str, ws_id=None):
    """ws_id=None runs statelessly (no shared context, nothing saved), which the eval script uses."""
    run_id = uuid.uuid4().hex[:8]
    tally_id = run_id if ws_id else None  # only tally saved runs, so the eval script does not skew it
    yield {"type": "run", "id": run_id}
    live: list = []
    try:
        pool = models.enabled_pool()
        if not pool:
            raise RuntimeError("No models are enabled. Turn at least one on in the Models tab.")
        ctx = store.context(ws_id)
        skill_text, picked = skill_selector.for_job(prompt, ws_id)  # analyze the job, pick the best saved skills
        if picked:
            ctx = skill_text + chr(10) * 2 + ctx if ctx else skill_text
            yield {"type": "skills", "picked": [{"name": p["name"], "matched": p["matched"]} for p in picked]}
        full = f"{ctx}\n\nCurrent request:\n{prompt}" if ctx else prompt
        route = parse_route(await run_agent(SUPERVISOR, full[:4000], os.getenv("ROUTER_TIER", "fast")))
        route["private"] = store.has_private(ws_id)
        route["models"] = plan_models(route, pool)  # where each job STARTS; escalation may move it up
        yield {"type": "route", **route}
        outputs: dict = {}

        if route["mode"] == "direct":  # fast path: one job, no synthesis
            yield {"type": "direct_start"}
            async for ev in step("direct", "direct", route, pool, DIRECT, full, prompt, tally_id, show_output=False):
                if ev["type"] == "_result":
                    final = ev["text"]
                else:
                    yield ev
        else:
            for stage in route["stages"]:
                q = asyncio.Queue()
                context = notes_from(outputs)
                task_prompt = f"Request: {full}\n\nEarlier work from teammates:\n{context}" if context else full

                async def work(name):
                    async with LIMIT:
                        try:
                            async for ev in step(name, name, route, pool, PROMPTS[name], task_prompt, prompt, tally_id, tools=(name == "research")):
                                if ev["type"] == "_result":
                                    outputs[name] = ev["text"]
                                else:
                                    await q.put(ev)
                        except Exception as e:
                            outputs[name] = f"(failed: {e})"
                            await q.put({"type": "agent_done", "agent": name, "output": outputs[name], "model": None})
                        finally:
                            await q.put({"type": "_end"})

                tasks = [asyncio.create_task(work(a)) for a in stage]
                live += tasks
                ended = 0
                while ended < len(tasks):
                    ev = await q.get()
                    if ev["type"] == "_end":
                        ended += 1
                    else:
                        yield ev

            # one revision pass if the fact-checker asked for it
            synth_src = dict(outputs)
            target = next((a for a in ("writer", "coder", "planner") if a in outputs), None)
            if target and "VERDICT: REVISE" in outputs.get("factcheck", ""):
                rev = f"{target}-revision"
                try:
                    async for ev in step(rev, target, route, pool, f"{PROMPTS[target]}\n\n{REVISE}",
                                         f"Request: {full}\n\nYour earlier draft:\n{outputs[target]}\n\nFact-check flags:\n{outputs['factcheck']}", prompt, tally_id):
                        if ev["type"] == "_result":
                            outputs[rev] = ev["text"]
                            synth_src.pop(target)
                            synth_src[rev] = outputs[rev]
                        else:
                            yield ev
                except Exception as e:
                    outputs[rev] = f"(failed: {e})"
                    yield {"type": "agent_done", "agent": rev, "output": outputs[rev], "model": None}

            yield {"type": "synth_start"}
            async for ev in step("synthesis", "synthesis", route, pool, SYNTH,
                                 f"Request: {full}\n\nSpecialist notes:\n{notes_from(synth_src)}", prompt, tally_id, show_output=False):
                if ev["type"] == "_result":
                    final = ev["text"]
                else:
                    yield ev

        if ws_id:
            store.save_run(run_id, ws_id, prompt, route, outputs, final)
            brain.tune()  # cheap local aggregation, no model calls: capability ratings keep improving with use
        yield {"type": "final", "text": final}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
    finally:
        for t in live:  # client hit Stop or disconnected: don't keep burning calls
            t.cancel()
