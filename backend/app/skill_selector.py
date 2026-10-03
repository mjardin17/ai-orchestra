"""Picks the best saved skills for the job at hand, with no model call and no API key.
Scores each skill by how many of the job's meaningful words appear in the skill's name (weighted x3)
and body, so what gets picked is explainable ('matched: ebay, listing'). Nothing is picked unless a skill
actually matches, so unrelated jobs aren't padded with irrelevant instructions."""
import re
from . import store

NAME_WEIGHT = 3
MIN_SCORE = 2          # one stray body word is not enough to pick a skill
DEFAULT_K = 3
STOP = set("""a an and are as at be by can do for from has have how i in is it its me my of on or our so that the
their this to was we what when where which who why will with you your please make write create get give help""".split())


def _terms(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in STOP}


def select_skills(prompt, skills, k=DEFAULT_K):
    job = _terms(prompt)
    if not job or not skills:
        return []
    ranked = []
    for sk in skills:
        name_hits = job & _terms(sk.get("name"))
        body_hits = (job & _terms(sk.get("text"))) - name_hits
        score = NAME_WEIGHT * len(name_hits) + len(body_hits)
        if score >= MIN_SCORE:
            ranked.append({"id": sk["id"], "name": sk.get("name"), "score": score,
                           "matched": sorted(name_hits | body_hits)})
    ranked.sort(key=lambda r: (-r["score"], r["name"] or ""))
    return ranked[:k]


def _library():
    return [dict(r) for r in store.DB.execute("select id, name, text from skills order by rowid")]


def select_for(prompt, k=DEFAULT_K):
    return select_skills(prompt, _library(), k)


def skill_block(prompt, skills, limit=6000, k=DEFAULT_K):
    """Prompt text carrying the chosen skills, or '' when none fit."""
    chosen = select_skills(prompt, skills, k)
    if not chosen:
        return ""
    by_id = {s["id"]: s for s in skills}
    parts = [f"Skill \"{c['name']}\" (matched: {', '.join(c['matched'])}):\n{by_id[c['id']]['text']}" for c in chosen]
    return ("Skills picked for this job (follow them):\n\n" + "\n\n".join(parts))[:limit]


def for_job(prompt, ws_id=None, k=DEFAULT_K):
    """(prompt text, picked list) for a run: best-matching library skills the workspace hasn't already pinned."""
    import json
    pinned = set()
    if ws_id:
        ws = store.DB.execute("select skills from workspaces where id=?", (ws_id,)).fetchone()
        pinned = set(json.loads(ws["skills"] or "[]")) if ws else set()
    library = [s for s in _library() if s["id"] not in pinned]
    picked = select_skills(prompt, library, k)
    return (skill_block(prompt, library, k=k) if picked else ""), picked
