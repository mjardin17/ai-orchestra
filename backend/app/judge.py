"""Blind LLM judge. The judge sees "Submission 1..n" in a shuffled order and nothing that says who wrote what.
The label -> attempt mapping stays inside this app (stored on the episode and the attempts), never in the prompt.
The judge model is chosen from the enabled pool, independent of the contestants where the pool allows it; when it
can't be, the conflict is recorded rather than hidden."""
import hashlib, json, random, re
from . import providers
from .providers import RunSpec

CRITERIA = ("groundedness", "citation_validity", "instruction_following", "overall_quality")
RUBRIC = """Score every submission from 1 (worst) to 5 (best) on each criterion:
- groundedness: every claim is supported by the supplied evidence; nothing is invented.
- citation_validity: cited rows and quoted values really appear in the evidence as cited (check them against the evidence).
- instruction_following: follows the request and the required output format.
- overall_quality: overall usefulness and clarity.
Judge only what is written. Do not guess who wrote a submission and do not reward length or style."""
JUDGE_SYSTEM = ("You are an impartial evaluator comparing anonymous submissions to the same request. " + RUBRIC +
                '\nReply with ONLY a JSON object: {"scores": {"1": {"groundedness": 4, "citation_validity": 3, '
                '"instruction_following": 5, "overall_quality": 4}, ...}, "ranking": [2, 1, ...] (best first, every submission '
                'number once), "reasoning": {"1": "one or two sentences", ...}}.')

# Best-effort scrubbing of obvious self-identification. Writing style can still leak identity; that is reported as a risk.
SELF_ID = re.compile(
    r"\b(I am|I'm|this is|as|created by|made by|developed by|trained by)\s+(?:an?\s+)?(?:(?:AI|assistant|model|language model)\s+)?"
    r"(?:called |named |by )?(Claude|Anthropic|ChatGPT|GPT[-\w.]*|OpenAI|Gemini|Google|Grok|xAI|Llama|Meta|Mistral|DeepSeek|Groq)\b", re.I)


def identity_strings(contestants: list) -> list:
    """Strings that name a contestant: label, catalog id, provider-qualified ids, and distinctive model ids."""
    out = set()
    for c in contestants:
        for s in (c.get("label"), c.get("id")):
            if s:
                out.add(s)
        model = c.get("model") or (c.get("id", "").split(":", 1)[-1])
        if model and re.search(r"[-.\d]", model):       # skip plain words like "haiku" that are also normal vocabulary
            out.add(model)
    return sorted((s for s in out if len(s) >= 3), key=len, reverse=True)


def redact(text: str, identities: list):
    """(scrubbed text, number of replacements)."""
    n = 0
    for s in identities:
        text, k = re.subn(r"(?<![\w-])" + re.escape(s) + r"(?![\w-])", "[model]", text, flags=re.I)
        n += k
    text, k = SELF_ID.subn(lambda m: m.group(0)[: m.start(2) - m.start(0)] + "[model]", text)
    return text, n + k


def anonymize(attempts: list, contestants: list, seed: int):
    """attempts: [{'id', 'output', ...}] of successful attempts. Returns (submissions, mapping).
    submissions: [{'label': 1.., 'text': scrubbed}]  (what the judge sees)
    mapping:     {label: attempt_id}                 (internal only)"""
    order = list(attempts)
    random.Random(seed).shuffle(order)
    idents = identity_strings(contestants)
    subs, mapping = [], {}
    for i, a in enumerate(order, 1):
        text, n = redact(a["output"], idents)
        subs.append({"label": i, "text": text, "redactions": n})
        mapping[i] = a["id"]
    return subs, mapping


def build_prompt(task_text: str, evidence_text, submissions: list) -> str:
    parts = [f"Request given to every submission:\n{task_text}"]
    if evidence_text:
        parts.append("Every submission was given exactly this evidence:\n" + evidence_text)
    parts.append("\n\n".join(f"Submission {s['label']}:\n{s['text']}" for s in submissions))
    return "\n\n".join(parts)


class JudgeParseError(ValueError):
    pass


def parse_verdict(text: str, labels: list) -> dict:
    """Validate the judge's JSON against the submission labels. Raises JudgeParseError rather than guessing."""
    from .scoring import extract_json
    obj, err = extract_json(text)
    if obj is None:
        raise JudgeParseError(err)
    want = {str(l) for l in labels}
    scores = obj.get("scores")
    if not isinstance(scores, dict) or set(map(str, scores)) != want:
        raise JudgeParseError(f"scores must cover exactly submissions {sorted(want)}")
    clean = {}
    for l, row in scores.items():
        if not isinstance(row, dict) or any(c not in row for c in CRITERIA):
            raise JudgeParseError(f"submission {l} is missing a criterion")
        for c in CRITERIA:
            v = row[c]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5:
                raise JudgeParseError(f"submission {l} {c} must be 1-5")
        clean[int(l)] = {c: float(row[c]) for c in CRITERIA}
    ranking = obj.get("ranking")
    if not isinstance(ranking, list) or sorted(map(str, ranking)) != sorted(want):
        raise JudgeParseError("ranking must list every submission exactly once")
    reasoning = obj.get("reasoning") if isinstance(obj.get("reasoning"), dict) else {}
    return {"scores": clean, "ranking": [int(x) for x in ranking], "reasoning": {int(k): str(v) for k, v in reasoning.items()}}


def pick_judge(contestants: list, pool: list, preferred_id=None):
    """(judge model or None, conflict). conflict: 'none' (different provider), 'same_provider', 'same_model', or 'no_judge_available'.
    A judge configured by the user (setting judge_model) wins; its conflict level is still reported. Never hard-codes a vendor."""
    c_ids = {c["id"] for c in contestants}
    c_provs = {providers.provider_of(c) for c in contestants}

    def level(m):
        return 2 if m["id"] in c_ids else 1 if providers.provider_of(m) in c_provs else 0
    names = {0: "none", 1: "same_provider", 2: "same_model"}
    if preferred_id:
        m = next((x for x in pool if x["id"] == preferred_id), None)
        if m:
            return m, names[level(m)]
    if not pool:
        return None, "no_judge_available"
    best = min(pool, key=lambda m: (level(m), not providers.support_status(m)["supported"], -m.get("power", 0), m["id"]))
    return best, names[level(best)]


async def run_judge(task_text: str, evidence_text, attempts: list, contestants: list, pool: list, seed: int,
                    preferred_id=None, max_tokens=None, timeout=None) -> dict:
    """Judge the successful attempts. Returns a JSON-able record; never raises for provider or parse problems."""
    base = {"status": "skipped", "seed": seed, "criteria": list(CRITERIA)}
    if len(attempts) < 2:
        return {**base, "reason": "fewer than two successful submissions to compare"}
    model, conflict = pick_judge(contestants, pool, preferred_id)
    if model is None:
        return {**base, "status": "error", "error": {"kind": "not_configured", "message": "no judge model is enabled"}, "conflict": conflict}
    subs, mapping = anonymize(attempts, contestants, seed)
    spec = RunSpec(JUDGE_SYSTEM, build_prompt(task_text, evidence_text, subs), max_tokens=max_tokens, timeout=timeout)
    res = await providers.run(model, spec)
    rec = {**base, "judge_model": model["id"], "judge_provider": providers.provider_of(model), "conflict": conflict,
           "mapping": {str(k): v for k, v in mapping.items()}, "redactions": {str(s["label"]): s["redactions"] for s in subs},
           "judge_model_version": res.model_version, "latency": res.latency, "usage": res.usage}
    if not res.ok:
        return {**rec, "status": "error", "error": res.error}
    try:
        v = parse_verdict(res.text, list(mapping))
    except JudgeParseError as e:
        return {**rec, "status": "error", "error": {"kind": "provider_error", "message": f"unparseable judge reply: {e}"},
                "raw_reply_sha256": hashlib.sha256(res.text.encode()).hexdigest()}
    return {**rec, "status": "ok", "scores": {str(k): v_ for k, v_ in v["scores"].items()}, "ranking": v["ranking"],
            "reasoning": {str(k): t for k, t in v["reasoning"].items()}}
