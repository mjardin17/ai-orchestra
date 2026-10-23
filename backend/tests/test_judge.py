"""Blind judge: anonymity (tested directly and end to end), shuffling, judge selection, explicit failures."""
import asyncio, json
import pytest
from app import arena, bench, judge, models, providers, store

CONTESTANTS = [
    {"id": "claude:sonnet", "label": "Claude Sonnet", "provider": "claude", "model": None},
    {"id": "openai:gpt-5", "label": "GPT-5 (OpenAI API)", "provider": "openai", "model": "gpt-5"},
    {"id": "gemini:gemini-flash-latest", "label": "Gemini Flash", "provider": "gemini", "model": "gemini-flash-latest"},
    {"id": "xai:grok-4", "label": "Grok 4 (xAI API)", "provider": "xai", "model": "grok-4"},
]
LEAKS = ["claude", "sonnet", "openai", "gpt-5", "gemini", "grok", "xai", "anthropic", "[model]"]


def attempts(texts):
    return [{"id": f"att{i}", "output": t} for i, t in enumerate(texts)]


# ---- anonymization ----------------------------------------------------------------------------------------------
def test_identity_strings_are_scrubbed_from_submissions():
    texts = ["Claude Sonnet here: niche X is strong (claude:sonnet).", "As GPT-5 (OpenAI API) I think gpt-5 would say Y.",
             "I am Gemini, made by Google. gemini-flash-latest says Z.", "Grok 4 (xAI API) / grok-4 reports W. I'm Grok."]
    subs, mapping = judge.anonymize(attempts(texts), CONTESTANTS, seed=7)
    blob = " ".join(s["text"] for s in subs).lower()
    for leak in ("claude sonnet", "claude:sonnet", "gpt-5", "openai api", "gemini-flash-latest", "grok 4", "grok-4", "xai api", "i am gemini", "i'm grok"):
        assert leak not in blob, leak
    assert all(s["redactions"] >= 1 for s in subs)
    assert sorted(s["label"] for s in subs) == [1, 2, 3, 4]


def test_mapping_is_internal_and_complete():
    subs, mapping = judge.anonymize(attempts(["a", "b", "c"]), CONTESTANTS, seed=1)
    assert sorted(mapping.values()) == ["att0", "att1", "att2"]
    assert all("att" not in s["text"] for s in subs)
    assert set(subs[0]) == {"label", "text", "redactions"}            # nothing that points back to a contestant


def test_shuffle_is_seeded_and_actually_reorders():
    texts = [f"text-{i}" for i in range(6)]
    run = lambda seed: [m for _, m in sorted(judge.anonymize(attempts(texts), CONTESTANTS, seed)[1].items())]
    assert run(5) == run(5)
    identity = tuple(f"att{i}" for i in range(6))
    orders = {tuple(run(s)) for s in range(30)}
    assert len(orders) > 1 and any(o != identity for o in orders)


def test_judge_prompt_uses_generic_labels_only():
    subs, _ = judge.anonymize(attempts(["Claude Sonnet says one", "GPT-5 says two"]), CONTESTANTS, 3)
    prompt = judge.build_prompt("the task", "THE EVIDENCE", subs)
    assert "Submission 1:" in prompt and "Submission 2:" in prompt
    low = prompt.lower()
    assert "claude" not in low and "gpt-5" not in low
    assert "THE EVIDENCE" in prompt and "the task" in prompt


# ---- verdict parsing --------------------------------------------------------------------------------------------
def verdict(n=2, **over):
    v = {"scores": {str(i): {c: 4 for c in judge.CRITERIA} for i in range(1, n + 1)}, "ranking": list(range(1, n + 1)), "reasoning": {"1": "ok"}}
    v.update(over)
    return json.dumps(v)


def test_parse_verdict_accepts_valid_and_rejects_bad():
    ok = judge.parse_verdict("```json\n" + verdict(3) + "\n```", [1, 2, 3])
    assert ok["ranking"] == [1, 2, 3] and ok["scores"][2]["groundedness"] == 4.0
    bad = [verdict(3, ranking=[1, 2]), verdict(2, ranking=[1, 1]), verdict(1), "not json",
           json.dumps({"scores": {"1": {"groundedness": 9, "citation_validity": 1, "instruction_following": 1, "overall_quality": 1},
                                   "2": {c: 3 for c in judge.CRITERIA}}, "ranking": [1, 2]}),
           json.dumps({"scores": {"1": {"groundedness": 3}, "2": {c: 3 for c in judge.CRITERIA}}, "ranking": [1, 2]})]
    for b in bad:
        with pytest.raises(judge.JudgeParseError):
            judge.parse_verdict(b, [1, 2] if b is not bad[0] else [1, 2, 3])


# ---- judge selection --------------------------------------------------------------------------------------------
def m(mid, kind="free", provider=None, power=3):
    return {"id": mid, "label": mid, "kind": kind, "provider": provider or mid.split(":")[0], "model": mid.split(":")[-1], "power": power}


def test_judge_prefers_a_different_provider_than_every_contestant():
    contestants = [m("claude:haiku", "claude"), m("claude:sonnet", "claude")]
    pool = contestants + [m("claude:opus", "claude", power=5), m("gemini:flash", power=3), m("openai:gpt-5", power=5)]
    pool = [{**x, "kind": "claude", "tier": "x"} if x["id"].startswith("claude") else x for x in pool]
    chosen, conflict = judge.pick_judge(contestants, pool)
    assert chosen["id"] == "openai:gpt-5" and conflict == "none"


def test_judge_is_not_hard_coded_to_claude():
    contestants = [m("claude:haiku", "claude")]
    chosen, conflict = judge.pick_judge(contestants, [m("gemini:flash"), m("xai:grok-4")])
    assert chosen["id"] in ("gemini:flash", "xai:grok-4") and conflict == "none"
    only_openai, _ = judge.pick_judge([m("gemini:flash")], [m("gemini:flash"), m("openai:gpt-5")])
    assert only_openai["id"] == "openai:gpt-5"


def test_conflicts_are_reported_not_hidden():
    a, b = m("gemini:flash"), m("gemini:flash-lite")
    assert judge.pick_judge([a], [a, b]) == (b, "same_provider")
    assert judge.pick_judge([a], [a]) == (a, "same_model")
    assert judge.pick_judge([a], []) == (None, "no_judge_available")


def test_configured_judge_wins_and_still_reports_conflict():
    a, b = m("gemini:flash"), m("openai:gpt-5")
    assert judge.pick_judge([a], [a, b], preferred_id="gemini:flash") == (a, "same_model")


# ---- run_judge failure handling ---------------------------------------------------------------------------------
def run_judge(fake, atts, pool=None, contestants=None):
    async def go():
        return await judge.run_judge("task", None, atts, contestants or CONTESTANTS[:2],
                                     pool or [m("gemini:flash")], seed=1)
    return asyncio.run(go())


def patch_run(monkeypatch, fake):
    calls = []

    async def run(model, spec):
        calls.append(model["id"])
        return fake(model, spec)
    monkeypatch.setattr(providers, "run", run)
    return calls


def test_judge_provider_failure_is_explicit_and_not_retried_elsewhere(monkeypatch):
    calls = patch_run(monkeypatch, lambda mm, s: providers.RunResult("g", "flash", False, error={"kind": "timeout", "message": "slow"}))
    rec = run_judge(None, attempts(["a", "b"]), pool=[m("gemini:flash"), m("openai:gpt-5")])
    assert rec["status"] == "error" and rec["error"]["kind"] == "timeout" and len(calls) == 1


def test_unparseable_judge_reply_is_an_error_not_a_guess(monkeypatch):
    patch_run(monkeypatch, lambda mm, s: providers.RunResult("g", "flash", True, text="I like the first one"))
    rec = run_judge(None, attempts(["a", "b"]))
    assert rec["status"] == "error" and "unparseable" in rec["error"]["message"] and "scores" not in rec


def test_fewer_than_two_submissions_skips_the_judge(monkeypatch):
    calls = patch_run(monkeypatch, lambda mm, s: pytest.fail("judge called"))
    rec = run_judge(None, attempts(["only one"]))
    assert rec["status"] == "skipped" and calls == []


def test_valid_verdict_maps_back_through_the_internal_mapping(monkeypatch):
    patch_run(monkeypatch, lambda mm, s: providers.RunResult("g", "flash", True, text=verdict(2, ranking=[2, 1]), model_version="v9"))
    rec = run_judge(None, attempts(["a", "b"]))
    assert rec["status"] == "ok" and rec["judge_model_version"] == "v9"
    assert sorted(rec["mapping"].values()) == ["att0", "att1"] and rec["ranking"] == [2, 1]


# ---- end to end through the Arena ------------------------------------------------------------------------------
def arena_pool(monkeypatch, contestant_text=lambda mid: f"answer from {mid}", judge_fail=False):
    pool = [m("claude:sonnet", "claude"), m("gpt:gpt-5x", provider="openai"), m("gemini:flashy"), m("xai:grok-4")]
    pool[0].update(tier="main", label="Claude Sonnet")
    pool[1].update(label="GPT-5 (OpenAI API)", model="gpt-5x")
    pool[2].update(label="Gemini Flash", model="flashy")
    pool[3].update(label="Grok 4 (xAI API)", model="grok-4")
    monkeypatch.setattr(models, "enabled_pool", lambda: pool)
    seen = {"judge_prompts": [], "judge_models": [], "judge_systems": [], "contestant_prompts": []}

    async def fake(mm, spec):
        if spec.system == judge.JUDGE_SYSTEM:
            seen["judge_prompts"].append(spec.prompt)
            seen["judge_models"].append(mm["id"])
            if judge_fail:
                return providers.RunResult("j", "j", False, error={"kind": "provider_error", "message": "judge down"})
            return providers.RunResult("j", "j", True, text=verdict(3, ranking=[3, 1, 2]))
        seen["contestant_prompts"].append((mm["id"], spec.prompt))
        return providers.RunResult(providers.provider_of(mm), mm["id"], True, text=contestant_text(mm["id"]), model_version="v-" + mm["id"])
    monkeypatch.setattr(providers, "run", fake)
    return seen


def run_arena(ids, **kw):
    async def go():
        return [e async for e in arena.run("Which niches are rising?", ids, **kw)]
    return asyncio.run(go())


def test_arena_judge_never_sees_model_identity(monkeypatch):
    seen = arena_pool(monkeypatch, contestant_text=lambda mid: f"I am {mid.split(':')[0]}. My answer from {mid}: niche X.")
    events = run_arena(["claude:sonnet", "gpt:gpt-5x", "xai:grok-4"])
    prompt = seen["judge_prompts"][0]
    low = prompt.lower()
    for forbidden in ("claude", "sonnet", "gpt-5", "gpt:", "openai", "xai", "grok", "contestant", "claude:sonnet"):
        assert forbidden not in low, forbidden
    assert "Submission 1:" in prompt and "Submission 3:" in prompt
    assert seen["judge_models"] == ["gemini:flashy"]                    # the only model not competing judged
    final = next(e for e in events if e["type"] == "arena_final")
    judge_ev = next(e for e in events if e["type"] == "judge")
    assert judge_ev["status"] == "ok" and judge_ev["conflict"] == "none"
    assert set(final["suggested"]) == {"claude:sonnet", "gpt:gpt-5x", "xai:grok-4"}


def test_contestants_receive_identical_input_and_never_each_others_answers(monkeypatch):
    seen = arena_pool(monkeypatch)
    run_arena(["claude:sonnet", "gpt:gpt-5x", "xai:grok-4"])
    prompts = {p for _, p in seen["contestant_prompts"]}
    assert len(prompts) == 1 and "answer from" not in next(iter(prompts))


def test_judge_failure_is_visible_in_the_stream_and_the_episode(monkeypatch):
    arena_pool(monkeypatch, judge_fail=True)
    events = run_arena(["claude:sonnet", "gpt:gpt-5x"])
    j = next(e for e in events if e["type"] == "judge")
    final = next(e for e in events if e["type"] == "arena_final")
    assert j["status"] == "error" and j["error"]["kind"] == "provider_error" and final["judge_status"] == "error"
    assert set(final["suggested"]) == {"claude:sonnet", "gpt:gpt-5x"}      # run order, labelled as a judge failure
    ep = bench.get_episode(final["episode_id"])
    assert ep["judge"]["status"] == "error" and ep["status"] == "complete"


def test_judge_scores_and_mapping_are_stored_per_attempt(monkeypatch):
    arena_pool(monkeypatch)
    events = run_arena(["claude:sonnet", "gpt:gpt-5x", "xai:grok-4"])
    ep_id = next(e for e in events if e["type"] == "arena_final")["episode_id"]
    card = bench.scorecard(ep_id)
    for a in card["attempts"]:
        assert set(a["judge_scores"]) == set(judge.CRITERIA)
        assert a["judge_scores"]["groundedness"]["details"]["judge_model"] == "gemini:flashy"
        assert a["judge_scores"]["groundedness"]["details"]["seed"] == card["episode"]["seed"]
    assert sorted(x["anon_label"] for x in [bench.get_attempt(a["attempt_id"]) for a in card["attempts"]]) == [1, 2, 3]


def test_same_model_conflict_is_recorded_when_no_independent_judge_exists(monkeypatch):
    """Only the two contestants are enabled, so one of them must judge. The conflict is recorded, not hidden, and the judge is not Claude by default."""
    pool = [m("gemini:flash"), m("openai:gpt-5x")]
    monkeypatch.setattr(models, "enabled_pool", lambda: pool)
    judged_by = []

    async def fake(mm, spec):
        if spec.system == judge.JUDGE_SYSTEM:
            judged_by.append(mm["id"])
            return providers.RunResult("j", "j", True, text=verdict(2))
        return providers.RunResult(providers.provider_of(mm), mm["id"], True, text="answer " + mm["id"])
    monkeypatch.setattr(providers, "run", fake)
    events = run_arena(["gemini:flash", "openai:gpt-5x"])
    final = next(e for e in events if e["type"] == "arena_final")
    ep = bench.get_episode(final["episode_id"])
    assert ep["judge"]["status"] == "ok" and ep["judge"]["conflict"] in ("same_model", "same_provider")
    assert ep["judge"]["conflict"] == "same_model" and judged_by[0] in ("gemini:flash", "openai:gpt-5x")
    # the internal label -> attempt mapping is kept, covers every submission, and points at real attempts
    mapping = ep["judge"]["mapping"]
    assert sorted(mapping) == ["1", "2"] and {bench.get_attempt(a)["episode_id"] for a in mapping.values()} == {final["episode_id"]}
    assert ep["seed"] is not None and ep["judge"]["seed"] == ep["seed"]


def test_same_seed_reproduces_the_same_submission_order_and_mapping():
    atts = attempts([f"t{i}" for i in range(5)])
    a = judge.anonymize(atts, CONTESTANTS, seed=123)
    b = judge.anonymize(atts, CONTESTANTS, seed=123)
    assert a == b
