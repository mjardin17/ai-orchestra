"""Model catalog with a capability rating (1-5) and a cost rank (0 free, 1 cheap, 2 mid, 3 high) per model.
The ratings are rough defaults, not benchmarks: edit capability in the dashboard to match your own experience."""
import json
from . import codex_client, free_models, store
from .claude_client import run_model

CLAUDE = [  # id, label, tier, capability, cost, description
    ("claude:haiku", "Claude Haiku", "fast", 3, 1, "Fast and cheap. Simple to medium tasks."),
    ("claude:sonnet", "Claude Sonnet", "main", 4, 2, "Balanced. Strong at writing, code and analysis."),
    ("claude:opus", "Claude Opus", "top", 5, 3, "Strongest Claude. Hard reasoning and tricky code. Uses the most plan limit."),
]
CODEX = [  # model id (None = your plan's default), label, capability, cost, description
    (None, "ChatGPT (your login)", 4, 2, "OpenAI model through your ChatGPT plan's Codex access. Slower than a plain chat call."),
]
FREE = [  # provider, model id, label, capability, description. Model ids change: edit or add your own in the dashboard.
    ("gemini", "gemini-flash-latest", "Gemini Flash", 3, "Google. Fast, very long context, good all-rounder."),
    ("gemini", "gemini-flash-lite-latest", "Gemini Flash-Lite", 2, "Google. Lightest and quickest. Simple tasks."),
    ("groq", "openai/gpt-oss-120b", "GPT-OSS 120B (Groq)", 3, "Open-weight, strong reasoning, very fast on Groq."),
    ("groq", "openai/gpt-oss-20b", "GPT-OSS 20B (Groq)", 2, "Small open-weight model. Quick, simple tasks."),
    ("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", 3, "Meta open model. Solid general writing and chat."),
    ("mistral", "mistral-small-latest", "Mistral Small", 2, "Mistral. Cheap, quick, decent writing."),
    ("mistral", "mistral-large-latest", "Mistral Large", 3, "Mistral. Stronger general model."),
]
LOW_COST = [  # provider, model id, label, capability, cost, description. Pay-per-token after any trial credits.
    ("deepseek", "deepseek-v4-flash", "DeepSeek V4 Flash", 3, 1, "DeepSeek. Very cheap per token, 1M-token context. Good for bulk and coding-flavoured work."),
    ("deepseek", "deepseek-v4-pro", "DeepSeek V4 Pro", 4, 2, "DeepSeek. Higher-capability model for harder reasoning and coding."),
]
CODEX_HINT = "Needs the Codex CLI: run `npm i -g @openai/codex`, then `codex login`, then restart the app."


def catalog():
    items = [{"id": i, "label": l, "kind": "claude", "tier": t, "power": p, "cost": c, "blurb": b, "available": True}
             for i, l, t, p, c, b in CLAUDE]
    rows = [("codex", m, l, p, c, b, False) for m, l, p, c, b in CODEX]
    rows += [(pr, m, l, p, 0, b, False) for pr, m, l, p, b in FREE]
    rows += [(pr, m, l, p, c, b, False) for pr, m, l, p, c, b in LOW_COST]
    rows += [(c["provider"], c["model"], c["label"], 4 if c["provider"] == "codex" else 3,
              2 if c["provider"] == "codex" else 0, "Custom model you added.", True) for c in store.list_custom()]
    for pr, m, l, p, c, b, custom in rows:
        if pr == "codex":
            items.append({"id": f"codex:{m or 'default'}", "label": l, "kind": "codex", "provider": pr, "model": m, "power": p,
                          "cost": c, "blurb": b, "custom": custom, "available": codex_client.available(), "hint": CODEX_HINT})
        else:
            items.append({"id": f"{pr}:{m}", "label": l, "kind": "free", "provider": pr, "model": m, "power": p, "cost": c,
                          "blurb": b, "custom": custom, "available": free_models.has_key(pr),
                          "hint": f"Needs {free_models.key_var(pr)} (a key from the provider's site), then restart the app."})
    flags, power = store.model_flags(), json.loads(store.get_setting("power", "{}") or "{}")
    for it in items:  # Claude on by default; everything else is off until you tick it
        it["power"] = int(power.get(it["id"], it["power"]))
        it["enabled"] = it["available"] and flags.get(it["id"], it["kind"] == "claude")
    return items


def find(mid):
    return next((m for m in catalog() if m["id"] == mid), None)


def enabled_pool():
    return [m for m in catalog() if m["enabled"]]


async def run_strict(m, system, prompt, tools=False):
    """One attempt on one model. Raises on failure so the council can move to the next model."""
    if m["kind"] == "codex":
        return await codex_client.call(system, prompt, m["model"])
    if m["kind"] == "free":  # free providers and Codex have no web search wired in
        return await free_models.call_one(m["provider"], m["model"], system, prompt)
    return (await run_model(system, prompt, m["tier"], tools))[0]
