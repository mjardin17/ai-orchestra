"""One interface in front of every model client: RunSpec -> run() -> RunResult.

This wraps the clients that already exist (claude_client, codex_client, free_models) instead of replacing them.
Rules this module enforces:
  * A failure is a RunResult with an explicit error kind (context_exceeded, timeout, provider_error, rate_limited,
    model_not_found, not_configured). Nothing is retried on a different model and no evidence is truncated.
  * A model is "supported" only if its last health check succeeded AND its capabilities are explicitly declared.
  * Tool calls are reported only when the provider actually returned them. Nothing is fabricated."""
import asyncio, json, os, re, shutil, time
from dataclasses import dataclass, field
from typing import Optional
from . import claude_client, codex_client, free_models, store

ERROR_KINDS = ("context_exceeded", "timeout", "provider_error", "rate_limited", "model_not_found", "not_configured")

store.DB.executescript("""
create table if not exists provider_health(
  model_id text primary key, provider text, ok int, kind text, message text, latency real,
  model_version text, key_source text, checked_at real);
""")


class ProviderError(RuntimeError):
    """A provider failure with an explicit kind (one of ERROR_KINDS)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


@dataclass
class RunSpec:
    system: str
    prompt: str
    max_tokens: Optional[int] = None      # None = the provider adapter's configured default
    timeout: Optional[float] = None       # seconds; None = the adapter's configured default
    tools: bool = False


@dataclass
class RunResult:
    provider: str
    model: str
    ok: bool
    text: str = ""
    tool_calls: list = field(default_factory=list)   # only what the provider returned; [] means none reported
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    model_version: Optional[str] = None
    error: Optional[dict] = None                      # {"kind": ..., "message": ...} when ok is False
    parameters: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(self.__dict__)


# ---- capability matrix --------------------------------------------------------------------------------------------
# DECLARED values (from this catalog's descriptions and vendor documentation as I know them), not measured. The health
# check proves a round trip works; it does not measure context size. Override per model with the "capabilities" setting:
#   store.set_setting("capabilities", json.dumps({"<model id>": {"max_context": 123456}}))
# tools_native: the provider's API supports tool/function calling.  tools_traceable: THIS app can capture the tool calls
# as a trace today. No tool harness exists yet (P5), so it is False everywhere.
# structured_output: the provider can enforce a JSON schema natively (Claude CLI cannot).
_CAP_KEYS = ("tools_native", "tools_traceable", "max_context", "structured_output")


def _caps(tools_native, max_context, structured_output):
    return {"tools_native": tools_native, "tools_traceable": False, "max_context": max_context,
            "structured_output": structured_output}


PROVIDER_CAPS = {
    "claude": _caps(True, 200_000, False),
    "codex": _caps(True, None, False),                 # context depends on the plan's default model: unknown
    "gemini": _caps(True, 1_000_000, True),
    "groq": _caps(True, 131_072, True),
    "mistral": _caps(True, 128_000, True),
    "deepseek": _caps(True, 1_000_000, True),
    "openai": _caps(True, 400_000, True),
    "xai": _caps(True, 256_000, True),
}
MODEL_CAPS: dict = {}   # per-model exceptions, keyed by catalog id, e.g. {"mistral:mistral-small-latest": {"max_context": 32_000}}


def provider_of(m: dict) -> str:
    if m.get("kind") == "claude":
        return "claude"
    return m.get("provider") or "codex"


def capabilities(m: dict) -> dict:
    """{'tools_native','tools_traceable','max_context','structured_output','known','source'}; known=False if any value is None."""
    caps = dict(PROVIDER_CAPS.get(provider_of(m), {k: None for k in _CAP_KEYS}))
    caps.update(MODEL_CAPS.get(m.get("id"), {}))
    try:
        caps.update(json.loads(store.get_setting("capabilities", "{}") or "{}").get(m.get("id"), {}))
    except ValueError:
        pass
    caps["known"] = all(caps.get(k) is not None for k in _CAP_KEYS)
    caps["source"] = "declared"
    return caps


def estimate_tokens(*texts) -> int:
    return sum(len(t or "") for t in texts) // 4   # rough: ~4 chars/token. Only used as a coarse guard.


# ---- adapters ------------------------------------------------------------------------------------------------------
def _cli_error(text: str) -> ProviderError:
    low = (text or "").lower()
    if any(w in low for w in ("prompt is too long", "context", "too long", "exceed")):
        return ProviderError("context_exceeded", " ".join(text.split())[:300])
    if any(w in low for w in ("rate limit", "usage limit", "429", "overloaded")):
        return ProviderError("rate_limited", " ".join(text.split())[:300])
    if any(w in low for w in ("not logged in", "login", "authenticate", "api key", "is not recognized", "no such file", "cannot find the path")):
        return ProviderError("not_configured", " ".join(text.split())[:300])
    return ProviderError("provider_error", " ".join(text.split())[:300] or "CLI failed with no message")


async def _run_claude(m, spec):
    tier = m["tier"]
    if claude_client.TIERS.index(tier) > claude_client.TIERS.index(claude_client.MAX_TIER):
        # claude_client.run_model would quietly step down to a weaker model; here that is a visible failure.
        raise ProviderError("provider_error", f"{tier} tier is above MAX_TIER={claude_client.MAX_TIER}; not substituting another model")
    model, started = claude_client.MODELS[tier], time.time()
    try:
        text = await claude_client._call(spec.system, spec.prompt, model, spec.tools, spec.timeout)
    except (asyncio.TimeoutError, TimeoutError):
        raise ProviderError("timeout", f"claude CLI gave no reply within {spec.timeout or claude_client.TIMEOUT:g}s")
    except FileNotFoundError:
        raise ProviderError("not_configured", "claude CLI not found")
    except ProviderError:
        raise
    except RuntimeError as e:
        raise _cli_error(str(e))
    if re.match(r"\s*(prompt is too long|error:)", text, re.I):
        raise _cli_error(text)
    if not text.strip():
        raise ProviderError("provider_error", "claude returned nothing")
    return {"text": text, "usage": {}, "model_version": model, "tool_calls": [], "latency": time.time() - started,
            "notes": {"model_version_source": "requested alias (the CLI does not report the resolved version)",
                      "max_tokens_enforced": claude_client.PROVIDER == "api"}}


async def _run_codex(m, spec):
    if not codex_client.available():
        raise ProviderError("not_configured", "Codex CLI not installed")
    started = time.time()
    try:
        text = await codex_client.call(spec.system, spec.prompt, m.get("model"), spec.timeout)
    except (asyncio.TimeoutError, TimeoutError):
        raise ProviderError("timeout", "codex gave no reply in time")
    except RuntimeError as e:
        raise _cli_error(str(e))
    return {"text": text, "usage": {}, "model_version": m.get("model") or "plan default", "tool_calls": [],
            "latency": time.time() - started, "notes": {}}


async def _run_http(m, spec):
    r = await free_models.call_full(m["provider"], m["model"], spec.system, spec.prompt, spec.max_tokens, spec.timeout)
    return {**r, "notes": {"finish_reason": r.get("finish_reason")}}


def _configured(m) -> bool:
    kind = m.get("kind")
    if kind == "claude":
        return bool(os.getenv("CLAUDE_BIN") and os.path.isfile(os.getenv("CLAUDE_BIN"))) or bool(shutil.which("claude"))             or claude_client.PROVIDER == "api"
    if kind == "codex":
        return codex_client.available()
    return free_models.has_key(m.get("provider"))


async def run(m: dict, spec: RunSpec) -> RunResult:
    """Run one model once. Never raises for provider problems; returns a RunResult with ok=False and an error kind."""
    prov = provider_of(m)
    params = {"max_tokens": spec.max_tokens, "timeout": spec.timeout, "tools": spec.tools}
    res = RunResult(provider=prov, model=m.get("model") or m.get("id", "?"), ok=False, parameters=params)
    started = time.time()
    try:
        caps = capabilities(m)
        need = estimate_tokens(spec.system, spec.prompt)
        if caps.get("max_context") and need > caps["max_context"]:
            raise ProviderError("context_exceeded", f"estimated {need} tokens exceeds declared max_context "
                                                    f"{caps['max_context']}; evidence is never truncated, so this attempt is not sent")
        kind = m.get("kind")
        out = await (_run_claude if kind == "claude" else _run_codex if kind == "codex" else _run_http)(m, spec)
        res.ok, res.text = True, out["text"]
        res.usage, res.tool_calls, res.model_version = out.get("usage") or {}, out.get("tool_calls") or [], out.get("model_version")
        params.update(out.get("notes") or {})
    except ProviderError as e:
        res.error = {"kind": e.kind, "message": e.message}
    except asyncio.CancelledError:
        raise
    except Exception as e:  # unexpected: still explicit, never swallowed
        res.error = {"kind": "provider_error", "message": f"{type(e).__name__}: {e}"[:300]}
    res.latency = time.time() - started
    return res


# ---- health --------------------------------------------------------------------------------------------------------
HEALTH_SPEC = RunSpec("You are a connectivity check.", "Reply with the single word OK.", max_tokens=512, timeout=60)


def _save_health(h: dict):
    store.DB.execute("insert or replace into provider_health values(?,?,?,?,?,?,?,?,?)",
                     (h["model_id"], h["provider"], int(h["ok"]), h["kind"], h["message"], h["latency"],
                      h["model_version"], h["key_source"], h["checked_at"]))
    store.DB.commit()


async def health_check(m: dict) -> dict:
    """Does one tiny real round trip (skipped, with no network call, when the provider isn't configured)."""
    prov = provider_of(m)
    key_src = free_models.key_source(prov) if prov in free_models.PROVIDERS else None
    h = {"model_id": m["id"], "provider": prov, "key_source": key_src, "checked_at": time.time(),
         "ok": False, "kind": None, "message": "", "latency": 0.0, "model_version": None}
    if not _configured(m):
        h.update(kind="not_configured", message=_not_configured_message(m))
    else:
        r = await run(m, HEALTH_SPEC)
        h.update(ok=r.ok, latency=r.latency, model_version=r.model_version)
        if not r.ok:
            h.update(kind=r.error["kind"], message=r.error["message"])
        elif not r.text.strip():
            h.update(ok=False, kind="provider_error", message="empty reply")
    _save_health(h)
    return h


def _not_configured_message(m) -> str:
    kind = m.get("kind")
    if kind == "claude":
        return "claude CLI not found on PATH"
    if kind == "codex":
        return "Codex CLI not installed"
    return f"{free_models.key_var(m['provider'])} is not set"


def last_health(model_id: str) -> Optional[dict]:
    r = store.DB.execute("select * from provider_health where model_id=?", (model_id,)).fetchone()
    return dict(r) if r else None


def support_status(m: dict) -> dict:
    """supported = last health check succeeded AND capabilities are explicitly known. 'reasons' lists what's missing."""
    caps, h, reasons = capabilities(m), last_health(m["id"]), []
    if not h:
        reasons.append("no health check has been run")
    elif not h["ok"]:
        reasons.append(f"last health check failed: {h['kind']}")
    if not caps["known"]:
        reasons.append("capabilities not explicitly known")
    return {"supported": not reasons, "reasons": reasons}


def overview(catalog_items: list) -> list:
    out = []
    for m in catalog_items:
        prov = provider_of(m)
        out.append({"id": m["id"], "label": m["label"], "provider": prov, "configured": _configured(m),
                    "key_source": free_models.key_source(prov) if prov in free_models.PROVIDERS else None,
                    "capabilities": capabilities(m), "health": last_health(m["id"]), **support_status(m)})
    return out
