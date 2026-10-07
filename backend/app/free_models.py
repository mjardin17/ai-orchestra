"""Free-tier providers with OpenAI-compatible chat endpoints. Each needs a FREE key from the provider's site.
A rate-limited provider sits out for 60s. Free tiers can change and may train on your prompts."""
import asyncio, json, os, socket, time, urllib.error, urllib.request
from . import store

PROVIDERS = {  # name: (base url, key env var)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
}
# Credentials that already exist on this machine under another name. They are READ in place, never copied or renamed,
# and are used only after the primary variable and the app's saved key. Whether one is a valid credential for the
# provider is decided by a health check, not assumed.
KEY_ALIASES = {"gemini": ("GOOGLE_API_KEY",)}
# OpenAI's newer models reject "max_tokens"; everything else here accepts it.
TOKEN_PARAM = {"openai": "max_completion_tokens"}
DEFAULT_MAX_TOKENS = int(os.getenv("FREE_MODEL_MAX_TOKENS", "2000"))
DEFAULT_TIMEOUT = float(os.getenv("FREE_MODEL_TIMEOUT", "60"))
SIGNUP = {  # where to create an account and find the key, and roughly what it takes. You still do this yourself.
    "gemini": ("https://aistudio.google.com/apikey", "Google account. Free, no card."),
    "groq": ("https://console.groq.com/keys", "Email sign-up. Free, no card."),
    "mistral": ("https://console.mistral.ai/api-keys", "Email sign-up. Free tier, may ask for a phone number."),
    "openrouter": ("https://openrouter.ai/settings/keys", "Email or Google sign-up. Free models need no card; paid ones do."),
    "sambanova": ("https://cloud.sambanova.ai/apis", "Email sign-up. Free tier, no card."),
    "deepseek": ("https://platform.deepseek.com/api_keys", "Email sign-up. Bills per token after any trial credit."),
    "openai": ("https://platform.openai.com/api-keys", "OpenAI API. Paid, per token."),
    "xai": ("https://console.x.ai", "xAI API (Grok). Paid, per token."),
    "higgsfield": ("https://console.higgsfield.ai", "Video generation. Paid: register a card and top up a balance (some accounts get starter credit). Paste as KEY_ID:KEY_SECRET."),
}
COOLDOWN: dict = {}


def key_var(p):
    return PROVIDERS[p][1]


def has_key(p):
    return bool(get_key(p))


def get_key(p):
    """An environment variable always wins (so a server deploy can lock it down); otherwise use the key saved in the app's Setup screen,
    then any alias variable (see KEY_ALIASES)."""
    return os.getenv(key_var(p)) or store.get_provider_key(p) or next(
        (os.getenv(v) for v in KEY_ALIASES.get(p, ()) if os.getenv(v)), None)


def key_source(p):
    """Where the key comes from, by NAME only (never the value): an env var name, 'saved in app', or None."""
    if os.getenv(key_var(p)):
        return key_var(p)
    if store.get_provider_key(p):
        return "saved in app"
    return next((v for v in KEY_ALIASES.get(p, ()) if os.getenv(v)), None)


def _post(url, key, body, timeout=None):
    req = urllib.request.Request(url, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "ai-orchestra/1.0"})
    with urllib.request.urlopen(req, timeout=timeout or DEFAULT_TIMEOUT) as r:
        return json.load(r)


def _get(url, key, timeout=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "ai-orchestra/1.0"})
    with urllib.request.urlopen(req, timeout=timeout or DEFAULT_TIMEOUT) as r:
        return json.load(r)


CONTEXT_HINTS = ("context length", "context_length", "maximum context", "too many tokens", "token limit", "prompt is too long",
                 "reduce the length", "exceeds the maximum", "input is too long", "request too large")


def classify_http_error(provider, code, body_text):
    """(kind, message) for an HTTP failure. Kinds: context_exceeded, rate_limited, model_not_found, provider_error."""
    low = (body_text or "").lower()
    snippet = " ".join((body_text or "").split())[:300]
    if code == 413 or any(h in low for h in CONTEXT_HINTS):
        return "context_exceeded", f"{provider}: HTTP {code} {snippet}"
    if code in (429, 503):
        return "rate_limited", f"{provider}: HTTP {code} {snippet}"
    if code == 404 or ("model" in low and ("not found" in low or "does not exist" in low)):
        return "model_not_found", f"{provider}: HTTP {code} {snippet}"
    return "provider_error", f"{provider}: HTTP {code} {snippet}"


def _fail(kind, message):
    from .providers import ProviderError  # imported lazily: providers imports this module
    return ProviderError(kind, message)


def _body_of(e):
    try:
        return e.read().decode("utf-8", "replace")
    except Exception:
        return ""


def _call_full_sync(provider, model, system, prompt, max_tokens, timeout):
    if COOLDOWN.get(provider, 0) > time.time():
        raise _fail("rate_limited", f"{provider} is rate-limited, retrying shortly")
    if not has_key(provider):
        raise _fail("not_configured", f"{key_var(provider)} is not set")
    body = {"model": model, TOKEN_PARAM.get(provider, "max_tokens"): max_tokens or DEFAULT_MAX_TOKENS,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    started = time.time()
    try:
        data = _post(f"{PROVIDERS[provider][0]}/chat/completions", get_key(provider), body, timeout)
    except urllib.error.HTTPError as e:
        kind, msg = classify_http_error(provider, e.code, _body_of(e))
        if kind == "rate_limited":
            COOLDOWN[provider] = time.time() + 60
        raise _fail(kind, msg)
    except (socket.timeout, TimeoutError):
        raise _fail("timeout", f"{provider}: no reply within {timeout or DEFAULT_TIMEOUT:g}s")
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise _fail("timeout", f"{provider}: no reply within {timeout or DEFAULT_TIMEOUT:g}s")
        raise _fail("provider_error", f"{provider}: {e.reason}")
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise _fail("provider_error", f"{provider}: empty reply (finish_reason={choice.get('finish_reason')})")
    return {"text": text, "usage": data.get("usage") or {}, "model_version": data.get("model"),
            "tool_calls": msg.get("tool_calls") or [], "finish_reason": choice.get("finish_reason"),
            "latency": time.time() - started}


async def call_full(provider: str, model: str, system: str, prompt: str, max_tokens=None, timeout=None) -> dict:
    """Like call_one but returns text plus usage, the model version the provider reports, and latency.
    Raises providers.ProviderError (a RuntimeError) with an explicit kind on failure."""
    return await asyncio.to_thread(_call_full_sync, provider, model, system, prompt, max_tokens, timeout)


async def list_models(provider: str, timeout=None) -> list:
    """Model ids the provider says this key can use (used by the health check)."""
    if not has_key(provider):
        raise _fail("not_configured", f"{key_var(provider)} is not set")
    try:
        data = await asyncio.to_thread(_get, f"{PROVIDERS[provider][0]}/models", get_key(provider), timeout)
    except urllib.error.HTTPError as e:
        kind, msg = classify_http_error(provider, e.code, _body_of(e))
        raise _fail(kind, msg)
    except (socket.timeout, TimeoutError):
        raise _fail("timeout", f"{provider}: model list timed out")
    except urllib.error.URLError as e:
        raise _fail("provider_error", f"{provider}: {e.reason}")
    return [str(m.get("id", "")).removeprefix("models/") for m in (data.get("data") or data.get("models") or [])]


async def call_one(provider: str, model: str, system: str, prompt: str) -> str:
    """Original text-only entry point, kept for the council. Same defaults as before (2000 tokens, 60 s) unless the
    FREE_MODEL_MAX_TOKENS / FREE_MODEL_TIMEOUT environment variables say otherwise."""
    return (await call_full(provider, model, system, prompt))["text"]
