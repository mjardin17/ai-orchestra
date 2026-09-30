"""Free-tier providers with OpenAI-compatible chat endpoints. Each needs a FREE key from the provider's site.
A rate-limited provider sits out for 60s. Free tiers can change and may train on your prompts."""
import asyncio, json, os, time, urllib.error, urllib.request
from . import store

PROVIDERS = {  # name: (base url, key env var)
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "sambanova": ("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}
SIGNUP = {  # where to create an account and find the key, and roughly what it takes. You still do this yourself.
    "gemini": ("https://aistudio.google.com/apikey", "Google account. Free, no card."),
    "groq": ("https://console.groq.com/keys", "Email sign-up. Free, no card."),
    "mistral": ("https://console.mistral.ai/api-keys", "Email sign-up. Free tier, may ask for a phone number."),
    "openrouter": ("https://openrouter.ai/settings/keys", "Email or Google sign-up. Free models need no card; paid ones do."),
    "sambanova": ("https://cloud.sambanova.ai/apis", "Email sign-up. Free tier, no card."),
    "deepseek": ("https://platform.deepseek.com/api_keys", "Email sign-up. Bills per token after any trial credit."),
    "higgsfield": ("https://console.higgsfield.ai", "Video generation. Paid: register a card and top up a balance (some accounts get starter credit). Paste as KEY_ID:KEY_SECRET."),
}
COOLDOWN: dict = {}


def key_var(p):
    return PROVIDERS[p][1]


def has_key(p):
    return bool(get_key(p))


def get_key(p):
    """An environment variable always wins (so a server deploy can lock it down); otherwise use the key saved in the app's Setup screen."""
    return os.getenv(key_var(p)) or store.get_provider_key(p)


def _post(url, key, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "ai-orchestra/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


async def call_one(provider: str, model: str, system: str, prompt: str) -> str:
    if COOLDOWN.get(provider, 0) > time.time():
        raise RuntimeError(f"{provider} is rate-limited, retrying shortly")
    if not has_key(provider):
        raise RuntimeError(f"{key_var(provider)} is not set")
    body = {"model": model, "max_tokens": 2000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    try:
        data = await asyncio.to_thread(_post, f"{PROVIDERS[provider][0]}/chat/completions", get_key(provider), body)
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            COOLDOWN[provider] = time.time() + 60
        raise RuntimeError(f"{provider}: HTTP {e.code}")
    text = (data["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise RuntimeError(f"{provider}: empty reply")
    return text
