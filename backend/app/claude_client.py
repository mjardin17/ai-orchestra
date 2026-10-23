"""One place every agent calls. Handles provider choice AND model routing.

ORCHESTRA_PROVIDER: cli = your own `claude login` subscription (personal use), api = ANTHROPIC_API_KEY.
Tiers: fast < main < top. The router picks a tier per task; this file enforces a ceiling
(MAX_TIER) and steps down automatically if a model isn't available on your plan/account.
"""
import asyncio, os, re, shutil

PROVIDER = os.getenv("ORCHESTRA_PROVIDER", "cli")
TIERS = ["fast", "main", "top"]
_D = ({"fast": "claude-haiku-4-5-20251001", "main": "claude-sonnet-5", "top": "claude-opus-5"}
      if PROVIDER == "api" else {"fast": "haiku", "main": "sonnet", "top": "opus"})
MODELS = {t: os.getenv(f"MODEL_{t.upper()}", _D[t]) for t in TIERS}
TIMEOUT = float(os.getenv("CLAUDE_TIMEOUT", "240"))              # seconds per CLI call
MAX_OUTPUT_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "2000"))  # API mode only; the CLI sets its own limit
MAX_TIER = os.getenv("MAX_TIER", "top")          # lower this to protect subscription limits
if MAX_TIER not in TIERS:
    MAX_TIER = "top"
UNAVAILABLE: set = set()                          # tiers that failed with a model-access error
MODEL_ERR = re.compile(r"model|not available|access|permission|not found|does not exist|not supported", re.I)


def cli_path() -> str:
    """The claude executable to run: CLAUDE_BIN if set (use it when the `claude` on PATH is a stale shim), else PATH."""
    return os.getenv("CLAUDE_BIN") or shutil.which("claude") or "claude"


def resolve(tier: str) -> str:
    """Clamp to MAX_TIER, then step down past tiers known to be unavailable."""
    i = min(TIERS.index(tier if tier in TIERS else "main"), TIERS.index(MAX_TIER))
    while i > 0 and TIERS[i] in UNAVAILABLE:
        i -= 1
    return TIERS[i]


async def _call(system: str, prompt: str, model: str, tools: bool = False, timeout=None) -> str:
    if PROVIDER == "api":
        from anthropic import AsyncAnthropic
        extra = {"tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]} if tools else {}
        try:
            msg = await AsyncAnthropic().messages.create(
                model=model, max_tokens=MAX_OUTPUT_TOKENS, system=system,
                messages=[{"role": "user", "content": prompt}], **extra)
        except Exception as e:
            raise RuntimeError(str(e)[:300])
        return "".join(b.text for b in msg.content if b.type == "text")

    full = f"{system}\n\n---\nUser request:\n{prompt}"
    args = [cli_path(), "-p", "--model", model] + (["--allowedTools", "WebSearch,WebFetch"] if tools else [])
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(full.encode()), timeout=timeout or TIMEOUT)
    except BaseException:  # timeout or Stop: don't leave the process running
        if proc.returncode is None:
            proc.kill()
        raise
    if proc.returncode:
        raise RuntimeError(err.decode()[:300] or "claude CLI failed. Run `claude login` first.")
    return out.decode().strip()


async def run_model(system: str, prompt: str, tier: str = "main", tools: bool = False):
    """Returns (text, tier_actually_used). Falls back down the ladder on model-access errors."""
    t = resolve(tier)
    while True:
        try:
            return await _call(system, prompt, MODELS[t], tools), t
        except RuntimeError as e:
            if tools:  # web search may be unsupported here: retry once without it before blaming the model
                tools = False
                continue
            if t != "fast" and MODEL_ERR.search(str(e)):
                UNAVAILABLE.add(t)
                t = resolve(t)
                continue
            raise


async def run_agent(system: str, prompt: str, tier: str = "main") -> str:
    return (await run_model(system, prompt, tier))[0]
