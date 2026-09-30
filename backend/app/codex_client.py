"""ChatGPT through the official Codex CLI, signed in with your ChatGPT account (no API key).
Setup: npm i -g @openai/codex, then `codex login`. Usage counts against your ChatGPT plan's Codex allowance."""
import asyncio, shutil


def available() -> bool:
    return shutil.which("codex") is not None


async def call(system: str, prompt: str, model=None) -> str:
    exe = shutil.which("codex")
    if not exe:
        raise RuntimeError("Codex CLI not installed")
    args = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only"] + (["-m", model] if model else []) + ["-"]
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(f"{system}\n\n---\nUser request:\n{prompt}".encode()), timeout=300)
    except BaseException:  # timeout or Stop: don't leave the process running
        if proc.returncode is None:
            proc.kill()
        raise
    if proc.returncode:
        raise RuntimeError(err.decode()[-300:] or "codex failed. Run `codex login` first.")
    text = out.decode().strip()
    if not text:
        raise RuntimeError("codex returned nothing")
    return text
