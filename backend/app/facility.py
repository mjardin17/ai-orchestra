"""Facility task loop (Milestone 1): one task -> isolated workspace -> agent -> verification by command -> receipt.

Rules this module enforces:
- The agent works in a temp COPY of a fixture; the source fixture is never touched.
- read_only tasks must leave the workspace byte-identical, else status permission_violation (and nothing is trusted).
- A result is "verified" only when the task's own acceptance command (python -m pytest ...) exits 0. The model's say-so counts for nothing.
- Failures become CANDIDATE lessons (pending, unverified, never injected). Only approved AND verified lessons reach a prompt.
No shell is used anywhere; the verify command is restricted to `python -m pytest <relative paths>`."""
import asyncio, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from . import claude_client, lessons

FIXTURE_ROOT = Path(os.getenv("FACILITY_FIXTURE_ROOT") or Path(__file__).resolve().parents[1] / "fixtures" / "facility")
PERMISSIONS = ("read_only", "write")
MIN_TIMEOUT, MAX_TIMEOUT = 1, 900
OUTPUT_TAIL = 4000
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_ARG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*(::[A-Za-z0-9_]+)?$")
_IGNORED = {"__pycache__", ".pytest_cache"}
SYSTEM = ("You are a careful engineer working inside one project directory. Change only files in the current directory. "
          "Do the task, keep changes minimal, and do not run the tests yourself: an independent verifier will.")


class TaskError(ValueError):
    """The task record is malformed or asks for something not allowed. Nothing runs."""


def validate_task(t: dict) -> dict:
    if not isinstance(t, dict):
        raise TaskError("task must be an object")
    for f in ("id", "fixture"):
        if not isinstance(t.get(f), str) or not _NAME.match(t[f]) or t[f] in (".", ".."):
            raise TaskError(f"{f} must be 1-64 chars [A-Za-z0-9_.-]")
    for f in ("title", "prompt"):
        if not isinstance(t.get(f), str) or not t[f].strip():
            raise TaskError(f"{f} is required")
    if not (FIXTURE_ROOT / t["fixture"]).is_dir():
        raise TaskError(f"unknown fixture: {t['fixture']}")
    if t.get("permissions") not in PERMISSIONS:
        raise TaskError(f"permissions must be one of {PERMISSIONS}")
    v = t.get("verify")
    if not (isinstance(v, list) and v[:3] == ["python", "-m", "pytest"] and len(v) > 3):
        raise TaskError("verify must be ['python','-m','pytest', <relative test paths>]")
    for a in v[3:]:
        if a not in ("-q", "-x") and (not isinstance(a, str) or not _ARG.match(a) or ".." in a.split("/")):
            raise TaskError(f"verify argument not allowed: {a!r}")
    timeout = t.get("timeout_s", 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not MIN_TIMEOUT <= timeout <= MAX_TIMEOUT:
        raise TaskError(f"timeout_s must be {MIN_TIMEOUT}-{MAX_TIMEOUT}")
    return t


def snapshot(root: Path) -> dict:
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (set(rel.parts) & _IGNORED):
            out[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def diff(before: dict, after: dict) -> dict:
    return {"added": sorted(set(after) - set(before)), "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in before if k in after and before[k] != after[k])}


def select_model(t: dict) -> str:
    """Claude only for Milestone 1 (subscription CLI, no API spend). Cheap tier for read-only work."""
    return t.get("model") if t.get("model") in ("claude:haiku", "claude:sonnet", "claude:opus") else (
        "claude:haiku" if t["permissions"] == "read_only" else "claude:sonnet")


async def claude_runner(ws: Path, prompt: str, write: bool, model: str, timeout: float) -> dict:
    """Run the Claude CLI inside the workspace. Tools are restricted by flag: read-only gets no Edit/Write/Bash."""
    tools = "Read,Glob,Grep" + (",Edit,Write" if write else "")
    tier = {"haiku": "fast", "sonnet": "main", "opus": "top"}[model.split(":")[1]]
    args = [claude_client.cli_path(), "-p", "--model", claude_client.MODELS[claude_client.resolve(tier)],
            "--allowedTools", tools] + (["--permission-mode", "acceptEdits"] if write else [])
    proc = await asyncio.create_subprocess_exec(*args, cwd=str(ws), stdin=asyncio.subprocess.PIPE,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
    except BaseException:  # timeout or cancel: never leave the process running
        if proc.returncode is None:
            proc.kill()
        raise
    if proc.returncode:
        raise RuntimeError(err.decode()[:300] or "claude CLI failed")
    return {"text": out.decode().strip(), "exit_code": proc.returncode}


DEFAULT_RUNNER = claude_runner


def _verify(ws: Path, cmd: list, timeout: float) -> dict:
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + [a for a in cmd[3:] if a != "-q"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(argv, cwd=ws, capture_output=True, text=True, timeout=timeout, env=env)
        return {"command": cmd, "exit_code": p.returncode, "output_tail": (p.stdout + p.stderr)[-OUTPUT_TAIL:]}
    except subprocess.TimeoutExpired:
        return {"command": cmd, "exit_code": -1, "output_tail": f"verifier timed out after {timeout}s"}


def _prompt(t: dict, injected: list) -> str:
    rules = "".join(f"\n- {r['rule']}" for r in injected)
    head = SYSTEM + ("\n\nVerified rules (follow them):" + rules if rules else "")
    mode = "You may edit files." if t["permissions"] == "write" else "READ ONLY: do not create or change any file."
    return f"{head}\n\n{mode}\n\nTask: {t['title']}\n{t['prompt']}"


def _write_receipt(receipt: dict) -> str:
    d = Path(os.getenv("FACILITY_RECEIPT_DIR") or "receipts")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{receipt['task_id']}-{int(receipt['started'])}.json"
    p.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return str(p)


async def run_task(task: dict, runner=None) -> dict:
    """Run one task end to end and return the evidence receipt (also written to FACILITY_RECEIPT_DIR). Never raises for task outcomes."""
    t = validate_task(dict(task))
    runner = runner or DEFAULT_RUNNER
    model, write, timeout = select_model(t), t["permissions"] == "write", float(t.get("timeout_s", 300))
    ws = Path(tempfile.mkdtemp(prefix=f"facility-{t['id']}-"))
    shutil.copytree(FIXTURE_ROOT / t["fixture"], ws, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    injected = lessons.injectable_for("facility")
    receipt = {"task_id": t["id"], "title": t["title"], "model": model, "permissions": t["permissions"],
               "workspace": str(ws), "started": time.time(), "lessons_injected": [r["id"] for r in injected],
               "status": None, "error": None, "agent": None, "diff": None, "verify": None, "candidate_lessons": []}
    before = snapshot(ws)
    try:
        receipt["agent"] = await asyncio.wait_for(runner(ws, _prompt(t, injected), write, model, timeout), timeout)
        receipt["agent"] = {"exit_code": (receipt["agent"] or {}).get("exit_code"),
                            "text_tail": str((receipt["agent"] or {}).get("text", ""))[-OUTPUT_TAIL:]}
    except asyncio.TimeoutError:
        receipt["status"], receipt["error"] = "timeout", f"agent exceeded {timeout}s"
    except Exception as e:
        receipt["status"], receipt["error"] = "agent_error", str(e)[:300]
    receipt["diff"] = diff(before, snapshot(ws))      # taken BEFORE verification so the verifier cannot pollute it
    changed = any(receipt["diff"].values())
    if receipt["status"] is None and not write and changed:
        receipt["status"], receipt["error"] = "permission_violation", "read_only task changed the workspace"
    if receipt["status"] is None:
        receipt["verify"] = _verify(ws, t["verify"], timeout)
        receipt["status"] = "verified" if receipt["verify"]["exit_code"] == 0 else "verification_failed"
        if receipt["status"] == "verification_failed":
            receipt["error"] = "acceptance command exited %s" % receipt["verify"]["exit_code"]
    if receipt["status"] != "verified":
        lid = lessons.record_failure("facility", model, f"{receipt['status']}: {receipt['error']}")
        receipt["candidate_lessons"].append({"id": lid, "status": "pending", "verified": 0})
    receipt["finished"] = time.time()
    receipt["receipt_path"] = _write_receipt(receipt)
    return receipt
