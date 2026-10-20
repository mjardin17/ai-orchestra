"""Facility task loop (Milestone 1): one task -> isolated workspace -> agent -> verification by command -> receipt.

Enforced by code (see facility_sandbox.py / facility_auth.py; each rule has tests):
- The agent works in a temp COPY of a fixture; the source fixture is never touched; every directory we create is deleted afterwards.
- Trusted tests (test files and test config) live in a separate directory OUTSIDE the agent workspace. That directory is locked at the
  OS level (Windows ACL deny / POSIX chmod) before the agent starts and stays locked until verification has finished.
- read_only: the whole workspace is OS-locked while the agent runs. write: every existing file NOT on the task's explicit allowed_paths
  list is OS-locked, and changes outside allowed_paths are a violation even if they could not be blocked (e.g. a newly created file).
- Verification builds its own directory: pristine fixture files, plus the agent's output for allowed_paths ONLY, plus the trusted tests.
  It runs with an allow-listed environment (no keys/tokens) and is judged by pytest's junit report, not by exit code or printed text.
- "verified" requires: no violation AND exit code 0 AND a well-formed junit report with >=1 test and 0 failures/errors/skips.
- The real provider runner is disabled unless FACILITY_ENABLE_PROVIDER_EXEC=1 (and FACILITY_ENABLE_WRITE_EXEC=1 for write tasks).
- Failures become CANDIDATE lessons (pending, unverified, never injected). Only approved AND verified lessons reach a prompt.
NOT enforced (host limitation, see FACILITY.md): network access, subprocess creation and reads/writes outside the workspace by code that
runs as your Windows user (the agent CLI, or agent-written code under the verifier). No cancellation: a run ends by finishing or by deadline."""
import asyncio, json, os, re, secrets, shutil, sys, tempfile, time, uuid
from collections import deque
from pathlib import Path
from . import claude_client, facility_attestation, facility_sandbox as sb, lessons

FIXTURE_ROOT = Path(os.getenv("FACILITY_FIXTURE_ROOT") or Path(__file__).resolve().parents[1] / "fixtures" / "facility")
PERMISSIONS = ("read_only", "write")
MIN_TIMEOUT, MAX_TIMEOUT, MAX_VERIFY_TIMEOUT, DEFAULT_VERIFY_TIMEOUT = 2, 900, 300, 60
OUTPUT_TAIL = 4000
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}\Z")
_ARG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*(::[A-Za-z0-9_]+)?\Z")
_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,120}\Z")
SYSTEM = ("You are a careful engineer working inside one project directory. Change only files in the current directory. "
          "Do the task, keep changes minimal, and do not run the tests yourself: an independent verifier will.")
AGENT_TOOLS_RO, AGENT_TOOLS_RW = "Read,Glob,Grep", "Read,Glob,Grep,Edit,Write"
DENIED_TOOLS = "Bash,PowerShell,WebFetch,WebSearch,Task"
CLEANUP_GRACE_SEC = 10
KNOWN_KEYS = {"id", "title", "prompt", "fixture", "permissions", "allowed_paths", "verify", "timeout_s", "verify_timeout_s", "model"}


class TaskError(ValueError):
    """The task record is malformed or asks for something not allowed. Nothing runs."""


class ExecutionDisabled(RuntimeError):
    """Real provider execution is switched off, or the boundary could not be enforced. Fail closed."""


class Busy(RuntimeError):
    """Concurrency or hourly limit reached."""


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
    for f, lo, hi, default in (("timeout_s", MIN_TIMEOUT, MAX_TIMEOUT, 300), ("verify_timeout_s", 1, MAX_VERIFY_TIMEOUT, DEFAULT_VERIFY_TIMEOUT)):
        x = t.get(f, default)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not lo <= x <= hi:
            raise TaskError(f"{f} must be {lo}-{hi}")
    allowed = t.get("allowed_paths", [])
    if not isinstance(allowed, list) or len(allowed) > 20:
        raise TaskError("allowed_paths must be a list of at most 20 relative file paths")
    for a in allowed:
        if not isinstance(a, str) or not _PATH.match(a) or ".." in a.split("/") or "//" in a or a.endswith("/") or sb.is_protected(a):
            raise TaskError(f"allowed_paths entry not allowed: {a!r}")
    if t["permissions"] == "write" and not allowed:
        raise TaskError("write tasks must list allowed_paths (the only files the agent may change)")
    if t["permissions"] == "read_only" and allowed:
        raise TaskError("read_only tasks cannot have allowed_paths")
    return t


def _reject_unknown_keys(t: dict) -> None:
    extra = sorted(set(t) - KNOWN_KEYS)
    if extra:
        raise TaskError(f"unknown task fields are not allowed: {extra}")


def _norm_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_fixture_manifest(name: str) -> None:
    """If FIXTURE_ROOT/MANIFEST.json exists (it does for the shipped fixtures) the fixture must match it exactly, so a previous run or
    another process cannot have poisoned the 'pristine' copy. Line endings are normalised (git autocrlf)."""
    mf = FIXTURE_ROOT / "MANIFEST.json"
    if not mf.exists():
        return
    want = json.loads(mf.read_text(encoding="utf-8")).get(name)
    fix = FIXTURE_ROOT / name
    have = {k: _norm_hash(fix / k) for k in sb.scan(fix)["files"]}
    if want != have:
        raise ExecutionDisabled(f"fixture {name!r} does not match MANIFEST.json (modified, added or removed files); refusing to run")


def select_model(t: dict) -> str:
    """Claude only for Milestone 1 (subscription CLI, no API spend). Cheap tier for read-only work."""
    return t.get("model") if t.get("model") in ("claude:haiku", "claude:sonnet", "claude:opus") else (
        "claude:haiku" if t["permissions"] == "read_only" else "claude:sonnet")


def resolve_model(model: str) -> dict:
    """What will actually be requested from which provider (tier clamping via MAX_TIER and unavailable tiers is applied)."""
    tier = {"haiku": "fast", "sonnet": "main", "opus": "top"}[model.split(":")[1]]
    return {"provider": "claude-cli", "requested": model, "resolved_tier": claude_client.resolve(tier),
            "cli_model": claude_client.MODELS[claude_client.resolve(tier)]}


# ---- admission control: concurrency and hourly cap protect the subscription -------------------------------------
_active, _starts = 0, deque()


def admit() -> None:
    global _active
    now, per_hour = time.time(), int(os.getenv("FACILITY_MAX_PER_HOUR", "20"))
    while _starts and _starts[0] < now - 3600:
        _starts.popleft()
    if _active >= int(os.getenv("FACILITY_MAX_CONCURRENT", "1")):
        raise Busy("another facility task is running")
    if len(_starts) >= per_hour:
        raise Busy(f"hourly limit of {per_hour} facility tasks reached")
    _active += 1
    _starts.append(now)


def release() -> None:
    global _active
    _active = max(0, _active - 1)


# ---- the real provider runner -------------------------------------------------------------------------------------
def ensure_exec_enabled(write: bool) -> None:
    if os.getenv("FACILITY_ENABLE_PROVIDER_EXEC") != "1":
        raise ExecutionDisabled("real provider execution is disabled: set FACILITY_ENABLE_PROVIDER_EXEC=1 to allow it")
    if write and os.getenv("FACILITY_ENABLE_WRITE_EXEC") != "1":
        raise ExecutionDisabled("write execution is disabled: set FACILITY_ENABLE_WRITE_EXEC=1 to allow it")


def build_agent_argv(write: bool, model: str) -> list:
    """Claude CLI arguments. The tool list removes Bash/PowerShell/web tools from the agent; user/local settings and MCP servers are
    not loaded. These flags are defence in depth only: the hard boundary is the OS lock on the workspace."""
    tools = AGENT_TOOLS_RW if write else AGENT_TOOLS_RO
    return [claude_client.cli_path(), "-p", "--model", resolve_model(model)["cli_model"],
            "--tools", tools, "--allowedTools", tools, "--disallowedTools", DENIED_TOOLS,
            "--restricted", "--setting-sources", "project", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
            "--permission-prompts", "none", "--permission-mode", "acceptEdits" if write else "dontAsk"]


async def claude_runner(ws: Path, prompt: str, write: bool, model: str, timeout: float) -> dict:
    ensure_exec_enabled(write)
    proc = await asyncio.create_subprocess_exec(*build_agent_argv(write, model), cwd=str(ws), env=sb.agent_env(),
                                                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, **sb.SPAWN_KW)
    try:
        out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
    except BaseException:  # timeout or cancel: kill the whole process tree
        sb.kill_tree(proc)
        raise
    if proc.returncode:
        raise RuntimeError(err.decode(errors="replace")[:300] or "claude CLI failed")
    return {"text": out.decode(errors="replace").strip(), "exit_code": proc.returncode}


DEFAULT_RUNNER = claude_runner


# ---- verification -------------------------------------------------------------------------------------------------
async def _verify(vdir: Path, report: Path, cmd: list, timeout: float) -> dict:
    report.unlink(missing_ok=True)
    started_at = time.time()
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", "pytest.ini", "--rootdir", ".",
            f"--junitxml={report}"] + [a for a in cmd[3:] if a != "-q"]
    proc = await asyncio.create_subprocess_exec(*argv, cwd=str(vdir), env=sb.verify_env(vdir), stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **sb.SPAWN_KW)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        sb.kill_tree(proc)
        return {"command": cmd, "exit_code": -1, "report": None, "report_ok": False, "output_tail": f"verifier timed out after {timeout}s"}
    except BaseException:
        sb.kill_tree(proc)
        raise
    fresh = report.exists() and report.stat().st_mtime >= started_at - 1     # a pre-existing/stale file is never trusted
    counts = sb.parse_junit(report) if fresh else None
    ok = bool(counts) and proc.returncode == 0 and counts["tests"] >= 1 and not (counts["failures"] or counts["errors"] or counts["skipped"])
    return {"command": cmd, "exit_code": proc.returncode, "report": counts, "report_ok": ok,
            "output_tail": sb.redact(out.decode(errors="replace"))[-OUTPUT_TAIL:]}


def _prompt(t: dict, injected: list) -> str:
    rules = "".join(f"\n- {r['rule']}" for r in injected)
    head = SYSTEM + ("\n\nVerified rules (follow them):" + rules if rules else "")
    mode = ("You may edit ONLY these files: " + ", ".join(t["allowed_paths"]) + ".") if t["permissions"] == "write" \
        else "READ ONLY: do not create or change any file."
    return f"{head}\n\n{mode}\n\nTask: {t['title']}\n{t['prompt']}"


def _write_receipt(receipt: dict) -> str:
    d = Path(os.getenv("FACILITY_RECEIPT_DIR") or Path(__file__).resolve().parents[1] / "receipts")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{receipt['task_id']}-{receipt['run_id']}.json"
    with open(p, "x", encoding="utf-8") as f:       # 'x': never overwrite another run's receipt
        json.dump(receipt, f, indent=2)
    return str(p)


def _evaluate(receipt, write, allowed, before, after_scan, pristine) -> None:
    """Pick the first applicable failure status. Anything but a clean pass can never become 'verified'."""
    receipt["diff"] = d = sb.diff(before, after_scan["files"])
    touched = d["added"] + d["removed"] + d["changed"]
    receipt["protected_violations"] = sb.protected_violations(pristine, after_scan["files"])
    outside = sorted(p for p in touched if p not in allowed)
    if after_scan["problems"]:
        receipt["status"], receipt["error"] = "permission_violation", "; ".join(after_scan["problems"])[:300]
    elif receipt["protected_violations"]:
        receipt["status"] = "protected_file_violation"
        receipt["error"] = "protected test files were modified or added: " + ", ".join(receipt["protected_violations"])[:250]
    elif outside:
        why = "read_only task changed the workspace" if not write else "changes outside allowed_paths"
        receipt["status"], receipt["error"] = "permission_violation", f"{why}: " + ", ".join(outside)[:250]


async def run_task(task: dict, runner=None) -> dict:
    """Run one task end to end and return the evidence receipt (also written to FACILITY_RECEIPT_DIR). The whole run (agent +
    verification) shares ONE deadline of timeout_s; cleanup follows. Raises TaskError (bad task) or ExecutionDisabled (provider off,
    or the OS lock could not be applied); every task OUTCOME is reported in the receipt instead."""
    t = validate_task(dict(task))
    _reject_unknown_keys(t)
    runner = runner or DEFAULT_RUNNER
    model, write = select_model(t), t["permissions"] == "write"
    allowed = set(t.get("allowed_paths", []))
    started = time.time()
    deadline = started + float(t.get("timeout_s", 300))
    vbudget = min(float(t.get("verify_timeout_s", DEFAULT_VERIFY_TIMEOUT)), float(t.get("timeout_s", 300)) / 2)
    if runner is claude_runner:
        ensure_exec_enabled(write)
    fixture = FIXTURE_ROOT / t["fixture"]
    pristine_scan = sb.scan(fixture)
    if pristine_scan["problems"]:
        raise TaskError("fixture is not a clean regular-file tree: " + "; ".join(pristine_scan["problems"])[:200])
    pristine = pristine_scan["files"]
    check_fixture_manifest(t["fixture"])
    trusted_bytes = sb.read_protected_bytes(fixture)      # pinned in memory before anything else can touch the disk copy

    sb.sweep_stale()
    injected = lessons.injectable_for("facility")
    run_id = uuid.uuid4().hex[:12]
    ws = Path(tempfile.mkdtemp(prefix=f"{t['id']}-", dir=sb.workspace_root()))
    trusted, vdir = (ws.with_name(ws.name + s) for s in ("-trusted", "-verify"))
    # Unguessable report location outside the workspace and verify dirs; it cannot be derived from a directory name.
    report = sb.workspace_root() / ".reports" / f"{secrets.token_hex(16)}.xml"
    report.parent.mkdir(exist_ok=True)
    receipt = {"run_id": run_id, "task_id": t["id"], "title": t["title"], "model": model, "resolved_model": resolve_model(model),
               "permissions": t["permissions"], "allowed_paths": sorted(allowed), "started": started,
               "lessons_injected": [r["id"] for r in injected], "status": None, "error": None, "agent": None, "diff": None,
               "protected_violations": [], "verify": None, "candidate_lessons": [], "files_after": None,
               "workspace_deleted": False, "cleanup_error": None,
               "verification_trust": "agent_code_executed_same_user" if write else "pristine_code_only"}
    try:
        shutil.copytree(fixture, ws, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*sb.IGNORED))
        before = sb.scan(ws)["files"]
        sb.make_trusted_dir(fixture, trusted)
        try:     # cannot enforce the boundary -> do not run the agent at all
            sb.lock_tree(trusted)
            if write:
                sb.lock_files([ws / k for k in before if k not in allowed])
            else:
                sb.lock_tree(ws)
        except OSError as e:
            raise ExecutionDisabled(f"could not lock the workspace, refusing to run: {e}")
        agent_budget = max(1.0, deadline - time.time() - vbudget)
        try:
            res = await asyncio.wait_for(runner(ws, _prompt(t, injected), write, model, agent_budget), agent_budget)
            receipt["agent"] = {"exit_code": (res or {}).get("exit_code"),
                                "text_tail": sb.redact(str((res or {}).get("text", "")))[-OUTPUT_TAIL:]}
        except asyncio.TimeoutError:
            receipt["status"], receipt["error"] = "timeout", f"agent exceeded its {agent_budget:.0f}s share of the {t.get('timeout_s', 300)}s deadline"
        except ExecutionDisabled:
            raise
        except Exception as e:
            receipt["status"], receipt["error"] = "agent_error", sb.redact(str(e))[:300]
        finally:
            try:
                sb.unlock_tree(ws)       # trusted stays locked until verification is done
            except OSError as e:         # e.g. a name Win32 cannot address: report it instead of crashing the request
                receipt["status"] = receipt["status"] or "sandbox_error"
                receipt["error"] = receipt["error"] or sb.redact(f"could not unlock workspace: {e}")[:300]
        try:
            after = sb.scan(ws)
        except OSError as e:
            after = {"files": before, "problems": []}
            receipt["status"] = receipt["status"] or "sandbox_error"
            receipt["error"] = receipt["error"] or sb.redact(f"could not scan workspace: {e}")[:300]
        receipt["files_after"] = after["files"]
        if receipt["status"] is None:
            _evaluate(receipt, write, allowed, before, after, pristine)
        else:
            receipt["diff"] = sb.diff(before, after["files"])
        tampered = sb.trusted_dir_tampered(trusted, trusted_bytes) if receipt["status"] is None else []
        if tampered:      # the owner can reset an ACL, so detect it: verification would still use the in-memory bytes
            receipt["status"], receipt["error"] = "protected_file_violation", "trusted test directory was tampered with: " + ", ".join(tampered)[:250]
            receipt["protected_violations"] += tampered
        if receipt["status"] is None:
            sb.build_verify_dir(ws, fixture, trusted, allowed, vdir, trusted_bytes=trusted_bytes)
            receipt["verify"] = await _verify(vdir, report, t["verify"], max(1.0, min(vbudget, deadline - time.time())))
            receipt["status"] = "verified" if receipt["verify"]["report_ok"] else "verification_failed"
            if receipt["status"] == "verification_failed":
                receipt["error"] = f"acceptance failed (exit {receipt['verify']['exit_code']}, report {receipt['verify']['report']})"
    finally:
        errs = [sb.remove_tree(p) for p in (ws, trusted, vdir) if p.exists()]
        report.unlink(missing_ok=True)
        try:
            report.parent.rmdir()      # only succeeds when empty (e.g. no concurrent run)
        except OSError:
            pass
        receipt["cleanup_error"] = next((e for e in errs if e), None)
        receipt["workspace_deleted"] = not any(p.exists() for p in (ws, trusted, vdir))
    receipt["error"] = sb.redact(receipt["error"]) if receipt["error"] else None
    if receipt["status"] != "verified":
        lid = lessons.record_failure("facility", model, f"{receipt['status']}: {receipt['error']}")
        receipt["candidate_lessons"].append({"id": lid, "status": "pending", "verified": 0})
    receipt["attestation"] = facility_attestation.attestation_for(receipt)   # signed claims for the approval gate; None if unconfigured
    receipt["finished"] = time.time()
    receipt["receipt_path"] = _write_receipt(receipt)
    return receipt


async def run_guarded(task: dict, runner=None) -> dict:
    """run_task behind the concurrency and hourly limits. Raises Busy when full."""
    admit()
    try:
        return await run_task(task, runner)
    except (TaskError, ExecutionDisabled):
        if _starts:
            _starts.pop()      # a request refused before running must not use up the hourly allowance
        raise
    finally:
        release()
