"""Filesystem and process boundary helpers for the facility task loop.

What this module really enforces (each point has a test):
- Read-only workspaces are locked at the OS level (Windows ACL deny / POSIX chmod): create, modify, delete, rename and mkdir fail
  for the agent process no matter what the prompt or CLI flags say. Reads still work.
- Protected files (tests and test-config) are locked in write workspaces, and verification never uses the agent's copies:
  it builds a separate directory from the agent's non-protected files plus PRISTINE protected files from the fixture.
- Child processes get an allow-listed environment: no API keys, tokens or secrets.
What it does NOT do: sandbox code. A verifier or agent subprocess still runs as the same OS user and can read user files and use
the network. Strong isolation needs a separate low-privilege account or container."""
import csv, functools, hashlib, io, os, re, shutil, signal, site, stat, subprocess, sys, tempfile
from pathlib import Path

PROTECTED_NAME = re.compile(
    r"^(conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|pyproject\.toml|sitecustomize\.py|usercustomize\.py"
    r"|(check|test)_.*\.py|.*_(check|test)\.py|.*\.pth)$", re.I)
IGNORED = {"__pycache__", ".pytest_cache"}
# Names mkdtemp gives our workspaces: <task-id>-<8 random chars>[-trusted|-verify]. sweep_stale never touches anything else.
OWNED_DIR = re.compile(r"[A-Za-z0-9_.-]+-[a-z0-9_]{8}(-trusted|-verify)?", re.I)
MAX_FILES, MAX_BYTES = 200, 5 * 1024 * 1024
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE|DSN|DATABASE_URL|CONNECTION_STRING|(^|_)PAT($|_)", re.I)
AGENT_ENV_KEEP = ("PATH", "PATHEXT", "SYSTEMROOT", "COMSPEC", "USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH", "APPDATA",
                  "LOCALAPPDATA", "PROGRAMDATA", "USERNAME", "USERDOMAIN", "TEMP", "TMP", "LANG", "TERM", "CLAUDE_CONFIG_DIR")
# Windows rights denied to the current user: write data/add file, add subdir, delete child, delete, write attrs, write ext attrs.
_DENY = "(OI)(CI)(WD,AD,DC,DE,WA,WEA)"
_DENY_FILE = "(WD,AD,DE,WA,WEA)"


def workspace_root() -> Path:
    r = Path(os.getenv("FACILITY_WORKSPACE_ROOT") or Path(tempfile.gettempdir()) / "orchestra-facility")
    r.mkdir(parents=True, exist_ok=True)
    return r


@functools.lru_cache(maxsize=1)
def _sid() -> str:
    out = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True).stdout
    return next(csv.reader(io.StringIO(out)))[1]


def _icacls(*args) -> None:
    r = subprocess.run(["icacls", *args, "/Q"], capture_output=True, text=True)
    if r.returncode:
        raise OSError(f"icacls failed ({r.returncode}): {(r.stdout + r.stderr)[-200:]}")


def lock_tree(root: Path) -> None:
    """Make the whole tree unwritable for this user. Raises if the lock cannot be applied (callers must fail closed)."""
    if sys.platform == "win32":
        _icacls(str(root), "/deny", f"*{_sid()}:{_DENY}", "/T")
        return
    for p in sorted(root.rglob("*"), reverse=True):
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(root, 0o555)


def lock_files(paths) -> None:
    """Make individual files unwritable. On Windows their DIRECTORY stays writable, so delete/rename of a locked file is NOT blocked
    (the parent directory right overrides); such changes are caught afterwards by the diff and reported as permission_violation."""
    for p in paths:
        if sys.platform == "win32":
            _icacls(str(p), "/deny", f"*{_sid()}:{_DENY_FILE}")
        else:
            os.chmod(p, 0o444)


def unlock_tree(root: Path) -> None:
    if not root.exists():
        return
    if sys.platform == "win32":
        _icacls(str(root), "/remove:d", f"*{_sid()}", "/T")
        return
    os.chmod(root, 0o755)
    for p in root.rglob("*"):
        os.chmod(p, 0o755 if p.is_dir() else 0o644)


def remove_tree(root: Path) -> str | None:
    """Unlock then delete. Returns an error string instead of raising so cleanup never hides the task result."""
    try:
        unlock_tree(root)
        shutil.rmtree(root)
        return None
    except OSError as e:
        return str(e)[:200]


def sweep_stale(max_age_hours: float = 24) -> int:
    """Delete leftover workspaces older than max_age_hours from OUR dedicated root only. Returns how many were removed."""
    import time
    n, cutoff = 0, time.time() - max_age_hours * 3600
    for d in workspace_root().iterdir():
        if d.is_dir() and OWNED_DIR.fullmatch(d.name) and d.stat().st_mtime < cutoff and remove_tree(d) is None:
            n += 1
    return n


def _is_reparse(st) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def scan(root: Path) -> dict:
    """Hash every regular file. Returns {"files": {rel: sha256}, "problems": [...]}. Symlinks and special files are problems,
    never followed. Too many files or bytes is a problem too."""
    files, problems, total = {}, [], 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in IGNORED]
        for d in list(dirnames):
            dp = os.path.join(dirpath, d)
            if os.path.islink(dp) or _is_reparse(os.lstat(dp)):
                problems.append(f"symlink or junction dir: {os.path.relpath(dp, root)}")
                dirnames.remove(d)         # never walk into it
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            st = p.lstat()
            if p.is_symlink() or not stat.S_ISREG(st.st_mode) or _is_reparse(st) or st.st_nlink > 1:
                problems.append(f"non-regular, reparse-point or hard-linked file: {rel}")
                continue
            total += p.stat().st_size
            files[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if total <= MAX_BYTES else ""
    if len(files) > MAX_FILES or total > MAX_BYTES:
        problems.append(f"workspace limits exceeded ({len(files)} files, {total} bytes)")
    return {"files": files, "problems": problems}


def diff(before: dict, after: dict) -> dict:
    return {"added": sorted(set(after) - set(before)), "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in before if k in after and before[k] != after[k])}


def is_protected(rel: str) -> bool:
    """Windows ignores trailing dots and spaces in names, so normalise them before matching."""
    return bool(PROTECTED_NAME.fullmatch(rel.rsplit("/", 1)[-1].rstrip(". ")))


def protected_violations(pristine: dict, after: dict) -> list:
    """Protected files that were modified or removed, plus any NEW file that looks like a test or test config."""
    out = [f"modified: {k}" for k, v in pristine.items() if is_protected(k) and k in after and after[k] != v]
    out += [f"removed: {k}" for k in pristine if is_protected(k) and k not in after]
    out += [f"added: {k}" for k in after if k not in pristine and is_protected(k)]
    return sorted(out)


def make_trusted_dir(fixture: Path, dest: Path) -> Path:
    """Copy ONLY the protected files (tests and test config) from the fixture into a directory outside the agent workspace.
    Callers lock it (lock_tree) before the agent starts and keep it locked until verification is finished."""
    dest.mkdir(parents=True)
    for rel in scan(fixture)["files"]:
        if is_protected(rel):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture / rel, dest / rel)
    return dest


def read_protected_bytes(fixture: Path) -> dict:
    """Pristine bytes of every protected file, read into MEMORY before the agent starts. The OS lock on the trusted directory is only a
    tripwire (the owner can always reset an ACL), so verification writes tests from these bytes, never from disk."""
    return {rel: (fixture / rel).read_bytes() for rel in scan(fixture)["files"] if is_protected(rel)}


def trusted_dir_tampered(trusted: Path, pristine_bytes: dict) -> list:
    """Differences between the trusted directory on disk and the in-memory pristine bytes."""
    now = scan(trusted)["files"]
    want = {k: hashlib.sha256(v).hexdigest() for k, v in pristine_bytes.items()}
    return sorted([f"modified: {k}" for k in want if k in now and now[k] != want[k]] + [f"removed: {k}" for k in want if k not in now]
                  + [f"added: {k}" for k in now if k not in want])


def build_verify_dir(ws: Path, fixture: Path, trusted: Path, allowed, dest: Path, trusted_bytes: dict = None) -> Path:
    """The tree the verifier runs in. Agent output counts ONLY for explicitly allowed paths; every other file comes from the
    pristine fixture, and test files come from the locked trusted directory. Anything else the agent left behind is ignored."""
    dest.mkdir(parents=True)
    allowed = set(allowed)
    for rel in scan(fixture)["files"]:
        if is_protected(rel):
            continue
        src = ws / rel if rel in allowed else fixture / rel
        if src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest / rel)
    for rel in allowed:      # new files the agent was explicitly allowed to create
        if (ws / rel).is_file() and not (dest / rel).exists() and not is_protected(rel):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ws / rel, dest / rel)
    if trusted_bytes is not None:            # preferred: bytes captured in memory before the agent ran
        for rel, data in trusted_bytes.items():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_bytes(data)
    else:
        for rel in scan(trusted)["files"]:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trusted / rel, dest / rel)
    if not (dest / "pytest.ini").exists():
        (dest / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (dest / ".home").mkdir(exist_ok=True)
    return dest


def parse_junit(path: Path) -> dict | None:
    """Read pytest's junit report. Returns counts, or None when the report is missing, oversized or suspicious. The exit code alone is
    never trusted: code that kills pytest early (os._exit(0)) leaves no report."""
    import xml.etree.ElementTree as ET
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(raw) > 1_000_000 or "<!DOCTYPE" in raw or "<!ENTITY" in raw:
        return None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return None
    total = {k: sum(int(x.get(k, 0)) for x in suites) for k in ("tests", "failures", "errors", "skipped")}
    return total


def _clean(env: dict) -> dict:
    return {k: v for k, v in env.items() if not SECRET_NAME.search(k)}


def agent_env() -> dict:
    """Allow-listed environment for the agent CLI: it needs its login location, never API keys or tokens."""
    return _clean({k: v for k, v in os.environ.items() if k.upper() in AGENT_ENV_KEEP})


def verify_env(vdir: Path) -> dict:
    """Environment for the verifier: minimal, with HOME/TEMP pointing into the throwaway verify dir. The user site-packages
    is passed as PYTHONPATH because APPDATA is deliberately not forwarded."""
    home = str(vdir / ".home")
    env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PATH": str(Path(sys.executable).parent) + os.pathsep +
           os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32"),
           "HOME": home, "USERPROFILE": home, "TEMP": home, "TMP": home, "PYTHONDONTWRITEBYTECODE": "1"}
    usersite = site.getusersitepackages()
    if usersite and os.path.isdir(usersite):
        env["PYTHONPATH"] = usersite
    return _clean({k: v for k, v in env.items() if v})


def redact(text: str, extra=()) -> str:
    """Replace the value of every secret-looking environment variable (and any `extra` values) found in text."""
    vals = {v for k, v in os.environ.items() if SECRET_NAME.search(k) and len(v) >= 8} | {v for v in extra if v and len(v) >= 8}
    for v in sorted(vals, key=len, reverse=True):
        text = text.replace(v, "[REDACTED]")
    return text


def kill_tree(proc) -> None:
    """Kill the process and its children. Plain kill() leaves children running on both platforms.
    On POSIX this needs the process started with start_new_session=True, so it leads its own process group."""
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)     # the whole group: the child and anything it spawned
        except OSError:
            pass                                    # group already gone, or not ours to signal
        try:
            proc.kill()
        except ProcessLookupError:
            pass                                    # the process exited between the checks above
