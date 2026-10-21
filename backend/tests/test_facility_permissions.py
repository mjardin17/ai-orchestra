"""Adversarial tests for blocker B2: execution permissions enforced by OS-level controls, not by prompts or CLI flags.

Each test says what is ENFORCED (and proves it against the real OS) or documents a KNOWN LIMITATION (test_known_limitation_*) so
the limitation can never silently disappear from the docs. No test calls a model, the claude CLI, or the internet. Every
filesystem experiment lives under pytest's tmp_path. Tests that need a Windows privilege skip with a reason when the OS refuses."""
import asyncio, json, os, socket, subprocess, sys, threading, time
from pathlib import Path
import pytest
from app import claude_client, facility, facility_sandbox as sb, store

FIX = facility.FIXTURE_ROOT


def go(coro):
    return asyncio.run(coro)


def task(**over):
    t = {"id": "perm1", "title": "probe", "prompt": "probe", "fixture": "readonly_probe", "permissions": "read_only",
         "verify": ["python", "-m", "pytest", "check_noop.py"], "timeout_s": 60}
    t.update(over)
    return t


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("FACILITY_WORKSPACE_ROOT", str(tmp_path / "wsroot"))
    store.DB.execute("delete from lessons"); store.DB.commit()


@pytest.fixture
def track():
    """Register trees that may be OS-locked; they are unlocked and deleted at teardown so tmp_path cleanup never trips."""
    roots = []
    yield roots.append
    for r in roots:
        sb.remove_tree(Path(r))


def tree(base: Path) -> Path:
    root = base / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    (root / "sub" / "b.txt").write_text("beta")
    for extra in ("ren.txt", "del.txt"):          # separate files so rename/delete experiments do not disturb each other
        (root / extra).write_text(extra)
    return root


def attempts(root: Path) -> dict:
    """Try every write-ish operation in-process. Returns {op: 'blocked' | 'allowed'}."""
    def run(fn):
        try:
            fn()
            return "allowed"
        except PermissionError:
            return "blocked"
    res = {
        "create_file": run(lambda: (root / "new.txt").write_text("x")),
        "create_in_subdir": run(lambda: (root / "sub" / "new.txt").write_text("x")),
        "modify": run(lambda: (root / "a.txt").open("a").write("x")),
        "modify_nested": run(lambda: (root / "sub" / "b.txt").open("a").write("x")),
        "mkdir_root": run(lambda: (root / "newdir").mkdir()),
        "mkdir_subdir": run(lambda: (root / "sub" / "newdir").mkdir()),
        "rename": run(lambda: os.rename(root / "ren.txt", root / "renamed.txt")),
        "delete": run(lambda: os.remove(root / "del.txt")),
    }
    return res


CHILD = r"""
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
def run(fn):
    try:
        fn(); return "allowed"
    except PermissionError:
        return "blocked"
print(json.dumps({
 "create_file": run(lambda: (root / "child_new.txt").write_text("x")),
 "modify": run(lambda: (root / "a.txt").open("a").write("x")),
 "delete": run(lambda: os.remove(root / "del.txt")),
 "rename": run(lambda: os.rename(root / "ren.txt", root / "child_renamed.txt")),
 "mkdir": run(lambda: (root / "child_dir").mkdir()),
 "read": (root / "a.txt").read_text(),
}))
"""


def child_attempts(root: Path) -> dict:
    r = subprocess.run([sys.executable, "-c", CHILD, str(root)], capture_output=True, text=True, timeout=30, cwd=str(root))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# ---- 1. OS lock primitives ------------------------------------------------------------------------------------------
def test_lock_tree_blocks_every_write_but_not_reads(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_tree(root)
    assert set(attempts(root).values()) == {"blocked"}, attempts(root)
    assert (root / "a.txt").read_text() == "alpha" and (root / "sub" / "b.txt").read_text() == "beta"
    assert sorted(p.name for p in root.iterdir()) == ["a.txt", "del.txt", "ren.txt", "sub"]       # listing still works


def test_unlock_tree_restores_write_access(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_tree(root); sb.unlock_tree(root)
    assert set(attempts(root).values()) == {"allowed"}


def test_child_process_is_equally_blocked_by_lock_tree(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_tree(root)
    got = child_attempts(root)
    assert got.pop("read") == "alpha"
    assert set(got.values()) == {"blocked"}, got


def test_remove_tree_deletes_a_locked_tree(tmp_path):
    root = tree(tmp_path)
    sb.lock_tree(root)
    assert sb.remove_tree(root) is None
    assert not root.exists()


def test_remove_tree_reports_error_string_instead_of_raising(tmp_path, monkeypatch):
    root = tree(tmp_path)
    monkeypatch.setattr(sb.shutil, "rmtree", lambda p: (_ for _ in ()).throw(PermissionError("denied")))
    try:
        assert "denied" in sb.remove_tree(root)
    finally:
        monkeypatch.undo()
        sb.remove_tree(root)


def test_lock_files_blocks_modification_of_listed_files_only(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_files([root / "a.txt"])
    with pytest.raises(PermissionError):
        (root / "a.txt").open("a").write("x")
    assert (root / "a.txt").read_text() == "alpha"
    # everything else stays writable: new sibling file, other existing file, new dir
    (root / "sibling.txt").write_text("ok")
    (root / "sub" / "b.txt").open("a").write("more")
    (root / "newdir").mkdir()


# DEFECT (facility_sandbox.py:55-61 + docstring "unwritable, undeletable and un-renamable"): lock_files denies WD,AD,DE,WA,WEA on the file
# only. Windows lets a caller delete or rename a file when the PARENT directory grants FILE_DELETE_CHILD / FILE_ADD_FILE even if the file
# itself denies DELETE, and lock_files leaves the directory writable on purpose. Net effect: in a write-mode run an agent CAN delete or
# rename a locked, non-allowed file. run_task still sees it as a diff (removed/added) and reports permission_violation, so it is DETECTED
# but not BLOCKED, contradicting the docstring and the facility.py module doc ("every existing file NOT on allowed_paths is OS-locked").
@pytest.mark.xfail(strict=True, reason="DEFECT facility_sandbox.py:55-61: lock_files does not block deleting a locked file (detected, not blocked)")
def test_lock_files_blocks_delete_of_locked_file(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_files([root / "del.txt"])
    with pytest.raises(PermissionError):
        os.remove(root / "del.txt")


@pytest.mark.xfail(strict=True, reason="DEFECT facility_sandbox.py:55-61: lock_files does not block renaming a locked file (detected, not blocked)")
def test_lock_files_blocks_rename_of_locked_file(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_files([root / "ren.txt"])
    with pytest.raises(PermissionError):
        os.rename(root / "ren.txt", root / "renamed.txt")


def test_child_process_blocked_by_lock_files_for_modify(tmp_path, track):
    root = tree(tmp_path); track(root)
    sb.lock_files([root / "a.txt"])
    got = child_attempts(root)
    assert got["modify"] == "blocked"
    assert got["create_file"] == "allowed"          # only the listed file is locked, not its directory


def test_run_task_detects_deletion_of_locked_file_even_if_os_allows_it(tmp_path):
    """End to end: whatever the OS lets a write-mode agent do to a locked non-allowed file, the receipt must not say verified."""
    async def deleter(ws, prompt, write, model, timeout):
        try:
            os.remove(ws / "check_calc.py")
        except PermissionError:
            pass
        try:
            (ws / "check_calc.py").write_text("def test_x():\n    assert True\n")
        except PermissionError:
            pass
        return {"text": "", "exit_code": 0}
    r = go(facility.run_task(task(fixture="buggy_calc", permissions="write", allowed_paths=["calc.py"],
                                  verify=["python", "-m", "pytest", "check_calc.py"]), runner=deleter))
    assert r["status"] != "verified"
    assert r["workspace_deleted"] is True


# ---- 2. path traversal / argument injection -------------------------------------------------------------------------
BAD_PATHS = ["../x.py", "a/../../x.py", "..", "/etc/passwd", "/x.py", "C:/x.py", "C:\\x.py", "c:x.py", "a\\b.py", "..\\x.py",
             "x.py;rm", "x.py&calc", "x.py|y", "$(whoami).py", "`id`.py", "a b.py", "x.py\x00", "~/x.py", "-x.py", ".hidden/../x",
             "a//b.py", "dir/", "", "x" * 200]


@pytest.mark.parametrize("bad", BAD_PATHS)
def test_allowed_paths_rejects_traversal_and_metacharacters(bad):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(fixture="buggy_calc", permissions="write", allowed_paths=[bad]))


@pytest.mark.parametrize("bad", ["../check_calc.py", "/abs/check_calc.py", "C:/check_calc.py", "C:\\check_calc.py", "a\\b.py",
                                 "--rootdir=/", "-p", "-c", "-o", "--", "-k", "x.py;calc", "x.py&&calc", "x.py|calc",
                                 "$(id)", "`id`", "a b.py", "x\x00.py", "--rootdir", "-pevil", "-p evil", "..", "a/../../x.py"])
def test_verify_args_reject_traversal_option_injection_and_metacharacters(bad):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(verify=["python", "-m", "pytest", bad]))


@pytest.mark.parametrize("verify", [["python", "-m", "pytest"], ["sh", "-c", "id"], ["python", "-c", "print(1)"],
                                    ["python", "-m", "pytest", "-p", "evil", "x.py"], "python -m pytest x.py", None])
def test_verify_command_shape_is_locked_down(verify):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(verify=verify))


@pytest.mark.parametrize("field", ["id", "fixture"])
@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "..", ".", "", "x" * 65, "a b", "C:x", "a;b"])
def test_id_and_fixture_reject_traversal(field, bad):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(**{field: bad}))


@pytest.mark.parametrize("field,value", [("id", "t1\n")])   # fixture "x\n" is caught later by the is_dir check
def test_trailing_newline_is_rejected_in_names(field, value):
    with pytest.raises(facility.TaskError):
        facility.validate_task(task(**{field: value}))


@pytest.mark.parametrize("kind", ["verify", "allowed"])
def test_trailing_newline_is_rejected_in_paths(kind):
    t = task(verify=["python", "-m", "pytest", "check_noop.py\n"]) if kind == "verify" else \
        task(fixture="buggy_calc", permissions="write", allowed_paths=["calc.py\n"])
    with pytest.raises(facility.TaskError):
        facility.validate_task(t)


@pytest.mark.parametrize("name", ["conftest.py", "sub/conftest.py", "CONFTEST.PY", "sitecustomize.py", "usercustomize.py", "evil.pth",
                                  "sub/evil.PTH", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "check_calc.py",
                                  "test_calc.py", "calc_test.py", "calc_check.py", "deep/dir/test_x.py"])
def test_is_protected_recognises_test_and_config_files(name):
    assert sb.is_protected(name)


@pytest.mark.parametrize("name", ["calc.py", "notes.txt", "README.md", "src/calc.py", "contest.py", "testing.py"])
def test_is_protected_does_not_over_match(name):
    assert not sb.is_protected(name)


def test_protected_violations_flag_modified_removed_and_added():
    pristine = {"check_calc.py": "h1", "calc.py": "h2", "conftest.py": "h3"}
    after = {"check_calc.py": "CHANGED", "calc.py": "h2x", "new_conftest.py": "h", "sitecustomize.py": "h", "x.pth": "h"}
    v = sb.protected_violations(pristine, after)
    assert "modified: check_calc.py" in v and "removed: conftest.py" in v
    assert "added: sitecustomize.py" in v and "added: x.pth" in v
    assert not any("calc.py" == x.split(": ")[1] for x in v)         # unprotected edits are not protected violations


# ---- 3. symlinks ----------------------------------------------------------------------------------------------------
def _can_symlink(tmp_path: Path) -> bool:
    src = tmp_path / "_probe_target"; src.write_text("x")
    try:
        os.symlink(src, tmp_path / "_probe_link")
        return True
    except (OSError, NotImplementedError):
        return False


@pytest.fixture
def symlinks_ok(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("host refuses symlink creation (Windows needs Developer Mode or SeCreateSymbolicLinkPrivilege)")


def test_scan_reports_symlinked_file_and_dir(tmp_path, symlinks_ok):
    root = tmp_path / "w"; root.mkdir()
    (root / "real.txt").write_text("r")
    outside = tmp_path / "outside_t"; outside.mkdir(); (outside / "secret.txt").write_text("s")
    os.symlink(outside / "secret.txt", root / "flink.txt")
    os.symlink(outside, root / "dlink", target_is_directory=True)
    res = sb.scan(root)
    assert "real.txt" in res["files"] and "flink.txt" not in res["files"]
    joined = " ".join(res["problems"])
    assert "flink.txt" in joined and "dlink" in joined
    assert "secret.txt" not in res["files"] and not any("secret.txt" in k for k in res["files"])   # never followed


def test_run_task_flags_symlink_planted_by_agent(tmp_path, symlinks_ok):
    secret = tmp_path / "secret.txt"; secret.write_text("TOPSECRET")

    async def planter(ws, prompt, write, model, timeout):
        sb.unlock_tree(ws)      # simulates an agent that has defeated the lock; the symlink check is the second line of defence
        os.symlink(secret, ws / "leak.txt")
        return {"text": "", "exit_code": 0}
    r = go(facility.run_task(task(), runner=planter))
    assert r["status"] == "permission_violation" and "leak.txt" in r["error"]
    assert r["verify"] is None and r["workspace_deleted"] is True
    assert secret.read_text() == "TOPSECRET"        # deleting the workspace did not follow the link


def test_run_task_flags_directory_symlink_planted_by_agent(tmp_path, symlinks_ok):
    outside = tmp_path / "outside_dir"; outside.mkdir(); (outside / "keep.txt").write_text("keep")

    async def planter(ws, prompt, write, model, timeout):
        sb.unlock_tree(ws)
        os.symlink(outside, ws / "dl", target_is_directory=True)
        return {"text": "", "exit_code": 0}
    r = go(facility.run_task(task(), runner=planter))
    assert r["status"] == "permission_violation"
    assert (outside / "keep.txt").read_text() == "keep"


def test_pristine_fixture_containing_symlink_is_a_task_error(tmp_path, symlinks_ok, monkeypatch):
    fx = tmp_path / "fixroot"; d = fx / "bad"; d.mkdir(parents=True)
    (d / "check_x.py").write_text("def test_x():\n    assert True\n")
    target = tmp_path / "tgt.txt"; target.write_text("t")
    os.symlink(target, d / "link.txt")
    monkeypatch.setattr(facility, "FIXTURE_ROOT", fx)
    # run_task checks the provider gate before the fixture scan. Enable it so the test reaches the symlink check.
    # The scan raises TaskError before any provider is called, so nothing is executed.
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    with pytest.raises(facility.TaskError, match="clean regular-file tree"):
        go(facility.run_task(task(fixture="bad", verify=["python", "-m", "pytest", "check_x.py"]), runner=None))


# ---- 4. writes outside the workspace --------------------------------------------------------------------------------
def test_trusted_dir_is_locked_while_the_agent_runs_even_for_child_processes(tmp_path):
    seen = {}

    async def attacker(ws, prompt, write, model, timeout):
        trusted = ws.with_name(ws.name + "-trusted")
        seen["exists"] = trusted.is_dir()
        victim = next(trusted.glob("check_*.py"))
        seen["read"] = victim.read_text()                       # reading is allowed
        for label, fn in (("modify", lambda: victim.open("a").write("assert False")),
                          ("delete", lambda: os.remove(victim)),
                          ("create", lambda: (trusted / "conftest.py").write_text("import os")),
                          ("mkdir", lambda: (trusted / "d").mkdir())):
            try:
                fn(); seen[label] = "allowed"
            except PermissionError:
                seen[label] = "blocked"
        code = f"from pathlib import Path\ntry:\n Path(r'{victim}').open('a').write('x'); print('allowed')\nexcept PermissionError: print('blocked')"
        seen["child_modify"] = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30).stdout.strip()
        return {"text": "", "exit_code": 0}
    r = go(facility.run_task(task(), runner=attacker))
    assert seen["exists"] and seen["read"]
    assert {k: seen[k] for k in ("modify", "delete", "create", "mkdir", "child_modify")} == {k: "blocked" for k in
                                                                                             ("modify", "delete", "create", "mkdir", "child_modify")}
    assert r["status"] == "verified"            # the attempts changed nothing
    assert r["workspace_deleted"] is True       # locked trusted dir was still cleaned up afterwards


def test_known_limitation_writes_outside_workspace_are_not_blocked(tmp_path):
    """KNOWN LIMITATION (B2): the agent runs as the same OS user, so the OS lock does not stop it writing elsewhere. This test
    proves the limitation is real AND that the module docs still say so."""
    outside = tmp_path / "outside"; outside.mkdir()

    async def escaper(ws, prompt, write, model, timeout):
        (outside / "escaped.txt").write_text("written outside the workspace")
        return {"text": "", "exit_code": 0}
    r = go(facility.run_task(task(), runner=escaper))
    assert (outside / "escaped.txt").read_text() == "written outside the workspace"
    assert r["status"] == "verified"            # the receipt cannot see it either: this is exactly what the docs must warn about
    docs = (facility.__doc__ or "") + (sb.__doc__ or "")
    assert "NOT enforced" in facility.__doc__
    assert "does NOT" in sb.__doc__ and "sandbox code" in sb.__doc__
    assert "outside the workspace" in docs and "same OS user" in docs


def test_known_limitation_network_not_blocked():
    """KNOWN LIMITATION (B2): nothing blocks network access from the agent or verifier. A loopback listener (127.0.0.1 only, no
    internet) shows a child process connects freely, and the docs must say network is not enforced."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    got = {}
    t = threading.Thread(target=lambda: got.setdefault("data", srv.accept()[0].recv(16)), daemon=True)
    t.start()
    try:
        code = f"import socket; s=socket.create_connection(('127.0.0.1',{port}),5); s.sendall(b'hello'); print('connected')"
        env = sb.verify_env(Path(os.getcwd()))                  # the very environment a verifier child gets
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20, env=env)
        t.join(5)
    finally:
        srv.close()
    assert r.stdout.strip() == "connected", r.stderr
    assert got.get("data") == b"hello"
    assert "network access" in facility.__doc__ and "NOT enforced" in facility.__doc__
    assert "network" in sb.__doc__ and "does NOT" in sb.__doc__


# ---- 5. tool surface / CLI flags (proves what we PASS to the CLI, NOT that the CLI honours it) ----------------------------
@pytest.fixture
def argv(monkeypatch):
    monkeypatch.setattr(claude_client, "cli_path", lambda: "claude")
    return lambda write: facility.build_agent_argv(write, "claude:haiku")


def after(a: list, flag: str) -> str:
    return a[a.index(flag) + 1]


def test_argv_read_only_tool_surface(argv):
    # NOTE: only proves the arguments we hand to the CLI; whether the CLI enforces them needs a live model call (not made here).
    a = argv(False)
    assert after(a, "--tools") == "Read,Glob,Grep" and after(a, "--allowedTools") == "Read,Glob,Grep"
    for banned in ("Edit", "Write", "Bash", "PowerShell", "NotebookEdit", "WebFetch", "Task"):
        assert banned not in after(a, "--tools").split(",")
        assert banned not in after(a, "--allowedTools").split(",")
    denied = after(a, "--disallowedTools").split(",")
    assert {"Bash", "PowerShell", "WebFetch", "WebSearch", "Task"} <= set(denied)
    assert after(a, "--permission-mode") == "dontAsk"


def test_argv_write_adds_only_edit_and_write(argv):
    a = argv(True)
    assert after(a, "--tools").split(",") == ["Read", "Glob", "Grep", "Edit", "Write"]
    assert not {"Bash", "PowerShell", "WebFetch", "WebSearch", "Task"} & set(after(a, "--tools").split(","))
    assert {"Bash", "PowerShell"} <= set(after(a, "--disallowedTools").split(","))
    assert after(a, "--permission-mode") == "acceptEdits"


@pytest.mark.parametrize("write", [False, True])
def test_argv_isolates_settings_mcp_skills_and_prompts(argv, write):
    a = argv(write)
    assert "--strict-mcp-config" in a and "--disable-slash-commands" in a and "--no-session-persistence" in a
    assert after(a, "--setting-sources") == "project"            # user + local settings (global allow rules) are NOT loaded
    assert after(a, "--permission-prompts") == "none"
    assert "--mcp-config" not in a                                # no MCP servers are supplied either
    assert "bypassPermissions" not in a and "--dangerously-skip-permissions" not in a
    assert "-p" in a and a[0] == "claude"


def test_ensure_exec_enabled_fails_closed(monkeypatch):
    monkeypatch.delenv("FACILITY_ENABLE_PROVIDER_EXEC", raising=False)
    monkeypatch.delenv("FACILITY_ENABLE_WRITE_EXEC", raising=False)
    for w in (False, True):
        with pytest.raises(facility.ExecutionDisabled):
            facility.ensure_exec_enabled(w)
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    facility.ensure_exec_enabled(False)
    with pytest.raises(facility.ExecutionDisabled):
        facility.ensure_exec_enabled(True)
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "true")      # only the literal "1" enables it
    with pytest.raises(facility.ExecutionDisabled):
        facility.ensure_exec_enabled(False)


def test_run_task_refuses_real_runner_when_disabled_before_creating_anything(tmp_path, monkeypatch):
    monkeypatch.delenv("FACILITY_ENABLE_PROVIDER_EXEC", raising=False)
    with pytest.raises(facility.ExecutionDisabled):
        go(facility.run_task(task(), runner=facility.claude_runner))
    assert list((tmp_path / "wsroot").glob("*")) == [] if (tmp_path / "wsroot").exists() else True


# ---- 6. environment -------------------------------------------------------------------------------------------------
PLANTED = {"ANTHROPIC_API_KEY": "sk-ant-PLANTED-0001", "OPENAI_API_KEY": "sk-PLANTED-0002", "MY_SERVICE_TOKEN": "tok-PLANTED-0003",
           "DB_PASSWORD": "pw-PLANTED-0004", "CLIENT_SECRET": "sec-PLANTED-0005", "AUTH_HEADER": "auth-PLANTED-0006",
           "SESSION_ID": "sess-PLANTED-0007", "MY_COOKIE": "cook-PLANTED-0008", "CLAUDE_CODE_OAUTH_TOKEN": "oauth-PLANTED-0009",
           "AWS_SECRET_ACCESS_KEY": "aws-PLANTED-0010", "HTTP_PROXY": "http://proxy.invalid:8080", "HTTPS_PROXY": "http://proxy.invalid:8080",
           "ALL_PROXY": "http://proxy.invalid:8080", "NO_PROXY": "x", "GITHUB_PAT": "ghp-PLANTED-0011", "DATABASE_URL": "postgres://u:p@h/db"}


def test_agent_and_verify_env_never_carry_secrets_or_proxies(monkeypatch, tmp_path):
    for k, v in PLANTED.items():
        monkeypatch.setenv(k, v)
    vdir = tmp_path / "v"; vdir.mkdir()
    for env in (sb.agent_env(), sb.verify_env(vdir)):
        assert not [k for k in env if sb.SECRET_NAME.search(k)]
        assert not set(PLANTED) & {k.upper() for k in env}, set(PLANTED) & {k.upper() for k in env}
        assert not [v for v in env.values() if "PLANTED" in v]
    assert "PATH" in {k.upper() for k in sb.agent_env()}          # allow-list is not empty: the CLI can still start


def test_verify_env_points_home_and_temp_into_the_throwaway_dir(tmp_path):
    vdir = tmp_path / "v"; vdir.mkdir()
    env = sb.verify_env(vdir)
    for k in ("HOME", "USERPROFILE", "TEMP", "TMP"):
        assert Path(env[k]) == vdir / ".home"
    assert "APPDATA" not in env and "LOCALAPPDATA" not in env and "CLAUDE_CONFIG_DIR" not in env


def test_redact_hides_planted_secret_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-PLANTED-REDACT-1234")
    assert "PLANTED-REDACT" not in sb.redact("leak: sk-ant-PLANTED-REDACT-1234 end")


# ---- 8. kill_tree ---------------------------------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _force_kill(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def test_kill_tree_kills_grandchildren(tmp_path):
    if sys.platform == "win32" and not subprocess.run(["tasklist", "/FI", "PID eq 4", "/NH"], capture_output=True).returncode == 0:
        pytest.skip("tasklist unavailable; cannot observe process liveness")
    pidfile = tmp_path / "grandchild.pid"
    parent = ("import subprocess, sys, time\n"
              f"g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
              f"open(r'{pidfile}', 'w').write(str(g.pid))\n"
              "time.sleep(60)\n")
    # Start it as a process-group leader, the way the facility starts its agent and verifier. kill_tree relies on that.
    proc = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    gpid = None
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not (pidfile.exists() and pidfile.read_text().strip()):
            time.sleep(0.1)
        assert pidfile.exists(), "grandchild never started"
        gpid = int(pidfile.read_text())
        assert _alive(proc.pid) and _alive(gpid)
        sb.kill_tree(proc)
        proc.wait(timeout=10)
        deadline = time.time() + 10
        while time.time() < deadline and (_alive(proc.pid) or _alive(gpid)):
            time.sleep(0.2)
        assert not _alive(proc.pid), "child survived kill_tree"
        assert not _alive(gpid), "grandchild survived kill_tree"
    finally:
        _force_kill(proc.pid)
        if gpid:
            _force_kill(gpid)


def test_kill_tree_is_a_noop_for_an_already_finished_process():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=20)
    sb.kill_tree(proc)          # must not raise


# ---- 9. sweep_stale / remove_tree stay inside the workspace root ---------------------------------------------------------
def _age(p: Path, hours: float) -> None:
    t = time.time() - hours * 3600
    os.utime(p, (t, t))


def test_sweep_stale_only_touches_the_workspace_root(tmp_path):
    root = sb.workspace_root()
    assert root == tmp_path / "wsroot"
    old_in, fresh_in = root / "old-aaaaaaaa", root / "fresh-bbbbbbbb"
    for d in (old_in, fresh_in):
        (d / "sub").mkdir(parents=True); (d / "sub" / "f.txt").write_text("x")
    locked_old = root / "lock-cccccccc"
    foreign_old = root / "not-ours"          # inside the root but not named like a facility workspace: never deleted
    foreign_old.mkdir(); (foreign_old / "keep.txt").write_text("keep")
    (locked_old / "sub").mkdir(parents=True); (locked_old / "sub" / "f.txt").write_text("x")
    outside_old = tmp_path / "unrelated-old"
    outside_old.mkdir(); (outside_old / "keep.txt").write_text("keep")
    sibling_old = tmp_path / "wsroot-sibling"          # same name prefix as the root: must not be confused with it
    sibling_old.mkdir(); (sibling_old / "keep.txt").write_text("keep")
    for d in (old_in, locked_old, outside_old, sibling_old, foreign_old):
        _age(d, 48)
    sb.lock_tree(locked_old)             # lock AFTER aging: a locked dir cannot have its times changed
    try:
        assert sb.sweep_stale(max_age_hours=24) == 2
    finally:
        sb.remove_tree(locked_old)
    assert not old_in.exists() and not locked_old.exists()
    assert fresh_in.exists() and (foreign_old / "keep.txt").read_text() == "keep"
    assert (outside_old / "keep.txt").read_text() == "keep" and (sibling_old / "keep.txt").read_text() == "keep"


def test_sweep_stale_does_not_follow_a_junction_out_of_the_root(tmp_path):
    """Adversarial: an old junction/symlink dropped inside the root pointing at an unrelated directory must not get its target
    emptied. (A junction needs no privilege on Windows; a symlink may be skipped by the host.)"""
    root = sb.workspace_root()
    victim = tmp_path / "victim"; victim.mkdir(); (victim / "precious.txt").write_text("precious")
    link = root / "evil-link"
    try:
        if sys.platform == "win32":
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(victim)], capture_output=True, text=True)
            if r.returncode:
                pytest.skip(f"cannot create junction: {r.stdout.strip()[:100]}")
        else:
            os.symlink(victim, link, target_is_directory=True)
    except OSError as e:
        pytest.skip(f"cannot create link: {e}")
    _age(victim, 48)
    try:
        sb.sweep_stale(max_age_hours=24)
        assert (victim / "precious.txt").read_text() == "precious", "sweep_stale followed a link and deleted outside the root"
    finally:
        if link.exists() or link.is_symlink():
            if sys.platform == "win32":
                subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)      # removes the junction only
            else:
                link.unlink()


def test_workspace_root_defaults_to_dedicated_dir_not_bare_tempdir(monkeypatch):
    monkeypatch.delenv("FACILITY_WORKSPACE_ROOT", raising=False)
    import tempfile
    assert sb.workspace_root().resolve() != Path(tempfile.gettempdir()).resolve()
    assert sb.workspace_root().name == "orchestra-facility"
