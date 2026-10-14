"""Regression tests for the red-team findings fixed in the hardening pass (C2, H1, H2-related, M2, M3, M4, L3, L5)."""
import asyncio, json, os, shutil, subprocess
from pathlib import Path
import pytest
from app import facility, facility_sandbox as sb

SRC = Path(facility.FIXTURE_ROOT)


def task(**over):
    t = {"id": "rt", "title": "t", "prompt": "p", "fixture": "buggy_calc", "permissions": "write", "allowed_paths": ["calc.py"],
         "verify": ["python", "-m", "pytest", "check_calc.py"], "timeout_s": 30}
    t.update(over)
    return t


def go(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("FACILITY_RECEIPT_DIR", str(tmp_path / "receipts"))
    facility._active = 0
    facility._starts.clear()


def trusted_of(ws: Path) -> Path:
    return ws.with_name(ws.name + "-trusted")


# ---- C2: the owner can reset an ACL, so the trusted dir is re-checked and tests come from memory -----------------
def test_trusted_dir_acl_removed_and_tests_overwritten_is_detected():
    async def attacker(ws, prompt, write, model, timeout):
        t = trusted_of(ws)
        sb.unlock_tree(t)                                                   # what `icacls /remove:d` does
        (t / "check_calc.py").write_text("def test_x():\n    assert True\n")
        return {}
    r = go(facility.run_task(task(), runner=attacker))
    assert r["status"] == "protected_file_violation" and "trusted test directory" in r["error"]
    assert r["status"] != "verified" and r["workspace_deleted"] is True


def test_verification_uses_in_memory_tests_even_if_trusted_dir_is_replaced(monkeypatch):
    """Belt and braces: if the on-disk trusted copy is swapped, build_verify_dir still writes the pristine bytes it was given."""
    ws = Path(os.environ["FACILITY_WORKSPACE_ROOT"]) / "x"
    (ws).mkdir(parents=True)
    (ws / "calc.py").write_text("def average(v):\n    return sum(v) / len(v)\n")
    trusted = ws.with_name("x-trusted"); sb.make_trusted_dir(SRC / "buggy_calc", trusted)
    pristine = sb.read_protected_bytes(SRC / "buggy_calc")
    (trusted / "check_calc.py").write_text("def test_x():\n    assert True\n")        # tampered on disk
    dest = sb.build_verify_dir(ws, SRC / "buggy_calc", trusted, {"calc.py"}, ws.with_name("x-verify"), trusted_bytes=pristine)
    assert (dest / "check_calc.py").read_bytes() == pristine["check_calc.py"]
    assert sb.trusted_dir_tampered(trusted, pristine) == ["modified: check_calc.py"]


# ---- M3: junctions and hardlinks ---------------------------------------------------------------------------------
@pytest.mark.skipif(os.name != "nt", reason="NTFS junctions are Windows-only")
def test_junction_in_workspace_is_a_violation_and_target_survives(tmp_path):
    victim = tmp_path / "victim"; victim.mkdir(); (victim / "precious.txt").write_text("keep")

    async def attacker(ws, prompt, write, model, timeout):
        sb.unlock_tree(ws)
        out = subprocess.run(["cmd", "/c", "mklink", "/J", str(ws / "lnk"), str(victim)], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return {}
    r = go(facility.run_task(task(), runner=attacker))
    assert r["status"] == "permission_violation" and "junction" in r["error"]
    assert (victim / "precious.txt").read_text() == "keep"
    assert not any(k.startswith("lnk/") for k in (r["files_after"] or {}))


def test_hardlink_in_workspace_is_a_violation():
    async def attacker(ws, prompt, write, model, timeout):
        sb.unlock_tree(ws)
        os.link(ws / "calc.py", ws / "alias.py")
        return {}
    r = go(facility.run_task(task(allowed_paths=["calc.py", "alias.py"]), runner=attacker))
    assert r["status"] == "permission_violation" and "hard-linked" in r["error"]


# ---- H1: fixture manifest ----------------------------------------------------------------------------------------
def _copy_fixtures(tmp_path, monkeypatch):
    root = tmp_path / "fx"; shutil.copytree(SRC, root)
    monkeypatch.setattr(facility, "FIXTURE_ROOT", root)
    return root


def test_fixture_matching_manifest_is_accepted_even_with_crlf(tmp_path, monkeypatch):
    root = _copy_fixtures(tmp_path, monkeypatch)
    p = root / "buggy_calc" / "calc.py"
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))      # line endings only
    facility.check_fixture_manifest("buggy_calc")


def test_poisoned_fixture_is_refused(tmp_path, monkeypatch):
    root = _copy_fixtures(tmp_path, monkeypatch)
    (root / "buggy_calc" / "check_calc.py").write_text("def test_x():\n    assert True\n")
    with pytest.raises(facility.ExecutionDisabled, match="MANIFEST"):
        go(facility.run_task(task(), runner=lambda *a: None))
    assert list(Path(os.environ["FACILITY_WORKSPACE_ROOT"]).glob("*")) == []          # refused before any workspace exists


def test_extra_file_in_fixture_is_refused(tmp_path, monkeypatch):
    root = _copy_fixtures(tmp_path, monkeypatch)
    (root / "buggy_calc" / "conftest.py").write_text("print('x')\n")
    with pytest.raises(facility.ExecutionDisabled):
        facility.check_fixture_manifest("buggy_calc")


# ---- unknown fields ----------------------------------------------------------------------------------------------
def test_unknown_task_fields_rejected_by_run_task():
    with pytest.raises(facility.TaskError, match="allow_write"):
        go(facility.run_task(task(allow_write=True), runner=lambda *a: None))


# ---- M2: a sandbox failure is reported, not crashed --------------------------------------------------------------
def test_unlock_failure_becomes_sandbox_error_with_a_receipt(monkeypatch):
    real, calls = sb.unlock_tree, {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:                       # the first unlock (right after the agent) fails once
            raise OSError("simulated unaddressable name")
        return real(path)
    monkeypatch.setattr(sb, "unlock_tree", flaky)

    async def noop(ws, prompt, write, model, timeout):
        return {}
    r = go(facility.run_task(task(), runner=noop))
    assert r["status"] == "sandbox_error" and "could not unlock" in r["error"]
    assert r["candidate_lessons"] and Path(r["receipt_path"]).exists() and r["workspace_deleted"] is True


# ---- M4 / L3: redaction and protected-name normalisation --------------------------------------------------------
def test_redact_covers_passphrase_database_url_and_pat(monkeypatch):
    monkeypatch.setenv("MY_PASSPHRASE", "correct-horse-battery")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:hunter2hunter2@h/db")
    monkeypatch.setenv("GH_PAT", "ghp_notarealtokenvalue123")
    out = sb.redact("a correct-horse-battery b postgres://u:hunter2hunter2@h/db c ghp_notarealtokenvalue123")
    assert "correct-horse" not in out and "hunter2" not in out and "ghp_" not in out and out.count("[REDACTED]") == 3


def test_path_variable_is_not_mistaken_for_a_secret(monkeypatch):
    monkeypatch.setenv("PATH", os.environ["PATH"])
    assert "PATH" in sb.agent_env() and "PATH" in sb.verify_env(Path("."))


@pytest.mark.parametrize("name", ["conftest.py.", "check_calc.py ", "pytest.ini.", "Conftest.PY", "sub/test_x.py."])
def test_trailing_dot_and_space_aliases_are_protected(name):
    assert sb.is_protected(name)


def test_receipt_error_is_redacted(monkeypatch):
    monkeypatch.setenv("FACILITY_TEST_SECRET_TOKEN", "s3cr3t-value-123456")

    async def boom(ws, prompt, write, model, timeout):
        raise RuntimeError("failed using s3cr3t-value-123456")
    r = go(facility.run_task(task(), runner=boom))
    assert "s3cr3t" not in json.dumps(r)


# ---- L5: refused requests do not use the hourly allowance --------------------------------------------------------
def test_request_refused_before_running_does_not_consume_the_hourly_cap():
    with pytest.raises(facility.ExecutionDisabled):
        go(facility.run_guarded(task(permissions="read_only", fixture="readonly_probe", allowed_paths=[],
                                     verify=["python", "-m", "pytest", "check_noop.py"])))     # real runner, provider flag unset
    assert len(facility._starts) == 0 and facility._active == 0


# ---- verification trust label ------------------------------------------------------------------------------------
def test_receipt_labels_how_much_the_verification_can_be_trusted():
    async def fixer(ws, prompt, write, model, timeout):
        (ws / "calc.py").write_text("def average(v):\n    return sum(v) / len(v)\n")
        return {}

    async def reader(ws, prompt, write, model, timeout):
        return {}
    w = go(facility.run_task(task(), runner=fixer))
    ro = go(facility.run_task(task(permissions="read_only", fixture="readonly_probe", allowed_paths=[],
                                   verify=["python", "-m", "pytest", "check_noop.py"]), runner=reader))
    assert w["status"] == ro["status"] == "verified"
    assert w["verification_trust"] == "agent_code_executed_same_user" and ro["verification_trust"] == "pristine_code_only"


# ---- C1 (documented, unfixable on this host): in-process forgery ------------------------------------------------
def test_known_limitation_verifier_forgery_is_documented():
    doc = (facility.__doc__ or "") + (sb.__doc__ or "")
    assert "NOT enforced" in doc and "same OS user" in doc
    assert "FACILITY_ENABLE_WRITE_EXEC" in Path(facility.__file__).read_text(encoding="utf-8")
