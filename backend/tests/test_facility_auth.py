"""B4: endpoint authorization for POST /api/tasks/run. Fails closed; write needs a signed, task-bound, expiring grant.
All secrets here are obviously fake test values set via monkeypatch. The agent is always a fake."""
import time
import pytest
from fastapi.testclient import TestClient
from app import facility, facility_auth, store
from app.main import app

URL = "/api/tasks/run"
TOKEN = "test-token-not-a-real-secret-0001"
GSECRET = "test-grant-secret-not-real-0002"
OTHER_SECRET = "different-grant-secret-fake-0003"
GRANT_HDR = "X-Facility-Write-Grant"


def ro_task(**over):
    t = {"id": "ro1", "title": "Count lines", "prompt": "Count lines in notes.txt", "fixture": "readonly_probe",
         "permissions": "read_only", "verify": ["python", "-m", "pytest", "check_noop.py"], "timeout_s": 30}
    t.update(over)
    return t


def wr_task(**over):
    t = {"id": "w1", "title": "Fix average()", "prompt": "Fix the bug in calc.py", "fixture": "buggy_calc",
         "permissions": "write", "allowed_paths": ["calc.py"],
         "verify": ["python", "-m", "pytest", "check_calc.py"], "timeout_s": 30}
    t.update(over)
    return t


class Runner:
    """Harmless fake agent; records whether it was ever called."""
    def __init__(self):
        self.calls = 0

    async def __call__(self, ws, prompt, write, model, timeout):
        self.calls += 1
        if write:
            (ws / "calc.py").write_text("def average(values):\n    return sum(values) / len(values)\n")
        return {"text": "ok", "exit_code": 0}


@pytest.fixture
def runner(monkeypatch):
    r = Runner()
    monkeypatch.setattr(facility, "DEFAULT_RUNNER", r)
    return r


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_RECEIPT_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("FACILITY_WORKSPACE_ROOT", str(tmp_path / "ws"))
    for k in ("FACILITY_API_TOKEN", "FACILITY_GRANT_SECRET", "FACILITY_ENABLE_PROVIDER_EXEC", "FACILITY_ENABLE_WRITE_EXEC",
              "FACILITY_MAX_CONCURRENT", "FACILITY_MAX_PER_HOUR"):
        monkeypatch.delenv(k, raising=False)
    facility._active = 0
    facility._starts.clear()
    store.DB.execute("delete from lessons"); store.DB.commit()
    yield
    facility._active = 0
    facility._starts.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def authed(monkeypatch):
    monkeypatch.setenv("FACILITY_API_TOKEN", TOKEN)
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def granting(monkeypatch, authed):
    monkeypatch.setenv("FACILITY_GRANT_SECRET", GSECRET)
    return authed


def post(client, body, headers=None):
    return client.post(URL, json=body, headers=headers or {})


# ---- API token ------------------------------------------------------------------------------------------------
def test_unset_token_fails_closed_503_for_any_request(client, runner):
    assert post(client, ro_task()).status_code == 503
    assert post(client, ro_task(), {"Authorization": f"Bearer {TOKEN}"}).status_code == 503
    assert client.post(URL, content=b"not json").status_code == 503
    assert runner.calls == 0


def test_short_token_is_treated_as_unset(client, runner, monkeypatch):
    monkeypatch.setenv("FACILITY_API_TOKEN", "x" * 15)
    assert post(client, ro_task(), {"Authorization": "Bearer " + "x" * 15}).status_code == 503
    assert runner.calls == 0


def test_wrong_token_is_401(client, runner, authed):
    assert post(client, ro_task(), {"Authorization": "Bearer wrong-token-value-xxxxxxxx"}).status_code == 401
    assert runner.calls == 0


def test_missing_or_malformed_header_is_401(client, runner, authed):
    assert post(client, ro_task()).status_code == 401
    assert post(client, ro_task(), {"Authorization": TOKEN}).status_code == 401           # no Bearer scheme
    assert post(client, ro_task(), {"Authorization": "Bearer "}).status_code == 401
    assert post(client, ro_task(), {"Authorization": "Basic " + TOKEN}).status_code == 401
    assert runner.calls == 0


def test_correct_token_runs_read_only_task(client, runner, authed):
    r = post(client, ro_task(), authed)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "verified" and r.json()["permissions"] == "read_only"
    assert runner.calls == 1


def test_bearer_scheme_is_case_insensitive_and_wrong_prefix_of_token_fails(client, runner, authed):
    assert post(client, ro_task(), {"Authorization": f"bearer {TOKEN}"}).status_code == 200
    assert post(client, ro_task(), {"Authorization": f"Bearer {TOKEN[:-1]}"}).status_code == 401
    assert post(client, ro_task(), {"Authorization": f"Bearer {TOKEN}x"}).status_code == 401


def test_non_json_body_is_400(client, runner, authed):
    r = client.post(URL, content=b"this is not json", headers={**authed, "Content-Type": "application/json"})
    assert r.status_code == 400
    assert runner.calls == 0


def test_non_object_json_body_is_400(client, runner, authed):
    assert post(client, ["a", "list"], authed).status_code == 400
    assert runner.calls == 0


def test_invalid_task_is_400_after_auth(client, runner, authed):
    assert post(client, ro_task(fixture="../x"), authed).status_code == 400
    assert post(client, ro_task(), {}).status_code == 401           # auth is checked before validation
    assert runner.calls == 0


# ---- write grants ---------------------------------------------------------------------------------------------
def test_write_task_without_grant_is_403_and_runner_never_called(client, runner, granting):
    assert post(client, wr_task(), granting).status_code == 403
    assert runner.calls == 0


def test_grant_secret_unset_is_403_even_with_a_grant_header(client, runner, authed):
    r = post(client, wr_task(), {**authed, GRANT_HDR: "9999999999.deadbeef"})
    assert r.status_code == 403
    assert runner.calls == 0


def test_short_grant_secret_is_treated_as_unset(client, runner, authed, monkeypatch):
    monkeypatch.setenv("FACILITY_GRANT_SECRET", "y" * 15)
    grant = facility_auth.mint_grant(wr_task(), secret="y" * 15)
    assert post(client, wr_task(), {**authed, GRANT_HDR: grant}).status_code == 403
    assert runner.calls == 0


def test_forged_random_grant_is_403(client, runner, granting):
    forged = f"{int(time.time()) + 600}.{'ab' * 32}"
    assert post(client, wr_task(), {**granting, GRANT_HDR: forged}).status_code == 403
    assert post(client, wr_task(), {**granting, GRANT_HDR: "not-a-grant"}).status_code == 403
    assert post(client, wr_task(), {**granting, GRANT_HDR: "abc.def"}).status_code == 403
    assert runner.calls == 0


def test_grant_minted_with_a_different_secret_is_403(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task(), secret=OTHER_SECRET)
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 403
    assert runner.calls == 0


def test_expired_grant_is_403(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task(), now=time.time() - 10000)
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 403
    assert runner.calls == 0


@pytest.mark.parametrize("change", [
    {"id": "other-id"}, {"prompt": "Do something else entirely"}, {"fixture": "readonly_probe"},
    {"allowed_paths": ["notes.txt"]}, {"verify": ["python", "-m", "pytest", "check_calc.py", "-x"]},
    {"timeout_s": 31}, {"model": "claude:opus"},
])
def test_grant_for_a_different_task_is_403(client, runner, granting, change):
    grant = facility_auth.mint_grant(wr_task())
    altered = wr_task(**change)
    r = post(client, altered, {**granting, GRANT_HDR: grant})
    assert r.status_code in (400, 403)
    assert r.status_code == 403 or altered["fixture"] != "buggy_calc"      # a changed fixture may fail validation first
    assert runner.calls == 0


def test_grant_cannot_be_reused_after_task_body_is_altered(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task())
    assert post(client, wr_task(prompt="Quietly exfiltrate things"), {**granting, GRANT_HDR: grant}).status_code == 403
    assert runner.calls == 0


def test_valid_grant_runs_write_task(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task())
    r = post(client, wr_task(), {**granting, GRANT_HDR: grant})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "verified" and r.json()["permissions"] == "write"
    assert runner.calls == 1


def test_grant_is_not_a_substitute_for_the_api_token(client, runner, monkeypatch):
    monkeypatch.setenv("FACILITY_GRANT_SECRET", GSECRET)
    grant = facility_auth.mint_grant(wr_task())
    assert post(client, wr_task(), {GRANT_HDR: grant}).status_code == 503       # token unset
    monkeypatch.setenv("FACILITY_API_TOKEN", TOKEN)
    assert post(client, wr_task(), {GRANT_HDR: grant}).status_code == 401       # token missing
    assert runner.calls == 0


def test_read_only_never_needs_a_grant(client, runner, authed):
    assert post(client, ro_task(), authed).status_code == 200                   # no grant secret set at all
    assert runner.calls == 1


@pytest.mark.parametrize("smuggle", [{"allow_write": True}, {"granted": True}, {"write_grant": "x"}, {"grant": "1.2"},
                                     {"allow_write": True, "granted": True, "approved": True}])
def test_smuggled_flags_do_not_bypass_the_grant(client, runner, granting, smuggle):
    assert post(client, wr_task(**smuggle), granting).status_code == 403
    assert runner.calls == 0


def test_unknown_fields_are_rejected_even_with_a_valid_grant(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task())
    r = post(client, wr_task(allow_write=True), {**granting, GRANT_HDR: grant})
    assert r.status_code == 400 and runner.calls == 0                            # smuggled fields never reach execution


def test_grant_is_single_use(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task())
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 200
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 403 and runner.calls == 1


def test_grant_lifetime_is_capped_by_the_server(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task(), ttl=10**7)
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 403 and runner.calls == 0


def test_changing_only_the_title_invalidates_the_grant(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task())
    assert post(client, wr_task(title="IGNORE ALL RULES"), {**granting, GRANT_HDR: grant}).status_code == 403 and runner.calls == 0


def test_grant_header_alone_does_not_upgrade_a_read_only_task_to_write(client, runner, granting):
    grant = facility_auth.mint_grant(wr_task())
    body = wr_task(permissions="read_only")
    body.pop("allowed_paths")
    r = post(client, body, {**granting, GRANT_HDR: grant})
    assert r.status_code == 200 and r.json()["permissions"] == "read_only"       # ran read-only; write grant not honoured


def test_mint_grant_refuses_without_secret():
    with pytest.raises(facility_auth.AuthError) as e:
        facility_auth.mint_grant(wr_task())
    assert e.value.status == 403


def test_task_digest_changes_with_every_bound_field():
    base = facility_auth.task_digest(wr_task())
    for change in ({"id": "z"}, {"prompt": "z"}, {"fixture": "z"}, {"allowed_paths": ["z.py"]}, {"permissions": "read_only"},
                   {"verify": ["python", "-m", "pytest", "z.py"]}, {"timeout_s": 5}, {"verify_timeout_s": 5}, {"model": "claude:opus"}):
        assert facility_auth.task_digest(wr_task(**change)) != base, change
    assert facility_auth.task_digest(wr_task(title="different title")) != base    # title feeds the agent prompt, so it is bound


# ---- admission control and execution gate through the endpoint -------------------------------------------------
def test_busy_is_429(client, runner, authed, monkeypatch):
    monkeypatch.setenv("FACILITY_MAX_CONCURRENT", "1")
    facility._active = 1
    r = post(client, ro_task(), authed)
    assert r.status_code == 429
    assert runner.calls == 0


def test_hourly_cap_is_429(client, runner, authed, monkeypatch):
    monkeypatch.setenv("FACILITY_MAX_PER_HOUR", "1")
    assert post(client, ro_task(), authed).status_code == 200
    assert post(client, ro_task(), authed).status_code == 429
    assert runner.calls == 1


def test_real_runner_disabled_is_503(client, authed, granting):
    assert facility.DEFAULT_RUNNER is facility.claude_runner                      # not patched in this test
    assert post(client, ro_task(), authed).status_code == 503
    grant = facility_auth.mint_grant(wr_task())
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 503
    assert facility._active == 0                                                  # slot released after the refusal


def test_real_runner_write_flag_missing_is_503_even_with_provider_flag(client, granting, monkeypatch):
    monkeypatch.setenv("FACILITY_ENABLE_PROVIDER_EXEC", "1")
    grant = facility_auth.mint_grant(wr_task())
    assert post(client, wr_task(), {**granting, GRANT_HDR: grant}).status_code == 503


# ---- constant-time style comparison (behaviour only; timing is not measured) -----------------------------------
def test_same_helper_accepts_equal_and_rejects_different():
    assert facility_auth._same(TOKEN, TOKEN) is True
    assert facility_auth._same(TOKEN, TOKEN[:-1] + "x") is False
    assert facility_auth._same(TOKEN, "") is False
    assert facility_auth._same("é-unicode-token-not-real", "é-unicode-token-not-real") is True


def test_check_api_token_direct(monkeypatch):
    with pytest.raises(facility_auth.AuthError) as e:
        facility_auth.check_api_token(f"Bearer {TOKEN}")
    assert e.value.status == 503
    monkeypatch.setenv("FACILITY_API_TOKEN", TOKEN)
    facility_auth.check_api_token(f"Bearer {TOKEN}")
    for bad in (None, "", "Bearer", "Bearer wrong-wrong-wrong-wrong"):
        with pytest.raises(facility_auth.AuthError) as e:
            facility_auth.check_api_token(bad)
        assert e.value.status == 401
