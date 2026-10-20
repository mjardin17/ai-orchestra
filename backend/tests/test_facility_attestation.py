"""Signed verification claims on facility receipts. Offline: stub runners only, no model calls."""
import hashlib
import hmac

import pytest

from app import facility, facility_attestation as fa, store
from test_facility_verifier import BAD_CALC, CHECK, GOOD_CALC, go, make_fixture, runner_writing, task    # noqa: F401

SECRET = "attest-secret-0123456789abcdef0123"
# Cross-repo vector: empire-command-center tests/missionEvidence.test.ts asserts the same digest.
VECTOR = {"run_id": "run_vector_01", "task_id": "t-read", "status": "verified",
          "verification_trust": "pristine_code_only", "verify": {"report_ok": True}}
VECTOR_DIGEST = "0c257580aa824dcdb280bc39fc207588ef4538bc7bde9316121746218144b772"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("FACILITY_RECEIPT_DIR", str(tmp_path / "receipts"))
    store.DB.execute("delete from lessons"); store.DB.commit()


def test_canonical_message_matches_the_shared_format():
    assert fa.canonical_message(VECTOR) == "v1|run_vector_01|t-read|verified|true|pristine_code_only"
    assert fa.attest(VECTOR, "vector-secret-0123456789abcdef0123") == VECTOR_DIGEST


def test_report_ok_is_lowercase_so_it_matches_the_js_verifier():
    failed = dict(VECTOR, verify={"report_ok": False})
    assert fa.canonical_message(failed).endswith("|false|pristine_code_only")


def test_no_attestation_without_a_strong_secret(monkeypatch):
    monkeypatch.delenv("FACILITY_EVIDENCE_SECRET", raising=False)
    assert fa.attestation_for(VECTOR) is None
    monkeypatch.setenv("FACILITY_EVIDENCE_SECRET", "too-short")
    assert fa.attestation_for(VECTOR) is None


def test_attestation_changes_when_trust_changes(monkeypatch):
    monkeypatch.setenv("FACILITY_EVIDENCE_SECRET", SECRET)
    same_user = dict(VECTOR, verification_trust="agent_code_executed_same_user")
    assert fa.attestation_for(VECTOR) != fa.attestation_for(same_user)


def test_read_only_receipt_carries_a_verifiable_attestation(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_EVIDENCE_SECRET", SECRET)
    make_fixture(tmp_path, monkeypatch, {"calc.py": GOOD_CALC, "check_calc.py": CHECK})

    async def noop(ws, prompt, write, model, timeout):
        return {"text": "", "exit_code": 0}

    r = go(facility.run_task(task(fixture="custom", permissions="read_only", allowed_paths=[]), runner=noop))
    assert r["status"] == "verified", r["verify"]
    assert r["verification_trust"] == "pristine_code_only"
    expected = hmac.new(SECRET.encode(), fa.canonical_message(r).encode(), hashlib.sha256).hexdigest()
    assert r["attestation"] == expected


def test_same_user_write_receipt_is_attested_with_its_real_trust_level(tmp_path, monkeypatch):
    monkeypatch.setenv("FACILITY_EVIDENCE_SECRET", SECRET)
    make_fixture(tmp_path, monkeypatch, {"calc.py": BAD_CALC, "check_calc.py": CHECK})
    r = go(facility.run_task(task(fixture="custom"), runner=runner_writing({"calc.py": GOOD_CALC})))
    assert r["status"] == "verified", r["verify"]
    # The signature covers the trust level, so a consumer can refuse this receipt; it is never reported as pristine.
    assert r["verification_trust"] == "agent_code_executed_same_user"
    assert "|agent_code_executed_same_user" in fa.canonical_message(r)
