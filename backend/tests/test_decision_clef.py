"""Offline tests for the Clef decision helper. urlopen is blocked for every test unless a test mocks it."""
import io
import json
import urllib.error
import urllib.request

import pytest

from app import decision_clef as dc


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access attempted during an offline test")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sample_questions():
    return {
        "pick_fix": dc.build_question("choice", "Choose the best fix.", criteria={
            "patch_calc": "Edits calc.py only.", "rewrite_module": "Replaces the whole module."}),
        "is_safe": dc.build_question("noul", "Judge the change.", proposition="The change is safe.",
                                     true="Safe to merge after review.", false="Not safe."),
        "rate_quality": dc.build_question("score", "Rate the test coverage.", levels=["none", "partial", "full"]),
    }


# ---- request shape and transport ------------------------------------------------------------------

def test_request_goes_to_systemone_with_bearer_key_and_exact_model(monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")
    seen = {}

    def fake(req, timeout):
        seen["url"], seen["method"], seen["timeout"] = req.full_url, req.get_method(), timeout
        seen["auth"], seen["body"] = req.get_header("Authorization"), json.loads(req.data)
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    body = dc.build_request("clef", "diff: 2 files", sample_questions())
    assert dc.call_systemone(body) == {"ok": True}
    assert seen["url"] == "https://api.experientiallabs.ai/v1/systemone"
    assert seen["method"] == "POST"
    assert seen["timeout"] == dc.TIMEOUT_SECONDS and 0 < seen["timeout"] <= 60
    assert seen["auth"] == "Bearer test-key-not-real-0123456789"
    assert seen["body"]["model"] == "clef"
    assert set(seen["body"]) == {"model", "state", "questions"}


def test_payload_shape_for_each_question_type():
    body = dc.build_request("clef:free", {"diff": "x"}, sample_questions())
    assert body["model"] == "clef:free"                       # preserved exactly
    q = body["questions"]
    assert q["pick_fix"]["type"] == "choice" and q["pick_fix"]["instructions"] == "Choose the best fix."
    assert set(q["pick_fix"][dc.CHOICE_CRITERIA]) == {"patch_calc", "rewrite_module"}
    assert q["is_safe"]["type"] == "noul" and q["is_safe"][dc.NOUL_PROPOSITION] == "The change is safe."
    assert q["rate_quality"][dc.SCORE_LEVELS] == ["none", "partial", "full"]


def test_state_may_be_text_object_or_array():
    for state in ("text", {"k": "v"}, ["a", "b"]):
        assert dc.build_request("clef", state, sample_questions())["state"] == state
    with pytest.raises(ValueError):
        dc.build_request("clef", 42, sample_questions())


def test_model_spelling_is_exact():
    assert dc.build_request("clef", "s", sample_questions())["model"] == "clef"
    for bad in ("Clef", "clef-free", "clef:free ", "cloudflare/clef", "gpt-6-luna-decisions"):
        with pytest.raises(ValueError):
            dc.build_request(bad, "s", sample_questions())


# ---- missing key: no network ----------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_stops_before_any_request(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(dc.KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(dc.KEY_VAR, value)
    body = dc.build_request("clef", "s", sample_questions())
    with pytest.raises(dc.DecisionUnavailable, match="EXPERIENTIAL_API_KEY is not set"):
        dc.call_systemone(body)


# ---- error handling -------------------------------------------------------------------------------

def test_http_error_is_surfaced_once_without_retry_and_key_is_redacted(monkeypatch):
    key = "test-key-not-real-ABCDEFGHIJ"
    monkeypatch.setenv(dc.KEY_VAR, key)
    calls = []

    def fake(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 502, "bad gateway", {},
                                     io.BytesIO(f"all_routes_failed, echoed {key}".encode()))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(dc.DecisionError, match="HTTP 502") as info:
        dc.call_systemone(dc.build_request("clef", "s", sample_questions()))
    assert key not in str(info.value) and "<redacted>" in str(info.value)
    assert len(calls) == 1                                   # no automatic retries


def test_transport_error_is_surfaced_without_retry(monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")
    calls = []

    def fake(req, timeout):
        calls.append(1)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(dc.DecisionError, match="transport error"):
        dc.call_systemone(dc.build_request("clef", "s", sample_questions()))
    assert len(calls) == 1


def test_non_json_success_body_is_an_error(monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: FakeResponse(b"<html>oops</html>"))
    with pytest.raises(dc.DecisionError, match="not JSON"):
        dc.call_systemone(dc.build_request("clef", "s", sample_questions()))


def test_response_parsing_is_not_implemented_until_docs_are_verified():
    with pytest.raises(NotImplementedError):
        dc.parse_response({})


# ---- admission bounds (shared 65,536-token ceiling, 32 questions, 64 options, 2-10 levels) --------

def test_at_most_32_questions():
    many = {f"q{i}": dc.build_question("score", "rate", levels=["a", "b"]) for i in range(32)}
    assert len(dc.build_request("clef", "s", many)["questions"]) == 32
    many["q32"] = dc.build_question("score", "rate", levels=["a", "b"])
    with pytest.raises(ValueError, match="at most 32"):
        dc.build_request("clef", "s", many)


def test_at_most_64_options_per_choice():
    ok = {f"o{i}": "d" for i in range(64)}
    assert dc.build_question("choice", "pick", criteria=ok)[dc.CHOICE_CRITERIA]
    with pytest.raises(ValueError, match="1-64"):
        dc.build_question("choice", "pick", criteria={f"o{i}": "d" for i in range(65)})


def test_score_levels_between_2_and_10():
    dc.build_question("score", "rate", levels=["a", "b"])
    dc.build_question("score", "rate", levels=[str(i) for i in range(10)])
    for bad in (["only"], [str(i) for i in range(11)]):
        with pytest.raises(ValueError, match="2-10"):
            dc.build_question("score", "rate", levels=bad)


def test_shared_token_ceiling_is_enforced():
    # 30 questions (under the 32 cap) of ~7,000 characters each: about 70k estimated tokens, over the ceiling.
    huge = {f"q{i}": dc.build_question("choice", "x" * 7000, criteria={"a": "y"}) for i in range(30)}
    with pytest.raises(ValueError, match="ceiling"):
        dc.build_request("clef", "s", huge)


def test_questions_are_independent_objects_with_no_cross_references():
    body = dc.build_request("clef", "s", sample_questions())
    flat = json.dumps(body["questions"])
    for qid in body["questions"]:
        assert flat.count(f'"{qid}"') == 1      # IDs appear only as their own keys, never inside another question


def test_question_type_and_instruction_validation():
    with pytest.raises(ValueError):
        dc.build_question("essay", "x")
    with pytest.raises(ValueError):
        dc.build_question("score", "   ", levels=["a", "b"])


def test_state_summary_is_bounded_and_has_both_sections():
    text = dc.summarize_state("3 files changed", "42 passed")
    assert "DIFF SUMMARY" in text and "TEST SUMMARY" in text and "42 passed" in text
    assert len(dc.summarize_state("x" * 5000, "y", max_chars=300)) <= 300
