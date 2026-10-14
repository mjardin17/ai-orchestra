"""Offline tests for the Clef decision helper and its route. urlopen is blocked for every test that doesn't mock it."""
import io
import json
import urllib.error
import urllib.request

import pytest
from fastapi.testclient import TestClient

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
        "pick_fix": {"type": "choice", "instructions": "Choose the best fix.",
                     "criteria": {"patch_calc": "Edits calc.py only.", "rewrite_module": "Replaces the module."}},
        "is_safe": {"type": "noul", "instructions": "Judge: the change is safe.",
                    "criteria": {"true": "Safe to merge after review.", "false": "Not safe."}},
        "rate_quality": {"type": "score", "instructions": "Rate the test coverage.",
                         "criteria": ["none", "partial", "full"]},
    }


# ---- request schema and transport --------------------------------------------------------------

def test_request_is_one_bearer_post_to_systemone_with_exact_model(monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")
    seen = {}

    def fake(req, timeout):
        seen.update(url=req.full_url, method=req.get_method(), timeout=timeout,
                    auth=req.get_header("Authorization"), body=json.loads(req.data))
        return FakeResponse(b'{"answers": {}, "usage": null}')

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    dc.call_systemone(dc.build_request("clef", "diff: 2 files", sample_questions()))
    assert seen["url"] == "https://api.experientiallabs.ai/v1/systemone"
    assert seen["method"] == "POST"
    assert 0 < seen["timeout"] <= 60
    assert seen["auth"] == "Bearer test-key-not-real-0123456789"
    assert set(seen["body"]) == {"model", "state", "questions"}
    assert seen["body"]["model"] == "clef"


def test_clef_free_is_preserved_when_selected():
    assert dc.build_request("clef:free", "s", sample_questions())["model"] == "clef:free"


def test_model_spelling_is_exact():
    for bad in ("Clef", "clef-free", "clef:free ", "cloudflare/clef", "gpt-6-luna-decisions"):
        with pytest.raises(ValueError):
            dc.build_request(bad, "s", sample_questions())


def test_state_may_be_text_object_or_array():
    for state in ("text", {"k": "v"}, ["a", "b"]):
        assert dc.build_request("clef", state, sample_questions())["state"] == state
    with pytest.raises(ValueError):
        dc.build_request("clef", 42, sample_questions())


def test_payload_shape_per_question_type():
    q = dc.build_request("clef", "s", sample_questions())["questions"]
    assert q["pick_fix"] == sample_questions()["pick_fix"]
    assert q["is_safe"]["criteria"] == {"true": "Safe to merge after review.", "false": "Not safe."}
    assert q["rate_quality"]["criteria"] == ["none", "partial", "full"]


@pytest.mark.parametrize("bad, reason", [
    ({"type": "essay", "instructions": "x"}, "question type"),
    ({"type": "score", "instructions": "   ", "criteria": ["a", "b"]}, "instructions"),
    ({"type": "choice", "instructions": "x"}, "criteria"),
    ({"type": "choice", "instructions": "x", "criteria": {"a": ""}}, "description"),
    ({"type": "noul", "instructions": "x", "criteria": {"maybe": "d"}}, "true"),
    ({"type": "noul", "instructions": "x", "criteria": {}}, "true"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "2-10"),
    ({"type": "score", "instructions": "x", "criteria": "not a list"}, "2-10"),
])
def test_invalid_question_shapes_are_refused(bad, reason):
    with pytest.raises(ValueError, match=reason):
        dc.build_request("clef", "s", {"q": bad})


def test_noul_criteria_is_optional():
    dc.build_request("clef", "s", {"q": {"type": "noul", "instructions": "x"}})


# ---- admission bounds -------------------------------------------------------------------------

def test_at_most_32_questions():
    many = {f"q{i}": {"type": "score", "instructions": "r", "criteria": ["a", "b"]} for i in range(32)}
    assert len(dc.build_request("clef", "s", many)["questions"]) == 32
    many["q32"] = {"type": "score", "instructions": "r", "criteria": ["a", "b"]}
    with pytest.raises(ValueError, match="at most 32"):
        dc.build_request("clef", "s", many)


def test_at_most_64_options_per_choice():
    ok = {"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(64)}}}
    dc.build_request("clef", "s", ok)
    bad = {"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(65)}}}
    with pytest.raises(ValueError, match="1-64"):
        dc.build_request("clef", "s", bad)


def test_score_levels_between_2_and_10():
    dc.build_request("clef", "s", {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(10)]}})
    with pytest.raises(ValueError, match="2-10"):
        dc.build_request("clef", "s", {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}})


def test_local_size_precheck_refuses_oversized_requests():
    # 30 questions (under the 32 cap) of ~7,000 characters: the conservative estimate exceeds the ceiling.
    huge = {f"q{i}": {"type": "choice", "instructions": "x" * 7000, "criteria": {"a": "y"}} for i in range(30)}
    with pytest.raises(ValueError, match="pre-check"):
        dc.build_request("clef", "s", huge)


def test_state_summary_is_bounded_and_has_both_sections():
    text = dc.summarize_state("3 files changed", "42 passed")
    assert "DIFF SUMMARY" in text and "TEST SUMMARY" in text and "42 passed" in text
    assert len(dc.summarize_state("x" * 5000, "y", max_chars=300)) <= 300


# ---- missing key: no request ------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_stops_before_any_request(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(dc.KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(dc.KEY_VAR, value)
    with pytest.raises(dc.DecisionUnavailable, match="EXPERIENTIAL_API_KEY is not set"):
        dc.call_systemone(dc.build_request("clef", "s", sample_questions()))


# ---- errors: surfaced once, no retry, key redacted --------------------------------------------

def test_http_error_is_surfaced_once_and_key_is_redacted(monkeypatch):
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
    assert len(calls) == 1


def test_transport_error_is_surfaced_once(monkeypatch):
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


# ---- response parsing and usage display -------------------------------------------------------

def test_parse_reads_answers_by_id_and_reports_missing_and_unexpected():
    response = {"answers": {"pick_fix": {"choice": "patch_calc"}, "stray": "x"}, "usage": {"input_tokens": 120}}
    parsed = dc.parse_response(response, ["pick_fix", "is_safe"])
    assert parsed["answers"] == {"pick_fix": {"choice": "patch_calc"}}
    assert parsed["missing"] == ["is_safe"]
    assert parsed["unexpected"] == ["stray"]
    assert parsed["usage"] == {"input_tokens": 120}


def test_answer_shapes_are_passed_through_not_interpreted():
    value = {"any": ["shape", 1, None]}
    assert dc.parse_response({"answers": {"q": value}}, ["q"])["answers"]["q"] == value


def test_response_without_answers_object_is_an_error():
    for bad in ({}, {"answers": ["list"]}, [], "text"):
        with pytest.raises(dc.DecisionError):
            dc.parse_response(bad, ["q"])


def test_report_shows_answers_usage_missing_and_disclaimer():
    parsed = dc.parse_response({"answers": {"pick_fix": "patch_calc"}, "usage": {"total_tokens": 99}}, ["pick_fix", "is_safe"])
    report = dc.format_report(parsed)
    assert "pick_fix: \"patch_calc\"" in report
    assert "NO ANSWER RETURNED FOR: is_safe" in report
    assert 'USAGE (as reported): {"total_tokens": 99}' in report
    assert dc.DISCLAIMER in report


def test_report_says_when_usage_is_not_reported():
    report = dc.format_report(dc.parse_response({"answers": {}}, []))
    assert "USAGE (as reported): not reported" in report


# ---- human-invoked route ----------------------------------------------------------------------

@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


ROUTE_BODY = {"model": "clef", "state": "diff: 2 files", "questions": sample_questions()}


def test_route_refuses_without_explicit_confirm(client, monkeypatch):
    monkeypatch.setattr(dc, "call_systemone", lambda body, timeout=0: (_ for _ in ()).throw(AssertionError("called")))
    res = client.post("/api/decision/clef", json=ROUTE_BODY)
    assert res.status_code == 400 and "confirm=true" in res.json()["detail"]


def test_route_rejects_invalid_questions_before_any_request(client, monkeypatch):
    monkeypatch.setattr(dc, "call_systemone", lambda body, timeout=0: (_ for _ in ()).throw(AssertionError("called")))
    res = client.post("/api/decision/clef", json={**ROUTE_BODY, "confirm": True, "questions": {"q": {"type": "essay"}}})
    assert res.status_code == 400


def test_route_without_key_returns_503_and_makes_no_request(client, monkeypatch):
    monkeypatch.delenv(dc.KEY_VAR, raising=False)
    res = client.post("/api/decision/clef", json={**ROUTE_BODY, "confirm": True})
    assert res.status_code == 503 and "EXPERIENTIAL_API_KEY is not set" in res.json()["detail"]


def test_route_maps_http_errors_to_502(client, monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")

    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b"upstream failure"))

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    res = client.post("/api/decision/clef", json={**ROUTE_BODY, "confirm": True})
    assert res.status_code == 502 and "HTTP 500" in res.json()["detail"]


def test_route_returns_answers_usage_and_report_with_clef_free(client, monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")
    seen = {}

    def fake(req, timeout):
        seen["model"] = json.loads(req.data)["model"]
        return FakeResponse(json.dumps({
            "answers": {"pick_fix": "patch_calc", "is_safe": "true", "rate_quality": "partial"},
            "usage": {"input_tokens": 300, "output_tokens": 40}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    res = client.post("/api/decision/clef", json={**ROUTE_BODY, "model": "clef:free", "confirm": True})
    assert res.status_code == 200
    body = res.json()
    assert seen["model"] == "clef:free" and body["model"] == "clef:free"
    assert body["answers"] == {"pick_fix": "patch_calc", "is_safe": "true", "rate_quality": "partial"}
    assert body["usage"] == {"input_tokens": 300, "output_tokens": 40}
    assert body["missing"] == [] and body["unexpected"] == []
    assert "USAGE (as reported)" in body["report"] and body["disclaimer"] == dc.DISCLAIMER


def test_route_returns_502_when_response_has_no_answers(client, monkeypatch):
    monkeypatch.setenv(dc.KEY_VAR, "test-key-not-real-0123456789")
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: FakeResponse(b'{"error": "x"}'))
    res = client.post("/api/decision/clef", json={**ROUTE_BODY, "confirm": True})
    assert res.status_code == 502 and "answers" in res.json()["detail"]
