"""Provider layer: explicit errors, no substitution, no truncation, configurable limits, capability/support rules."""
import asyncio, io, json, socket, urllib.error
import pytest
from fastapi.testclient import TestClient
from app import claude_client, free_models, models, providers, store
from app.main import app
from app.providers import RunSpec

c = TestClient(app)
GROQ = {"id": "groq:test-model", "label": "Test", "kind": "free", "provider": "groq", "model": "test-model", "power": 3}
OPENAI = {"id": "openai:gpt-test", "label": "OA", "kind": "free", "provider": "openai", "model": "gpt-test", "power": 3}
XAI = {"id": "xai:grok-test", "label": "X", "kind": "free", "provider": "xai", "model": "grok-test", "power": 3}


def go(m, spec=None):
    return asyncio.run(providers.run(m, spec or RunSpec("sys", "hello")))


def http_error(code, body=""):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    free_models.COOLDOWN.clear()
    monkeypatch.setattr(free_models, "has_key", lambda p: True)
    monkeypatch.setattr(free_models, "get_key", lambda p: "test-key")
    store.DB.execute("delete from provider_health")
    store.DB.commit()


def ok_reply(**over):
    base = {"choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}, "model": "groq-model-2026-01"}
    return {**base, **over}


# ---- success shape ----------------------------------------------------------------------------------------------
def test_success_carries_text_usage_version_latency(monkeypatch):
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: ok_reply())
    r = go(GROQ)
    assert r.ok and r.text == "hi there" and r.error is None
    assert r.usage["total_tokens"] == 7 and r.model_version == "groq-model-2026-01" and r.latency >= 0
    assert r.tool_calls == []          # nothing is fabricated when the provider reports none


def test_tool_calls_appear_only_if_the_provider_returned_them(monkeypatch):
    call = {"id": "t1", "function": {"name": "lookup", "arguments": "{}"}}
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: ok_reply(choices=[{"message": {"content": "x", "tool_calls": [call]}}]))
    assert go(GROQ).tool_calls == [call]


# ---- explicit failures ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("exc,kind", [
    (lambda: http_error(400, '{"error":{"message":"This model\'s maximum context length is 8192 tokens"}}'), "context_exceeded"),
    (lambda: http_error(413, "too big"), "context_exceeded"),
    (lambda: http_error(429, "slow down"), "rate_limited"),
    (lambda: http_error(404, "nope"), "model_not_found"),
    (lambda: http_error(401, "bad key"), "provider_error"),
    (lambda: http_error(500, "boom"), "provider_error"),
    (lambda: socket.timeout("timed out"), "timeout"),
    (lambda: urllib.error.URLError(socket.timeout("timed out")), "timeout"),
    (lambda: urllib.error.URLError("dns failure"), "provider_error"),
])
def test_failures_are_explicit_kinds(monkeypatch, exc, kind):
    def boom(*a, **k):
        raise exc()
    monkeypatch.setattr(free_models, "_post", boom)
    r = go(GROQ)
    assert not r.ok and r.error["kind"] == kind and r.error["message"] and r.text == ""
    assert kind in providers.ERROR_KINDS


def test_empty_reply_is_a_provider_error_not_success(monkeypatch):
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: ok_reply(choices=[{"message": {"content": ""}}]))
    assert go(GROQ).error["kind"] == "provider_error"


def test_not_configured_makes_no_network_call(monkeypatch):
    monkeypatch.setattr(free_models, "has_key", lambda p: False)
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: pytest.fail("network call made without a key"))
    r = go(XAI)
    assert r.error["kind"] == "not_configured" and "XAI_API_KEY" in r.error["message"]


def test_oversized_prompt_is_refused_not_truncated(monkeypatch):
    sent = []
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: sent.append(1) or ok_reply())
    huge = "x" * 4_000_000          # ~1M tokens, over groq's declared 131072
    r = go(GROQ, RunSpec("sys", huge))
    assert r.error["kind"] == "context_exceeded" and not sent
    assert "never truncated" in r.error["message"]


def test_no_silent_substitution_on_failure(monkeypatch):
    seen = []

    def boom(url, key, body, timeout=None):
        seen.append(body["model"])
        raise http_error(500, "down")
    monkeypatch.setattr(free_models, "_post", boom)
    r = go(GROQ)
    assert seen == ["test-model"] and r.model == "test-model" and not r.ok      # one attempt, same model, error reported


def test_rate_limit_sets_cooldown_and_is_reported(monkeypatch):
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: (_ for _ in ()).throw(http_error(429, "x")))
    assert go(GROQ).error["kind"] == "rate_limited"
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: pytest.fail("called during cooldown"))
    assert go(GROQ).error["kind"] == "rate_limited"


# ---- configurable limits ----------------------------------------------------------------------------------------
def test_output_limit_and_timeout_are_configurable(monkeypatch):
    got = {}

    def capture(url, key, body, timeout=None):
        got.update(body=body, timeout=timeout)
        return ok_reply()
    monkeypatch.setattr(free_models, "_post", capture)
    go(GROQ, RunSpec("s", "p", max_tokens=9000, timeout=123))
    assert got["body"]["max_tokens"] == 9000 and got["timeout"] == 123
    go(GROQ)
    assert got["body"]["max_tokens"] == free_models.DEFAULT_MAX_TOKENS == 2000      # original default preserved
    go(OPENAI, RunSpec("s", "p", max_tokens=5000))
    assert got["body"]["max_completion_tokens"] == 5000 and "max_tokens" not in got["body"]    # newer OpenAI models


def test_original_call_one_still_works(monkeypatch):
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: ok_reply())
    assert asyncio.run(free_models.call_one("groq", "test-model", "s", "p")) == "hi there"


def test_http_error_classifier():
    k = free_models.classify_http_error
    assert k("p", 400, "Request too large for model")[0] == "context_exceeded"
    assert k("p", 503, "")[0] == "rate_limited"
    assert k("p", 400, "The model `x` does not exist")[0] == "model_not_found"


# ---- Claude adapter ---------------------------------------------------------------------------------------------
CLAUDE_MAIN = {"id": "claude:sonnet", "label": "Claude Sonnet", "kind": "claude", "tier": "main", "power": 4}
CLAUDE_TOP = {"id": "claude:opus", "label": "Claude Opus", "kind": "claude", "tier": "top", "power": 5}


def test_claude_success_reports_requested_alias_as_version(monkeypatch):
    async def fake(system, prompt, model, tools=False, timeout=None):
        return "answer"
    monkeypatch.setattr(claude_client, "_call", fake)
    r = go(CLAUDE_MAIN)
    assert r.ok and r.text == "answer" and r.model_version == claude_client.MODELS["main"] and r.tool_calls == [] and r.usage == {}


def test_claude_above_max_tier_fails_instead_of_stepping_down(monkeypatch):
    monkeypatch.setattr(claude_client, "MAX_TIER", "main")

    async def never(*a, **k):
        pytest.fail("a different model was called")
    monkeypatch.setattr(claude_client, "_call", never)
    r = go(CLAUDE_TOP)
    assert r.error["kind"] == "provider_error" and "not substituting" in r.error["message"]


@pytest.mark.parametrize("exc,kind", [
    (asyncio.TimeoutError(), "timeout"), (FileNotFoundError(), "not_configured"),
    (RuntimeError("Prompt is too long"), "context_exceeded"), (RuntimeError("usage limit reached"), "rate_limited"),
    (RuntimeError("something else"), "provider_error")])
def test_claude_errors_map_to_kinds(monkeypatch, exc, kind):
    async def fake(*a, **k):
        raise exc
    monkeypatch.setattr(claude_client, "_call", fake)
    assert go(CLAUDE_MAIN).error["kind"] == kind


# ---- capability matrix and support rule -------------------------------------------------------------------------
def test_every_known_provider_declares_all_four_capabilities():
    for name, caps in providers.PROVIDER_CAPS.items():
        assert set(caps) == {"tools_native", "tools_traceable", "max_context", "structured_output"}, name
    for prov in ("claude", "openai", "gemini", "xai"):
        assert providers.PROVIDER_CAPS[prov]["max_context"]
    assert all(v["tools_traceable"] is False for v in providers.PROVIDER_CAPS.values())     # no tool harness yet


def test_unknown_provider_or_context_means_capabilities_not_known():
    assert providers.capabilities({"id": "weird:x", "kind": "free", "provider": "weird"})["known"] is False
    assert providers.capabilities({"id": "codex:default", "kind": "codex"})["known"] is False     # context size unknown
    assert providers.capabilities(GROQ)["known"] is True


def test_capability_override_setting():
    store.set_setting("capabilities", json.dumps({"codex:default": {"max_context": 400000}}))
    try:
        assert providers.capabilities({"id": "codex:default", "kind": "codex"})["known"] is True
    finally:
        store.set_setting("capabilities", "{}")


def test_supported_requires_successful_health_and_known_capabilities(monkeypatch):
    assert providers.support_status(GROQ)["supported"] is False                  # never checked
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: ok_reply())
    h = asyncio.run(providers.health_check(GROQ))
    assert h["ok"] and providers.support_status(GROQ)["supported"] is True
    weird = {"id": "weird:x", "kind": "free", "provider": "weird", "model": "x"}
    providers._save_health({"model_id": "weird:x", "provider": "weird", "ok": True, "kind": None, "message": "", "latency": 0,
                            "model_version": None, "key_source": None, "checked_at": 0})
    s = providers.support_status(weird)
    assert s["supported"] is False and "capabilities not explicitly known" in s["reasons"]


def test_failed_health_check_means_unsupported(monkeypatch):
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: (_ for _ in ()).throw(http_error(401, "bad key")))
    h = asyncio.run(providers.health_check(GROQ))
    assert not h["ok"] and h["kind"] == "provider_error"
    s = providers.support_status(GROQ)
    assert not s["supported"] and "provider_error" in s["reasons"][0]


def test_health_check_for_unconfigured_provider_is_recorded_without_network(monkeypatch):
    monkeypatch.setattr(free_models, "has_key", lambda p: False)
    monkeypatch.setattr(free_models, "_post", lambda *a, **k: pytest.fail("network call for an unconfigured provider"))
    h = asyncio.run(providers.health_check(XAI))
    assert h["kind"] == "not_configured" and not h["ok"]
    assert providers.last_health(XAI["id"])["kind"] == "not_configured"


# ---- credentials ------------------------------------------------------------------------------------------------
def test_gemini_reads_google_key_in_place_and_never_exposes_it(monkeypatch):
    monkeypatch.undo()                      # use the real get_key/has_key for this test
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-value")
    assert free_models.get_key("gemini") == "super-secret-value"
    assert free_models.key_source("gemini") == "GOOGLE_API_KEY"
    assert "super-secret-value" not in json.dumps(c.get("/api/providers").json())
    monkeypatch.setenv("GEMINI_API_KEY", "primary")
    assert free_models.get_key("gemini") == "primary" and free_models.key_source("gemini") == "GEMINI_API_KEY"
    import os
    assert os.environ["GOOGLE_API_KEY"] == "super-secret-value"      # not renamed, copied or modified


def test_catalog_lists_openai_and_xai_models():
    ids = {m["id"] for m in models.catalog()}
    assert "openai:gpt-5" in ids and "xai:grok-4" in ids


# ---- endpoints --------------------------------------------------------------------------------------------------
def test_providers_endpoint_shape():
    rows = c.get("/api/providers").json()
    row = next(r for r in rows if r["id"] == "xai:grok-4")
    assert {"configured", "capabilities", "health", "supported", "reasons", "key_source"} <= set(row)
    assert row["supported"] is False


def test_health_endpoint_unconfigured_provider(monkeypatch):
    monkeypatch.setattr(free_models, "has_key", lambda p: False)
    out = c.post("/api/providers/health", json={"model_ids": ["xai:grok-4"]}).json()
    assert out[0]["kind"] == "not_configured"
    assert c.post("/api/providers/health", json={"model_ids": ["nope:x"]}).status_code == 404


def test_claude_cli_path_prefers_claude_bin(monkeypatch, tmp_path):
    exe = tmp_path / "claude.exe"
    exe.write_text("x")
    monkeypatch.setenv("CLAUDE_BIN", str(exe))
    assert claude_client.cli_path() == str(exe)
    monkeypatch.delenv("CLAUDE_BIN")
    assert claude_client.cli_path().lower().endswith(("claude", "claude.cmd", "claude.exe")) or claude_client.cli_path() == "claude"


def test_stale_cli_shim_message_is_reported_as_not_configured():
    err = providers._cli_error("'\"C:\\x\\claude.exe\"' is not recognized as an internal or external command")
    assert err.kind == "not_configured"
