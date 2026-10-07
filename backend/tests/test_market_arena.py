import asyncio, json, os
from pathlib import Path
import pytest
from app import arena, judge, market_data, models, providers

REAL = Path(r"C:\Users\jjard\Downloads")
COUNTS = {"books": 887, "merch": 902, "merch_counts": 102, "trends": 6}
needs_real = pytest.mark.skipif(not (REAL / "dataset1_books.csv").exists(), reason="real datasets not present")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A throwaway dir with a multi-line cell, to prove row IDs follow physical CSV lines."""
    (tmp_path / "dataset1_books.csv").write_text('title,author\nA,x\n"B\nsplit",y\nC,z\n', encoding="utf-8")
    (tmp_path / "dataset2_merch.csv").write_text("niche\nn1\n", encoding="utf-8")
    (tmp_path / "merch_listing_counts.csv").write_text("query\nq1\n", encoding="utf-8")
    (tmp_path / "book_trends.csv").write_text("term\nt1\n", encoding="utf-8")
    monkeypatch.setenv("MARKET_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(market_data, "EVIDENCE_COLS", {"books": ["title", "author"], "merch": ["niche"], "merch_counts": ["query"], "trends": ["term"]})
    return tmp_path


def test_data_dir_comes_from_env(tiny):
    assert market_data.data_dir() == tiny


def test_missing_dir_setting_raises(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_DIR", raising=False)
    monkeypatch.setattr(market_data, "ENV_FILE", Path("nope.env"))
    with pytest.raises(RuntimeError):
        market_data.data_dir()


@needs_real
def test_env_file_points_at_downloads(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_DIR", raising=False)
    assert market_data.data_dir() == REAL


@needs_real
def test_real_record_counts(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_DIR", str(REAL))
    assert {k: len(market_data.read_rows(k)) for k in COUNTS} == COUNTS
    assert market_data.read_rows("books")[0]["_row"] == "dataset1_books.csv:2"
    assert market_data.read_rows("merch")[0]["_row"] == "dataset2_merch.csv:2"


def test_row_ids_track_physical_lines_even_with_multiline_cells(tiny):
    assert [r["_row"] for r in market_data.read_rows("books")] == ["dataset1_books.csv:2", "dataset1_books.csv:3", "dataset1_books.csv:5"]


def test_digest_counts_and_range(tiny):
    d = market_data.digest()["files"]["books"]
    assert d["records"] == 3 and d["first_row"] == "dataset1_books.csv:2" and d["last_row"] == "dataset1_books.csv:5"


def test_evidence_is_deterministic_and_traceable(tiny):
    a, b = market_data.build_evidence(), market_data.build_evidence()
    assert a["text"] == b["text"] and a["sha256"] == b["sha256"]
    assert f"SOURCE DIRECTORY: {tiny}" in a["text"]
    assert "dataset1_books.csv:5\tC\tz" in a["text"] and "B split" in a["text"]
    assert "dataset2_merch.csv:2\tn1" in a["text"]


def test_evidence_refuses_to_truncate(tiny, monkeypatch):
    monkeypatch.setattr(market_data, "MAX_EVIDENCE_CHARS", 50)
    with pytest.raises(ValueError):
        market_data.build_evidence()


@needs_real
def test_real_evidence_has_every_row(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_DIR", str(REAL))
    ev = market_data.build_evidence()
    for key, n in COUNTS.items():
        name = market_data.FILES[key]
        assert sum(1 for l in ev["text"].splitlines() if l.startswith(name + ":")) == n
    assert "dataset1_books.csv:888\t" in ev["text"] and "dataset2_merch.csv:903\t" in ev["text"]
    assert "sample" not in ev["text"].split("\n")[1].lower()


def test_prompt_construction():
    assert arena.build_prompt("TASK", "EVID") == "TASK\n\nEVID"
    jp = judge.build_prompt("TASK", "EVID", [{"label": 1, "text": "OPTS"}])
    assert "TASK" in jp and "EVID" in jp and jp.endswith("Submission 1:\nOPTS")


def _fake_pool(monkeypatch, calls, judge_calls=None):
    """Three fake contestants; providers.run is replaced so no model is ever called. Judge calls (recognised by their
    system prompt) are recorded separately and answered with a valid anonymous verdict."""
    pool = [{"id": f"m{i}", "label": f"M{i}", "kind": "free", "provider": f"p{i}", "model": f"x{i}", "enabled": True, "power": 3}
            for i in range(3)]
    monkeypatch.setattr(models, "enabled_pool", lambda: pool)
    verdict = json.dumps({"scores": {str(i): {c: 4 for c in judge.CRITERIA} for i in (1, 2, 3)}, "ranking": [2, 1, 3], "reasoning": {}})

    async def fake(m, spec):
        if spec.system == judge.JUDGE_SYSTEM:
            (judge_calls if judge_calls is not None else []).append(spec.prompt)
            return providers.RunResult(provider="p", model="j", ok=True, text=verdict)
        calls.append((m["id"], spec.system, spec.prompt))
        return providers.RunResult(provider="p", model=m["id"], ok=True, text=f"answer-{m['id']}", model_version="v1")
    monkeypatch.setattr(providers, "run", fake)


def _run(market):
    async def go():
        return [ev async for ev in arena.run("Which niches are rising?", ["m0", "m1", "m2"], market)]
    return asyncio.run(go())


def test_every_model_and_judge_get_identical_evidence(tiny, monkeypatch):
    calls, judge_calls = [], []
    _fake_pool(monkeypatch, calls, judge_calls)
    monkeypatch.setattr(arena.store, "save_arena", lambda *a, **k: None)
    events = _run(True)
    ev = market_data.build_evidence()
    assert len(calls) == 3
    assert len({(s, p) for _, s, p in calls}) == 1                 # identical system + prompt for all models
    assert calls[0][2] == arena.build_prompt("Which niches are rising?", ev["text"])
    assert ev["text"] in judge_calls[0]                             # judge sees the same evidence
    assert "answer-m0" in judge_calls[0] and "answer-m0" not in calls[0][2]   # models never see each other's answers
    assert any(e["type"] == "evidence" and e["sha256"] == ev["sha256"] for e in events)
    assert "dataset1_books.csv:2" in calls[0][2]


def test_market_off_keeps_original_behaviour(monkeypatch):
    calls = []
    _fake_pool(monkeypatch, calls)
    monkeypatch.setattr(arena.store, "save_arena", lambda *a, **k: None)
    events = _run(False)
    assert all(p == "Which niches are rising?" for _, _, p in calls)
    assert not any(e["type"] == "evidence" for e in events)


def test_market_mode_stops_if_evidence_unavailable(monkeypatch):
    calls = []
    _fake_pool(monkeypatch, calls)
    monkeypatch.delenv("MARKET_DATA_DIR", raising=False)
    monkeypatch.setattr(market_data, "ENV_FILE", Path("nope.env"))
    events = _run(True)
    assert calls == [] and events[-1]["type"] == "error"
