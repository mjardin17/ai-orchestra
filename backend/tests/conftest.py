import os, sys, tempfile
from pathlib import Path

# store.py opens "orchestra.db" in the CWD at import time, so isolate every test run in a temp dir.
_tmp = tempfile.mkdtemp(prefix="orchestra-test-")
os.chdir(_tmp)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest


@pytest.fixture
def tiny_market(tmp_path, monkeypatch):
    """A tiny but complete market-data directory (4 files) so scorers and the Arena can be tested without the real data."""
    d = tmp_path / "market"
    d.mkdir()
    (d / "dataset1_books.csv").write_text(
        "title,author,price,list_category,source_url\nA,x,$10.00,Cat1,http://u1\nB,y,$20.00,Cat1,http://u2\nC,z,$30.00,Cat2,http://u3\n",
        encoding="utf-8")
    (d / "dataset2_merch.csv").write_text("niche,price\nn1,$5.00\nn2,\n", encoding="utf-8")
    (d / "merch_listing_counts.csv").write_text("query,marketplace\nq1,etsy\n", encoding="utf-8")
    (d / "book_trends.csv").write_text("term,direction\nt1,up\n", encoding="utf-8")
    monkeypatch.setenv("MARKET_DATA_DIR", str(d))
    from app import market_data
    monkeypatch.setattr(market_data, "EVIDENCE_COLS", {
        "books": ["title", "author", "price", "list_category"], "merch": ["niche", "price"],
        "merch_counts": ["query", "marketplace"], "trends": ["term", "direction"]})
    return d
