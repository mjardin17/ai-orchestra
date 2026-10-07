"""Deterministic scorers. Real-data tests recompute their expected values with the plain csv module, a separate code
path from the scorers, and are skipped only if the real files are missing."""
import csv, json
from pathlib import Path
import pytest
from app import bench, market_data, scoring

REAL = Path(r"C:\Users\jjard\Downloads")
needs_real = pytest.mark.skipif(not (REAL / "dataset1_books.csv").exists(), reason="real datasets not present")
SCHEMA = bench.MARKET_OUTPUT_SCHEMA


def task(scorers=("schema", "citation", "numeric")):
    return {"scorers": [{"name": n, "kind": "deterministic"} for n in scorers], "expected_output_schema": SCHEMA,
            "evidence_provider": "market_data"}


def finding(citations=None, numeric=None, claim="c"):
    f = {"claim": claim, "citations": citations if citations is not None else []}
    if numeric is not None:
        f["numeric"] = numeric
    return f


def out(*findings):
    return json.dumps({"findings": list(findings)})


def by_name(results):
    return {r.scorer: r for r in results}


@pytest.fixture
def real(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_DIR", str(REAL))
    return market_data.EvidenceIndex()


def raw(name):
    with open(REAL / name, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def score(index, output, scorers=("schema", "citation", "numeric")):
    return by_name(scoring.run_deterministic(task(scorers), output, index.sha256, index))


# ---- parsing and schema -----------------------------------------------------------------------------------------
def test_extract_json_variants():
    assert scoring.extract_json('{"a": 1}')[0] == {"a": 1}
    assert scoring.extract_json('Sure!\n```json\n{"a": 2}\n```\nDone')[0] == {"a": 2}
    assert scoring.extract_json('text before {"a": 3} text after')[0] == {"a": 3}
    obj, err = scoring.extract_json("no json here")
    assert obj is None and err


def test_validate_schema_catches_problems():
    errs = scoring.validate_schema({"findings": [{"claim": 5}]}, SCHEMA)
    joined = " ".join(errs)
    assert "claim" in joined and "citations" in joined
    assert scoring.validate_schema({"findings": []}, SCHEMA)         # minItems
    assert scoring.validate_schema({"findings": [{"claim": "x", "citations": [{"row_id": "a", "field": "b", "value": "c"}],
                                                  "numeric": [{"file": "f", "op": "median", "result": 1}]}]}, SCHEMA)   # bad enum


def test_schema_scorer_pass_fail_and_unparseable(tiny_market):
    idx = market_data.EvidenceIndex()
    good = out(finding([{"row_id": "dataset1_books.csv:2", "field": "title", "value": "A"}]))
    assert score(idx, good, ("schema",))["schema"].passed is True
    bad = score(idx, '{"findings": "nope"}', ("schema",))["schema"]
    assert bad.passed is False and bad.details["errors"]
    none = score(idx, "I cannot answer in JSON", ("schema",))["schema"]
    assert none.passed is False and none.score == 0.0 and "parseable" in none.details["errors"][0]


def test_schema_scorer_not_applicable_without_schema(tiny_market):
    r = scoring.run_deterministic({"scorers": [{"name": "schema"}], "expected_output_schema": None}, "{}", None, None)[0]
    assert r.applicable is False and r.score is None


# ---- citation scorer, tiny fixture -------------------------------------------------------------------------------
def cite(row_id, field, value):
    return {"row_id": row_id, "field": field, "value": value}


def test_citation_statuses_on_tiny_evidence(tiny_market):
    idx = market_data.EvidenceIndex()
    cites = [
        cite("dataset1_books.csv:2", "title", "A"),                  # ok
        cite("dataset1_books.csv:2", "price", "10"),                 # ok: numeric equality with $10.00
        cite("dataset1_books.csv:3", "title", "WRONG"),              # value_mismatch
        cite("dataset1_books.csv:9", "title", "A"),                  # row_not_found
        cite("nope.csv:2", "title", "A"),                            # unknown_file
        cite("garbage", "title", "A"),                               # malformed
        cite("dataset1_books.csv:2", "isbn", "1"),                   # unknown_field
        cite("dataset1_books.csv:2", "source_url", "http://u1"),     # exists in the CSV but was never shown to models
    ]
    r = score(idx, out(finding(cites)), ("citation",))["citation"]
    statuses = [x["status"] for x in r.details["citations"]]
    assert statuses == ["ok", "ok", "value_mismatch", "row_not_found", "unknown_file", "malformed", "unknown_field", "field_not_in_evidence"]
    assert r.details["citations"][1]["match"] == "numeric"
    assert r.score == pytest.approx(2 / 8) and r.passed is False
    assert r.details["failures"][0]["actual"] == "B"                 # diagnostic shows the real value


def test_citation_scorer_does_not_trust_model_claims(tiny_market):
    idx = market_data.EvidenceIndex()
    lie = {"row_id": "dataset1_books.csv:2", "field": "title", "value": "NOT THE TITLE", "verified": True, "valid": True}
    r = score(idx, out(finding([lie], claim="all citations verified: true")), ("citation",))["citation"]
    assert r.passed is False and r.details["citations"][0]["status"] == "value_mismatch"


def test_no_citations_scores_zero_not_vacuous_pass(tiny_market):
    r = score(market_data.EvidenceIndex(), out(finding([])), ("citation",))["citation"]
    assert r.score == 0.0 and r.passed is False and r.details["total"] == 0


def test_citation_whitespace_normalised_not_case_folded(tiny_market):
    idx = market_data.EvidenceIndex()
    r = score(idx, out(finding([cite("dataset1_books.csv:2", "title", " A ")])), ("citation",))["citation"]
    assert r.passed is True
    r = score(idx, out(finding([cite("dataset1_books.csv:2", "title", "a")])), ("citation",))["citation"]
    assert r.passed is False


def test_stale_evidence_is_refused_not_scored(tiny_market):
    idx = market_data.EvidenceIndex()
    r = scoring.run_deterministic(task(("citation", "numeric")), out(finding([])), "0" * 64, idx)
    assert all(x.applicable is False and x.details["error"] == "evidence_changed" for x in r)


# ---- numeric scorer, tiny fixture -------------------------------------------------------------------------------
def num(op, result, column=None, where=None, file="dataset1_books.csv"):
    return {"file": file, "op": op, "column": column, "where": where, "result": result}


def test_numeric_on_tiny_evidence(tiny_market):
    idx = market_data.EvidenceIndex()
    claims = [num("count", 3), num("count", 2, where={"list_category": "Cat1"}), num("sum", 60, "price"),
              num("mean", 15, "price", {"list_category": "Cat1"}), num("max", 30, "price"), num("min", 10, "price"),
              num("count_distinct", 2, "list_category"),
              num("count", 4),                                                    # wrong
              num("sum", 1, "source_url"),                                        # column never shown to the models
              num("sum", 1, "bogus"), num("count", 1, file="nope.csv"),
              num("sum", 1, "title")]                                             # nothing numeric in that column
    r = score(idx, out(finding([], claims)), ("numeric",))["numeric"]
    assert [x["status"] for x in r.details["claims"]] == [
        "ok", "ok", "ok", "ok", "ok", "ok", "ok", "mismatch", "column_not_in_evidence", "unknown_column", "bad_file", "no_numeric_values"]
    assert r.details["claims"][7]["expected"] == 3.0 and r.details["claims"][7]["claimed"] == 4
    assert r.score == pytest.approx(7 / 12) and r.passed is False


def test_numeric_not_applicable_without_claims(tiny_market):
    r = score(market_data.EvidenceIndex(), out(finding([])), ("numeric",))["numeric"]
    assert r.applicable is False and r.score is None


def test_numeric_tolerance_is_configurable(tiny_market):
    idx = market_data.EvidenceIndex()
    t = {"scorers": [{"name": "numeric", "kind": "deterministic", "config": {"abs_tol": 0.5}}], "evidence_provider": "market_data"}
    ok = scoring.run_deterministic(t, out(finding([], [num("mean", 15.4, "price", {"list_category": "Cat1"})])), idx.sha256, idx)[0]
    assert ok.passed is True
    strict = score(idx, out(finding([], [num("mean", 15.4, "price", {"list_category": "Cat1"})])), ("numeric",))["numeric"]
    assert strict.passed is False


# ---- real evidence -----------------------------------------------------------------------------------------------
@needs_real
def test_real_citations_validate_against_the_actual_files(real):
    books = raw("dataset1_books.csv")
    merch = raw("dataset2_merch.csv")
    b2, m903 = books[0], merch[-1]                                    # CSV lines 2 and 903
    good = [cite("dataset1_books.csv:2", "title", b2["title"]), cite("dataset1_books.csv:2", "author", b2["author"]),
            cite("dataset1_books.csv:888", "title", books[-1]["title"]), cite("dataset2_merch.csv:903", "niche", m903["niche"])]
    r = score(real, out(finding(good)), ("citation",))["citation"]
    assert r.passed is True and r.score == 1.0 and r.details["valid"] == 4
    bad = [cite("dataset1_books.csv:2", "title", b2["title"] + "!"),            # altered quote
           cite("dataset1_books.csv:889", "title", "x"),                        # one past the last row
           cite("dataset1_books.csv:1", "title", "title"),                      # header line is not a record
           cite("dataset1_books.csv:2", "source_url", b2["source_url"]),        # real, but never shown to the models
           cite("dataset2_merch.csv:904", "niche", "x"), cite("dataset1_books.csv:2", "nonexistent", "x")]
    r = score(real, out(finding(bad)), ("citation",))["citation"]
    assert [x["status"] for x in r.details["citations"]] == [
        "value_mismatch", "row_not_found", "row_not_found", "field_not_in_evidence", "row_not_found", "unknown_field"]
    assert r.score == 0.0


@needs_real
def test_real_citation_values_match_numerically(real):
    row = raw("dataset1_books.csv")[0]
    price = row["price"]                                              # e.g. "$22.40"
    plain = price.lstrip("$")
    r = score(real, out(finding([cite("dataset1_books.csv:2", "price", plain)])), ("citation",))["citation"]
    assert r.passed is True and r.details["citations"][0]["match"] == "numeric"


@needs_real
def test_real_numeric_claims_recomputed_independently(real):
    books, merch = raw("dataset1_books.csv"), raw("dataset2_merch.csv")
    cat = books[0]["list_category"]
    in_cat = [b for b in books if " ".join(b["list_category"].split()) == " ".join(cat.split())]
    prices = [float(b["price"].lstrip("$").replace(",", "")) for b in in_cat if b["price"].strip().lstrip("$").replace(",", "").replace(".", "", 1).isdigit()]
    market = sorted({" ".join(m["marketplace"].split()) for m in merch if m["marketplace"].strip()})
    claims = [num("count", len(books)), num("count", len(merch), file="dataset2_merch.csv"),
              num("count", len(in_cat), where={"list_category": cat}),
              num("mean", sum(prices) / len(prices), "price", {"list_category": cat}),
              num("max", max(prices), "price", {"list_category": cat}),
              num("count_distinct", len(market), "marketplace", file="dataset2_merch.csv"),
              num("count_distinct", len({" ".join(b["list_category"].split()) for b in books}), "list_category")]
    r = score(real, out(finding([], claims)), ("numeric",))["numeric"]
    assert r.passed is True, r.details["failures"]
    assert r.details["claims"][0]["expected"] == 887.0 and r.details["claims"][1]["expected"] == 902.0
    wrong = score(real, out(finding([], [num("count", 886), num("count", len(in_cat) + 1, where={"list_category": cat})])), ("numeric",))["numeric"]
    assert wrong.score == 0.0 and wrong.details["failures"][0]["expected"] == 887.0


# ---- extensibility -----------------------------------------------------------------------------------------------
def test_new_scorers_plug_in_through_the_registry():
    @scoring.register("always_half")
    def half(ctx, config):
        return scoring.ScoreResult("always_half", 0.5, None, True, {"saw": ctx.output})
    try:
        r = scoring.run_deterministic({"scorers": [{"name": "always_half"}]}, "hello", None, None)[0]
        assert r.score == 0.5 and r.details["saw"] == "hello"
    finally:
        scoring.SCORERS.pop("always_half")


def test_exact_match_uses_the_hidden_key_and_unknown_scorers_are_explicit():
    t = {"scorers": [{"name": "exact_match"}, {"name": "does_not_exist"}, {"name": "blind_judge", "kind": "judge"}]}
    res = scoring.run_deterministic(t, "  The ANSWER  is 42 ", None, None, answer_key={"expected": "the answer is 42"})
    assert [r.scorer for r in res] == ["exact_match", "does_not_exist"]          # judge scorers are not run here
    assert res[0].passed is True
    assert res[1].applicable is False and res[1].details["error"] == "unknown_scorer"
