"""Deterministic scorers. They never trust what a model says about its own answer: every citation and every number is
re-checked against the evidence files.

Interface (add new scorers with @register):
    fn(ctx: ScoreContext, config: dict) -> ScoreResult
ScoreResult.score is 0..1, or None when the scorer does not apply / cannot be computed (never a fake 0 or 1).
Hidden code tests, exact-match answer keys and other validators plug in the same way (ctx.answer_key carries the
task's hidden key; it is only ever visible to scorers, never to a contestant)."""
import json, math, re
from dataclasses import dataclass, field
from typing import Optional
from . import bench, market_data


@dataclass
class ScoreResult:
    scorer: str
    score: Optional[float]
    passed: Optional[bool]
    applicable: bool = True
    details: dict = field(default_factory=dict)


@dataclass
class ScoreContext:
    output: str
    schema: Optional[dict] = None
    evidence: object = None                 # an index object (e.g. market_data.EvidenceIndex) or None
    evidence_sha256: Optional[str] = None   # hash of the evidence the attempt was actually given
    answer_key: object = None
    parsed: Optional[dict] = None
    parse_error: Optional[str] = None


SCORERS: dict = {}
EVIDENCE_PROVIDERS = {"market_data": market_data.EvidenceIndex}


def register(name):
    def deco(fn):
        SCORERS[name] = fn
        return fn
    return deco


# ---- helpers -------------------------------------------------------------------------------------------------------
def extract_json(text):
    """(object, None) for the first JSON object in the text (plain or inside a ``` fence), else (None, reason)."""
    t = (text or "").strip()
    cands = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    if fence:
        cands.append(fence.group(1).strip())
    cands.append(t)
    dec = json.JSONDecoder()
    for c in cands:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj, None
        except ValueError:
            pass
        for m in re.finditer(r"\{", c):
            try:
                obj, _ = dec.raw_decode(c[m.start():])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj, None
    return None, "no parseable JSON object found in the output"


def parse_number(text):
    t = " ".join(str(text if text is not None else "").split()).replace(",", "")
    t = t[1:] if t.startswith("$") else t
    t = t[:-1] if t.endswith("%") else t
    return float(t) if re.fullmatch(r"-?\d+(\.\d+)?", t) else None


def values_match(quoted, actual):
    """'exact' (whitespace-normalised equality), 'numeric' (same number, e.g. 22.4 vs $22.40), or None."""
    q = " ".join(str(quoted).split())
    if q == actual:
        return "exact"
    nq, na = parse_number(q), parse_number(actual)
    if nq is not None and na is not None and math.isclose(nq, na, rel_tol=1e-9, abs_tol=1e-9):
        return "numeric"
    return None


_TYPES = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list), "string": lambda v: isinstance(v, str),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
          "boolean": lambda v: isinstance(v, bool), "null": lambda v: v is None}


def validate_schema(value, schema, path="$"):
    """Small JSON-Schema subset: type, enum, required, properties, items, minItems. Returns a list of problems."""
    errs = []
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[t](value) for t in types):
            return [f"{path}: expected {'|'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        errs += [f"{path}: missing required '{k}'" for k in schema.get("required", []) if k not in value]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += validate_schema(value[k], sub, f"{path}.{k}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: needs at least {schema['minItems']} item(s), has {len(value)}")
        if "items" in schema:
            for i, item in enumerate(value):
                errs += validate_schema(item, schema["items"], f"{path}[{i}]")
    return errs


def _stale(ctx, name):
    """A ScoreResult explaining why evidence-based scoring can't be trusted, or None if it can."""
    if ctx.evidence is None:
        return ScoreResult(name, None, None, False, {"error": "no_evidence_index", "message": "this task has no usable evidence provider"})
    if ctx.evidence_sha256 and ctx.evidence.sha256 != ctx.evidence_sha256:
        return ScoreResult(name, None, None, False, {
            "error": "evidence_changed", "attempt_evidence_sha256": ctx.evidence_sha256, "current_evidence_sha256": ctx.evidence.sha256,
            "message": "the files changed since this attempt ran; scoring against them would be meaningless"})
    return None


def _findings(ctx):
    f = (ctx.parsed or {}).get("findings")
    return [x for x in f if isinstance(x, dict)] if isinstance(f, list) else []


# ---- scorers -------------------------------------------------------------------------------------------------------
@register("schema")
def score_schema(ctx, config):
    if not ctx.schema:
        return ScoreResult("schema", None, None, False, {"message": "task defines no expected_output_schema"})
    if ctx.parsed is None:
        return ScoreResult("schema", 0.0, False, True, {"errors": [ctx.parse_error]})
    errs = validate_schema(ctx.parsed, ctx.schema)
    return ScoreResult("schema", 0.0 if errs else 1.0, not errs, True, {"errors": errs[:25], "error_count": len(errs)})


@register("citation")
def score_citation(ctx, config):
    stale = _stale(ctx, "citation")
    if stale:
        return stale
    per, total, valid = [], 0, 0
    for fi, finding in enumerate(_findings(ctx)):
        cites = finding.get("citations")
        for ci, c in enumerate(cites if isinstance(cites, list) else []):
            total += 1
            rec = {"finding": fi, "citation": ci, "row_id": c.get("row_id") if isinstance(c, dict) else None}
            if not isinstance(c, dict) or not {"row_id", "field", "value"} <= set(c):
                rec["status"] = "malformed"
            else:
                status, fname, row = ctx.evidence.locate(c["row_id"])
                rec.update(field=c["field"], quoted=c["value"])
                if status != "ok":
                    rec["status"] = {"unknown_file": "unknown_file", "row_not_found": "row_not_found"}.get(status, "malformed")
                elif c["field"] not in ctx.evidence.files[fname]["columns"]:
                    rec["status"] = "unknown_field"
                elif c["field"] not in ctx.evidence.files[fname]["evidence_columns"]:
                    rec["status"] = "field_not_in_evidence"   # exists in the CSV but the models were never shown it
                else:
                    actual = ctx.evidence.cell(row, c["field"])
                    how = values_match(c["value"], actual)
                    rec.update(status="ok" if how else "value_mismatch", match=how, actual=actual[:200])
            if rec["status"] == "ok":
                valid += 1
            per.append(rec)
    if total == 0:
        return ScoreResult("citation", 0.0, False, True, {"total": 0, "valid": 0, "message": "no citations to verify", "citations": []})
    bad = [r for r in per if r["status"] != "ok"]
    return ScoreResult("citation", valid / total, valid == total, True,
                       {"total": total, "valid": valid, "invalid": len(bad), "citations": per[:200], "failures": bad[:50]})


NUMERIC_OPS = ("count", "sum", "mean", "min", "max", "count_distinct")


def _recompute(ctx, claim, abs_tol, rel_tol):
    rec = {k: claim.get(k) for k in ("description", "file", "op", "column", "where")}
    rec["claimed"] = claim.get("result")
    idx = ctx.evidence
    name, op, col = claim.get("file"), claim.get("op"), claim.get("column")
    result = claim.get("result")
    if name not in idx.files:
        return {**rec, "status": "bad_file"}
    if op not in NUMERIC_OPS:
        return {**rec, "status": "bad_op"}
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        return {**rec, "status": "malformed", "message": "result must be a number"}
    f = idx.files[name]
    where = claim.get("where") or {}
    if not isinstance(where, dict):
        return {**rec, "status": "malformed", "message": "where must be an object"}
    for c in list(where) + ([col] if col else []):
        if c not in f["columns"]:
            return {**rec, "status": "unknown_column", "column_checked": c}
        if c not in f["evidence_columns"]:
            return {**rec, "status": "column_not_in_evidence", "column_checked": c}
    rows = [r for r in f["rows"].values() if all(values_match(v, idx.cell(r, c)) for c, v in where.items())]
    rec["matched_rows"] = len(rows)
    if op == "count":
        expected = float(len(rows))
    else:
        if not col:
            return {**rec, "status": "malformed", "message": f"{op} needs a column"}
        vals = [idx.cell(r, col) for r in rows]
        non_empty = [v for v in vals if v != ""]
        rec["skipped_empty"] = len(vals) - len(non_empty)
        if op == "count_distinct":
            expected = float(len(set(non_empty)))
        else:
            nums = [parse_number(v) for v in non_empty]
            rec["skipped_unparseable"] = sum(1 for n in nums if n is None)
            nums = [n for n in nums if n is not None]
            if not nums:
                return {**rec, "status": "no_numeric_values"}
            expected = {"sum": sum(nums), "mean": sum(nums) / len(nums), "min": min(nums), "max": max(nums)}[op]
    rec.update(expected=expected, tolerance={"abs": abs_tol, "rel": rel_tol})
    rec["status"] = "ok" if math.isclose(float(result), expected, rel_tol=rel_tol, abs_tol=abs_tol) else "mismatch"
    return rec


@register("numeric")
def score_numeric(ctx, config):
    stale = _stale(ctx, "numeric")
    if stale:
        return stale
    abs_tol, rel_tol = config.get("abs_tol", 0.005), config.get("rel_tol", 1e-4)
    per = []
    for fi, finding in enumerate(_findings(ctx)):
        claims = finding.get("numeric")
        for ni, claim in enumerate(claims if isinstance(claims, list) else []):
            per.append({"finding": fi, "numeric": ni, **(_recompute(ctx, claim, abs_tol, rel_tol) if isinstance(claim, dict)
                                                         else {"status": "malformed"})})
    if not per:
        return ScoreResult("numeric", None, None, False, {"message": "no numeric claims were made"})
    ok = sum(1 for r in per if r["status"] == "ok")
    return ScoreResult("numeric", ok / len(per), ok == len(per), True,
                       {"total": len(per), "correct": ok, "claims": per[:200], "failures": [r for r in per if r["status"] != "ok"][:50]})


@register("exact_match")
def score_exact_match(ctx, config):
    if ctx.answer_key is None:
        return ScoreResult("exact_match", None, None, False, {"message": "task has no hidden answer key"})
    norm = lambda s: " ".join(str(s).split()).casefold()
    ok = norm(ctx.output) == norm(ctx.answer_key if not isinstance(ctx.answer_key, dict) else ctx.answer_key.get("expected", ""))
    return ScoreResult("exact_match", 1.0 if ok else 0.0, ok, True, {})


# ---- running them --------------------------------------------------------------------------------------------------
def run_deterministic(task: dict, output: str, evidence_sha256=None, evidence=None, answer_key=None) -> list:
    """Run every deterministic scorer the task lists. `evidence` may be passed in to reuse one index across attempts."""
    specs = [s if isinstance(s, dict) else {"name": s} for s in task.get("scorers", [])]
    specs = [s for s in specs if s.get("kind", "deterministic") == "deterministic"]
    if evidence is None and task.get("evidence_provider") in EVIDENCE_PROVIDERS and any(
            s["name"] in ("citation", "numeric") for s in specs):
        evidence = EVIDENCE_PROVIDERS[task["evidence_provider"]]()
    parsed, perr = extract_json(output) if output else (None, "empty output")
    ctx = ScoreContext(output or "", task.get("expected_output_schema"), evidence, evidence_sha256, answer_key, parsed, perr)
    out = []
    for s in specs:
        fn = SCORERS.get(s["name"])
        out.append(fn(ctx, s.get("config") or {}) if fn else
                   ScoreResult(s["name"], None, None, False, {"error": "unknown_scorer"}))
    return out


def evaluate_attempt(episode_id, attempt: dict, task: dict, evidence=None, answer_key=None) -> list:
    """Score one finished attempt and store each result as an evaluations row (evaluator='deterministic')."""
    if attempt["status"] != "ok":
        return []
    results = run_deterministic(task, attempt["output"], bench.get_episode(episode_id)["evidence_sha256"], evidence, answer_key)
    for r in results:
        bench.add_evaluation(episode_id, attempt["id"], "deterministic", r.scorer, r.score, 1.0, r.passed, r.applicable, r.details)
    return results
