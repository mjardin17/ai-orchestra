"""Read-only access to the Market Intelligence datasets for the Arena.

The data directory comes from MARKET_DATA_DIR (process env, else backend/.env). Files are never copied,
modified or deleted. Every row is addressed as "<file>:<line>" where line is the 1-based line number in the
original CSV (header = line 1), so any claim in a model answer can be traced back to the source row."""
import csv, hashlib, os
from collections import Counter
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
FILES = {
    "books": "dataset1_books.csv",
    "merch": "dataset2_merch.csv",
    "merch_counts": "merch_listing_counts.csv",
    "trends": "book_trends.csv",
}
REPORT = "MARKET_INTELLIGENCE_REPORT.md"
SUMMARY_COLS = {  # columns worth tallying per file
    "books": ["list_category", "source_list", "format", "status_or_trend"],
    "merch": ["product_type", "niche", "marketplace", "ip_risk_flags"],
    "merch_counts": ["marketplace"],
    "trends": ["direction"],
}


def data_dir() -> Path:
    val = os.getenv("MARKET_DATA_DIR")
    if not val and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "MARKET_DATA_DIR":
                val = v.split("#")[0].strip().strip('"\'')
    if not val:
        raise RuntimeError("MARKET_DATA_DIR is not set (put it in backend/.env or the environment)")
    return Path(val)


def read_rows(key: str) -> list:
    """Rows as dicts, each tagged with _row = '<file>:<csv line>' for traceability."""
    name = FILES[key]
    with open(data_dir() / name, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # line_num is the physical line of the record's end; track start for multi-line cells
        rows, prev = [], 1
        for r in reader:
            r["_row"] = f"{name}:{prev + 1}"
            prev = reader.line_num
            rows.append(r)
    return rows


def status() -> dict:
    d = data_dir()
    out = {"dir": str(d), "files": {}}
    for key, name in {**FILES, "report": REPORT}.items():
        p = d / name
        info = {"file": name, "exists": p.is_file()}
        if p.is_file() and key != "report":
            info["records"] = len(read_rows(key))
        out["files"][key] = info
    return out


def digest(top: int = 10) -> dict:
    """Per-file evidence summary: record count, column list, top values, and the report text."""
    out = {"dir": str(data_dir()), "files": {}}
    for key in FILES:
        rows = read_rows(key)
        tallies = {c: Counter(r.get(c, "") for r in rows if r.get(c)).most_common(top) for c in SUMMARY_COLS[key]}
        out["files"][key] = {"file": FILES[key], "records": len(rows), "columns": [c for c in rows[0] if c != "_row"] if rows else [],
                             "top_values": tallies, "first_row": rows[0]["_row"] if rows else None, "last_row": rows[-1]["_row"] if rows else None}
    rp = data_dir() / REPORT
    out["report"] = rp.read_text(encoding="utf-8") if rp.is_file() else None
    return out


# ---- Deterministic evidence block handed (identically) to every Arena model and to the judge -------------------------
# The raw CSVs are ~780 KB, too large to paste whole into every model's context, so the evidence is a fixed projection:
# every record of every file, in file order, with the columns below. Columns left out are listed in the block header and
# stay retrievable through the row ID. Nothing is summarised, ranked or interpreted.
EVIDENCE_COLS = {
    "books": ["list_category", "source_list", "rank", "title", "author", "format", "price", "rating", "review_count",
              "publication_date", "series_name", "series_number", "status_or_trend", "ip_notes"],
    "merch": ["product_type", "niche", "design_concept", "slogan_text", "visual_style", "price", "rating", "review_count",
              "sales_indicator", "marketplace", "rank_or_trending_indicator", "personalization", "seasonality", "audience",
              "ip_risk_flags", "evidence_type"],
    "merch_counts": ["query", "marketplace", "result_count_text", "collected_at"],
    "trends": ["term", "interest", "direction", "collected_at"],
}
MAX_EVIDENCE_CHARS = int(os.getenv("MARKET_EVIDENCE_MAX_CHARS", "700000"))  # refuse rather than silently truncate


def _cell(v) -> str:
    return " ".join((v or "").split())  # tabs/newlines -> single spaces so each record stays on one line


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evidence() -> dict:
    """{'text', 'sha256', 'chars', 'est_tokens', 'source', 'files'}. Same input files -> byte-identical text."""
    d = data_dir()
    parts, files = [], {}
    header = [
        "=== MARKET INTELLIGENCE EVIDENCE (REAL DATA, NOT SYNTHETIC) ===",
        f"SOURCE DIRECTORY: {d}",
        "Every record below comes from the original CSV files in that directory. Nothing has been summarised or interpreted.",
        "ROW ID: each line starts with <file>:<n>, where n is the 1-based line number in the original CSV (header = line 1).",
        "Fields are TAB-separated in the column order shown. An empty field means the source cell was empty.",
    ]
    for key, name in FILES.items():
        rows = read_rows(key)
        cols = EVIDENCE_COLS[key]
        omitted = [c for c in rows[0] if c not in cols and c != "_row"]
        files[key] = {"file": name, "records": len(rows), "sha256": _sha(d / name), "omitted_columns": omitted}
        parts.append(f"\n--- {name} | {len(rows)} records | sha256 {files[key]['sha256']} ---\n"
                     "columns: row_id\t" + "\t".join(cols) + "\n"
                     f"omitted columns (look them up in the original file by row ID): {', '.join(omitted) or 'none'}")
        parts.extend(r["_row"] + "\t" + "\t".join(_cell(r.get(c)) for c in cols) for r in rows)
    text = "\n".join(header + parts) + "\n=== END OF EVIDENCE ===\n"
    if len(text) > MAX_EVIDENCE_CHARS:
        raise ValueError(f"Evidence is {len(text)} chars, over MARKET_EVIDENCE_MAX_CHARS={MAX_EVIDENCE_CHARS}; refusing to truncate")
    return {"text": text, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text),
            "est_tokens": len(text) // 4, "source": str(d), "files": files}


# ---- Evidence index used by the deterministic scorers ---------------------------------------------------------------
class EvidenceIndex:
    """Lets a scorer check a claim against the SAME files and columns the models were shown.
    `sha256` is the hash of the evidence text built from the files as they are right now; compare it with the hash an
    attempt was run on before trusting any score (the scorers do)."""

    def __init__(self):
        ev = build_evidence()
        self.sha256, self.source = ev["sha256"], ev["source"]
        self.files = {}
        for key, name in FILES.items():
            rows = read_rows(key)
            self.files[name] = {
                "columns": [c for c in rows[0] if c != "_row"] if rows else [],
                "evidence_columns": list(EVIDENCE_COLS[key]),
                "rows": {int(r["_row"].rsplit(":", 1)[1]): r for r in rows},
            }

    def locate(self, row_id):
        """(status, file, row dict or None). status: ok, malformed, unknown_file, row_not_found."""
        if not isinstance(row_id, str) or ":" not in row_id:
            return "malformed", None, None
        name, _, line = row_id.strip().rpartition(":")
        if not line.isdigit():
            return "malformed", None, None
        if name not in self.files:
            return "unknown_file", name, None
        row = self.files[name]["rows"].get(int(line))
        return ("ok", name, row) if row else ("row_not_found", name, None)

    def cell(self, row, field):
        return _cell(row.get(field))
