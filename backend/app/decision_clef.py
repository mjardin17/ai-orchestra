"""Clef: an optional, human-invoked decision helper. Not a chat or coding model.

It reads a state and a set of independent questions and returns answers and usage. It never executes a decision,
merges changes, or bypasses review. Callers show the answers to a person.

Transport: one native HTTP POST per call, JSON in and out, a bounded timeout, no retries, no streaming, no SDK,
no Idempotency-Key. The key is read only from EXPERIENTIAL_API_KEY and is never printed, logged, or returned.

Request schema (as specified for this project):
  body = {"model": "clef" | "clef:free", "state": <text | object | array>, "questions": {<id>: question}}
  question = {"type": "choice" | "noul" | "score", "instructions": str, "criteria": ...}
    choice: criteria = {option: description}, 1-64 options
    noul:   criteria optional, may hold "true" and/or "false" descriptions
    score:  criteria = ordered list of 2-10 level descriptions

Local checks are admission bounds only: 32 questions, 64 options, 2-10 levels, and a conservative size check
(characters / 3) against the shared 65,536-token context ceiling. The size check is an estimate, not a token count.
"""
import json
import math
import os
import urllib.error
import urllib.request

ENDPOINT = "https://api.experientiallabs.ai/v1/systemone"
KEY_VAR = "EXPERIENTIAL_API_KEY"
MODELS = ("clef", "clef:free")
TIMEOUT_SECONDS = 30
USER_AGENT = "ai-orchestra/1.0"

MAX_QUESTIONS = 32
MAX_OPTIONS = 64
MIN_LEVELS, MAX_LEVELS = 2, 10
CONTEXT_TOKEN_CEILING = 65_536
QUESTION_TYPES = ("choice", "noul", "score")
NOUL_KEYS = ("true", "false")

DISCLAIMER = ("Probabilities and confidence values are model outputs, not guarantees. Review every answer "
              "before acting on it. Nothing here executes a decision or merges changes.")


class DecisionError(RuntimeError):
    """HTTP, transport, or response-format failure. Messages never contain the key."""


class DecisionUnavailable(DecisionError):
    """EXPERIENTIAL_API_KEY is not set. No request was made."""


def validate_model(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS} exactly (got {model!r})")
    return model


def _text(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    return value


def validate_question(q) -> dict:
    if not isinstance(q, dict):
        raise ValueError("each question must be an object")
    qtype = q.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"question type must be one of {QUESTION_TYPES} (got {qtype!r})")
    _text(q.get("instructions"), "instructions")
    criteria = q.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not (1 <= len(criteria) <= MAX_OPTIONS):
            raise ValueError(f"choice criteria must be an option-to-description map with 1-{MAX_OPTIONS} options")
        for option, description in criteria.items():
            _text(option, "choice option name")
            _text(description, f"description for option {option!r}")
    elif qtype == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict) or not criteria or not set(criteria) <= set(NOUL_KEYS):
                raise ValueError('noul criteria, if given, must map "true" and/or "false" to descriptions')
            for key, description in criteria.items():
                _text(description, f'noul description for "{key}"')
    else:  # score
        if not isinstance(criteria, list) or not (MIN_LEVELS <= len(criteria) <= MAX_LEVELS):
            raise ValueError(f"score criteria must be an ordered list of {MIN_LEVELS}-{MAX_LEVELS} level descriptions")
        for level in criteria:
            _text(level, "score level")
    return q


def build_request(model: str, state, questions) -> dict:
    """Validate the request and return the JSON body. Raises ValueError with a readable reason."""
    validate_model(model)
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object keyed by question ID")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} questions per request (got {len(questions)})")
    for qid, q in questions.items():
        _text(qid, "question ID")
        validate_question(q)
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text, an object, or an array")
    body = {"model": model, "state": state, "questions": questions}
    upper = math.ceil(len(json.dumps(body)) / 3)  # conservative size estimate, not a token count
    if upper > CONTEXT_TOKEN_CEILING:
        raise ValueError(f"request is too large for the local pre-check: its size estimate ({upper}) exceeds "
                         f"the shared {CONTEXT_TOKEN_CEILING}-token ceiling")
    return body


def summarize_state(diff_summary: str, test_summary: str, max_chars: int = 2000) -> str:
    """Concise state: a diff summary and a test summary. Never the whole repository."""
    text = f"DIFF SUMMARY:\n{diff_summary.strip()}\n\nTEST SUMMARY:\n{test_summary.strip()}"
    return text if len(text) <= max_chars else text[: max_chars - 15] + "\n...[truncated]"


def _redact(text: str, key: str) -> str:
    return text.replace(key, "<redacted>") if key else text


def call_systemone(body: dict, timeout: float = TIMEOUT_SECONDS) -> dict:
    """POST the body once and return the parsed JSON response. No retries, no streaming."""
    key = os.getenv(KEY_VAR, "").strip()
    if not key:
        raise DecisionUnavailable(f"{KEY_VAR} is not set. Set it locally in your environment, then retry.")
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = _redact(e.read()[:300].decode("utf-8", "replace"), key)
        raise DecisionError(f"SystemOne returned HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DecisionError(f"SystemOne transport error: {_redact(str(getattr(e, 'reason', e)), key)}") from None
    try:
        return json.loads(raw)
    except ValueError:
        raise DecisionError("SystemOne returned a body that is not JSON") from None


def parse_response(response, question_ids) -> dict:
    """Read answers by question ID and usage. Shape of each answer is passed through, not interpreted."""
    if not isinstance(response, dict):
        raise DecisionError("SystemOne response is not a JSON object")
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise DecisionError('SystemOne response has no "answers" object')
    ids = list(question_ids)
    return {
        "answers": {qid: answers[qid] for qid in ids if qid in answers},
        "missing": [qid for qid in ids if qid not in answers],
        "unexpected": [qid for qid in answers if qid not in ids],
        "usage": response.get("usage"),
    }


def format_report(parsed: dict, max_answer_chars: int = 500) -> str:
    """Human-readable answers and usage, with the disclaimer."""
    lines = ["ANSWERS:"]
    for qid, value in parsed["answers"].items():
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"  {qid}: {text if len(text) <= max_answer_chars else text[:max_answer_chars] + '...'}")
    if parsed["missing"]:
        lines.append(f"NO ANSWER RETURNED FOR: {', '.join(parsed['missing'])}")
    if parsed["unexpected"]:
        lines.append(f"UNEXPECTED ANSWER IDS: {', '.join(parsed['unexpected'])}")
    usage = parsed["usage"]
    lines.append("USAGE (as reported): " + (json.dumps(usage, sort_keys=True) if usage is not None else "not reported"))
    lines.append(DISCLAIMER)
    return "\n".join(lines)
