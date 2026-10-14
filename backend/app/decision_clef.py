"""Clef: an optional decision helper. Not a chat or coding model; never routed into agent conversations.

Native HTTP + JSON only. One POST per call, no retries, no streaming, no SDK.
The key is read only from EXPERIENTIAL_API_KEY and is never printed or logged.

VERIFIED IN THIS CODE (from the request spec, not from live documentation):
  - endpoint:   POST https://api.experientiallabs.ai/v1/systemone
  - model id:   "clef" or "clef:free", spelled exactly
  - body keys:  model, state, questions (question IDs chosen by the caller)
  - per question: type ("choice" | "noul" | "score") and instructions
  - admission:  at most 32 questions, 64 options per choice, 2-10 levels per score
  - shared context ceiling: 65,536 tokens (estimated here, not measured)

UNVERIFIED (docs unreachable from this sandbox; confirm before any live call):
  - the per-type field names below (CHOICE_CRITERIA, NOUL_PROPOSITION, SCORE_LEVELS)
  - the response shape. parse_response() is deliberately not implemented yet.
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

# UNVERIFIED field names for the type-specific parts of a question. Replace with the documented names.
CHOICE_CRITERIA = "criteria"        # {option: description}
NOUL_PROPOSITION = "proposition"    # the statement to judge
NOUL_TRUE = "true"                  # what "true" means
NOUL_FALSE = "false"                # what "false" means
SCORE_LEVELS = "levels"             # ordered list, 2-10 descriptions

DISCLAIMER = ("Probabilities and confidence values are model outputs, not guarantees. "
              "Review every decision before acting on it. Nothing here is executed or merged automatically.")


class DecisionError(RuntimeError):
    """Raised for missing configuration, HTTP errors, and transport errors. Messages never contain the key."""


class DecisionUnavailable(DecisionError):
    """The key is not set locally. No request was made."""


def estimate_tokens(text: str) -> int:
    # Conservative estimate: about 3 characters per token, which overcounts ordinary English and JSON.
    return math.ceil(len(text) / 3)


def validate_model(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS} exactly (got {model!r})")
    return model


def build_question(qtype: str, instructions: str, **fields) -> dict:
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"question type must be one of {QUESTION_TYPES}")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    q = {"type": qtype, "instructions": instructions}
    if qtype == "choice":
        criteria = fields.get("criteria")
        if not isinstance(criteria, dict) or not (1 <= len(criteria) <= MAX_OPTIONS):
            raise ValueError(f"choice needs a criteria map of 1-{MAX_OPTIONS} options")
        if not all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in criteria.items()):
            raise ValueError("choice criteria must map non-empty option names to non-empty descriptions")
        q[CHOICE_CRITERIA] = dict(criteria)
    elif qtype == "noul":
        prop, t, f = fields.get("proposition"), fields.get("true"), fields.get("false")
        if not all(isinstance(x, str) and x.strip() for x in (prop, t, f)):
            raise ValueError("noul needs non-empty proposition, true and false descriptions")
        q[NOUL_PROPOSITION], q[NOUL_TRUE], q[NOUL_FALSE] = prop, t, f
    else:  # score
        levels = fields.get("levels")
        if not isinstance(levels, list) or not (MIN_LEVELS <= len(levels) <= MAX_LEVELS):
            raise ValueError(f"score needs an ordered list of {MIN_LEVELS}-{MAX_LEVELS} level descriptions")
        if not all(isinstance(x, str) and x.strip() for x in levels):
            raise ValueError("score levels must be non-empty strings")
        q[SCORE_LEVELS] = list(levels)
    return q


def build_request(model: str, state, questions: dict) -> dict:
    """Validate admission bounds and return the JSON body. questions maps caller IDs to build_question() results."""
    validate_model(model)
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty mapping of question ID to question")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} questions per request (got {len(questions)})")
    if not all(isinstance(qid, str) and qid.strip() for qid in questions):
        raise ValueError("question IDs must be non-empty strings")
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text, an object, or an array")
    body = {"model": model, "state": state, "questions": questions}
    size = estimate_tokens(json.dumps(body))
    if size > CONTEXT_TOKEN_CEILING:
        raise ValueError(f"request is about {size} tokens, over the {CONTEXT_TOKEN_CEILING}-token ceiling")
    return body


def summarize_state(diff_stat: str, test_summary: str, max_chars: int = 2000) -> str:
    """Concise state for a decision: a diff summary and a test summary. Never the whole repository."""
    text = f"DIFF SUMMARY:\n{diff_stat.strip()}\n\nTEST SUMMARY:\n{test_summary.strip()}"
    return text if len(text) <= max_chars else text[: max_chars - 15] + "\n...[truncated]"


def _redact(text: str, key: str) -> str:
    return text.replace(key, "<redacted>") if key else text


def call_systemone(body: dict, timeout: float = TIMEOUT_SECONDS) -> dict:
    """POST the validated body once. Returns the parsed JSON response. No retries, no streaming."""
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


def parse_response(response: dict) -> dict:
    """NOT IMPLEMENTED: the response shape is unverified. Implement only after the documented shape is confirmed."""
    raise NotImplementedError("Response parsing waits for the verified SystemOne response format.")
