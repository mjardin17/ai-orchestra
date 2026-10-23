"""Server-side authorization for the facility endpoint. Everything fails CLOSED: a missing secret disables the feature.

Grant format: <exp>.<nonce>.<hmac-sha256 of "exp|nonce|task_digest">, single use.
Two separate secrets (both >= 16 chars, set only in the server's environment):
- FACILITY_API_TOKEN: who may call the endpoint at all (Authorization: Bearer <token>).
- FACILITY_GRANT_SECRET: signs write grants. A grant is bound to ONE task's exact content and expires, so it cannot be
  reused for another task or later. Execution permission (a grant) is separate from approving the finished result."""
import hashlib, hmac, json, os, secrets, time

MIN_SECRET_LEN = 16
GRANT_TTL_SEC = 600
GRANT_SKEW_SEC = 60
_USED: dict = {}      # grant MACs already consumed (single use), kept until they expire; per process


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _secret(name: str):
    v = os.getenv(name, "")
    return v if len(v) >= MIN_SECRET_LEN else None


def _same(a: str, b: str) -> bool:
    return hmac.compare_digest(hashlib.sha256(a.encode()).digest(), hashlib.sha256(b.encode()).digest())


def check_api_token(authorization) -> None:
    token = _secret("FACILITY_API_TOKEN")
    if not token:
        raise AuthError(503, f"facility endpoint disabled: set FACILITY_API_TOKEN (at least {MIN_SECRET_LEN} characters)")
    presented = authorization[7:].strip() if authorization and authorization[:7].lower() == "bearer " else ""
    if not presented or not _same(presented, token):
        raise AuthError(401, "missing or invalid facility token")


def task_digest(task: dict) -> str:
    """Hash of everything that decides what a task does. Any change to it invalidates its grant."""
    keys = ("id", "title", "fixture", "permissions", "allowed_paths", "prompt", "verify", "timeout_s", "verify_timeout_s", "model")
    return hashlib.sha256(json.dumps({k: task.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()


def mint_grant(task: dict, secret: str = None, now: float = None, ttl: int = GRANT_TTL_SEC) -> str:
    secret = secret or _secret("FACILITY_GRANT_SECRET")
    if not secret:
        raise AuthError(403, "write disabled: FACILITY_GRANT_SECRET is not set")
    exp = int((now if now is not None else time.time()) + ttl)
    nonce = secrets.token_hex(8)          # makes every grant unique, so single-use tracking never collides
    mac = hmac.new(secret.encode(), f"{exp}|{nonce}|{task_digest(task)}".encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{nonce}.{mac}"


def check_write_grant(task: dict, grant, now: float = None) -> None:
    secret = _secret("FACILITY_GRANT_SECRET")
    if not secret:
        raise AuthError(403, "write disabled: FACILITY_GRANT_SECRET is not set")
    parts = grant.split(".") if grant else []
    if len(parts) != 3:
        raise AuthError(403, "write task requires a valid write grant")
    exp_s, nonce, mac = parts
    t = now if now is not None else time.time()
    if not exp_s.isdigit() or int(exp_s) < t:
        raise AuthError(403, "write grant expired or malformed")
    if int(exp_s) > t + GRANT_TTL_SEC + GRANT_SKEW_SEC:
        raise AuthError(403, "write grant lifetime exceeds the allowed maximum")
    want = hmac.new(secret.encode(), f"{exp_s}|{nonce}|{task_digest(task)}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, mac):
        raise AuthError(403, "write grant does not match this task")
    for k in [k for k, e in _USED.items() if e < t]:     # forget expired entries
        del _USED[k]
    if mac in _USED:
        raise AuthError(403, "write grant was already used")
    _USED[mac] = int(exp_s)
