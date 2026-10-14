# Facility task loop (Milestone 1) — security model, setup, limits

`POST /api/tasks/run` runs one task: isolated workspace -> agent -> verification by command -> evidence receipt.
Code: `backend/app/facility.py` (loop), `facility_sandbox.py` (OS locks, env scrubbing, verify dir), `facility_auth.py` (token + grants).

## Fails closed by default
Nothing runs until you configure it. Missing or too-short (<16 chars) secrets disable the feature.

| Variable | Purpose | Default |
|---|---|---|
| `FACILITY_API_TOKEN` | Bearer token required on every call (`Authorization: Bearer ...`) | unset = endpoint returns 503 |
| `FACILITY_GRANT_SECRET` | Signs write grants (`X-Facility-Write-Grant: <exp>.<nonce>.<hmac>`), bound to one task's exact content (id, title, fixture, permissions, allowed_paths, prompt, verify, timeouts, model), single use, valid at most 10 minutes | unset = write tasks refused (403) |
| `FACILITY_EVIDENCE_SECRET` | Signs each receipt's verification claims (`attestation`: HMAC-SHA256 over `v1|run_id|task_id|status|report_ok|verification_trust`), shared with empire-command-center's `ORCHESTRA_EVIDENCE_SECRET` | unset = receipts carry `attestation: null` and the approval gate refuses them (fail closed) |
| `FACILITY_ENABLE_PROVIDER_EXEC` | `1` allows the real Claude CLI to run | off = 503 |
| `FACILITY_ENABLE_WRITE_EXEC` | `1` additionally allows write tasks with the real CLI. Read the limitations below first | off |
| `FACILITY_MAX_CONCURRENT` / `FACILITY_MAX_PER_HOUR` | Admission limits protecting the subscription | `1` / `20` (429 when exceeded) |
| `FACILITY_WORKSPACE_ROOT` | Only directory the loop creates and deletes inside | `%TEMP%\orchestra-facility` |
| `FACILITY_RECEIPT_DIR` | Evidence receipts (never overwritten; named `<task>-<run_id>.json`) | `backend/receipts` (git-ignored) |
| `FACILITY_FIXTURE_ROOT` | Directory of fixture projects | `backend/fixtures/facility` |
| `CLAUDE_BIN` | Path to `claude.exe` (see README) | `claude` on PATH |

Generate secrets with a password manager or `python -c "import secrets;print(secrets.token_hex(32))"`; never commit them.

## Dependencies
Python 3.11+, packages in `backend/requirements.txt` (fastapi, uvicorn, pydantic; `pytest` is required at runtime because the verifier runs it;
`httpx` is needed only to run the test suite). Windows: `icacls` and `whoami` (built in). The proof script in `empire-os-wired` also needs Node + `tsx`
(from `empire-command-center`) for the mission board.

## What is enforced by code
- **Trusted tests live outside the agent workspace** in a sibling `-trusted` directory that is OS-locked (Windows ACL deny / POSIX chmod) before the agent starts and until verification ends.
- **Read-only tasks:** the whole workspace is OS-locked; any change that still occurs is a `permission_violation`.
- **Write tasks:** need an explicit `allowed_paths` list (cannot name tests or test config). Every existing file not on the list is OS-locked. Anything changed outside it is a `permission_violation`; touching tests or test config is a `protected_file_violation`.
- **Verification** runs in a fresh directory: pristine fixture files + the agent's output for `allowed_paths` only + the trusted tests, with an allow-listed environment (no keys or tokens). Acceptance requires exit code 0 **and** a well-formed junit report with at least one test and no failures, errors or skips.
- **Fails closed if the OS lock cannot be applied** (the agent is not started).
- Endpoint auth, write grants, concurrency and hourly caps, one overall deadline (`timeout_s` covers agent + verification), process-tree kill on timeout, cleanup of only facility-owned directories, collision-safe receipts, redaction of secret-looking environment values from receipts, and the actual resolved model recorded in `resolved_model`.
- Approval is separate from execution and verification: only a `verified` receipt can ever be offered for approval (also enforced by the mission board server for orders created with `requires_verification`).
- Receipts carry `attestation`, an HMAC over the verification claims (`backend/app/facility_attestation.py`). The mission board accepts evidence only when the attestation is valid, the status is `verified`, `report_ok` is true, and `verification_trust` is `pristine_code_only`. A same-user write receipt is attested with its real trust level and is refused. The attestation proves where the claims came from. It does not make a write-task result trustworthy.

## What is NOT enforced (unsupported on this Windows host — do not rely on it)
- **Network access** by the agent CLI or by agent-written code run by the verifier is not blocked.
- **Subprocess creation** by those same processes is not blocked.
- **Reads and writes outside the workspace** by code running as your Windows user are not blocked (only the workspace and trusted directories are locked).
- **The Claude CLI's own tool permissions** (`--tools`, `--disallowedTools`, `--setting-sources`, `--strict-mcp-config`, ...) are passed as defence in depth. Whether the installed CLI honours them has NOT been proven without a live model call.
- **Cancellation is not supported**: there is no cancel endpoint and a client disconnect does not stop a run. A run ends by finishing or by its deadline.
- A verifier can run agent-written code (allowed paths). That code can try to forge results from inside the verifier process; the junit-report check and env scrubbing make this harder, not impossible.

Because of this, real execution is off by default. Strong isolation requires a dedicated low-privilege Windows account, a container/VM or Windows Sandbox, with firewall rules; run the server under that account before enabling `FACILITY_ENABLE_WRITE_EXEC`.

## Fixture manifest
`backend/fixtures/facility/MANIFEST.json` pins the exact contents (line endings normalised) of every shipped fixture. If a fixture differs from it, the run is refused. Regenerate it only when you intentionally change a fixture, and review the diff.

## Known gaps found by the red-team review (not fixed; do not rely on them)
- **Verification can be forged by code the verifier runs (write tasks).** `allowed_paths` code runs inside the pytest process as your Windows user; it can read the report path from `sys.argv`, patch pytest internals, or kill the run. Receipts therefore carry `verification_trust: "agent_code_executed_same_user"` for write tasks (and `"pristine_code_only"` for read-only tasks, where only trusted code runs). Treat a write-task `verified` as "no obvious failure", never as proof. Fixing this needs a separate OS account, container or VM.
- ACL locks are tripwires, not boundaries: the file owner can always reset an ACL. Tampering is detected afterwards (hash comparison, trusted tests are re-written from memory) rather than prevented.
- Deleting or renaming a locked non-allowed file in a write workspace is not blocked on Windows (the parent directory right overrides); it is detected and reported as `permission_violation`.
- A detached child process started by agent code can outlive the run (Job Objects are not used); a workspace it holds open may fail cleanup (`cleanup_error` in the receipt).
- The verifier receives the user's Python user-site directory on `PYTHONPATH` (pytest lives there on this machine), so a poisoned user-site would persist across runs.
- Symlink handling is only unit-tested where Windows allows creating symlinks (skipped on this machine); junctions and hardlinks are detected and tested.
- Grants are tracked in memory per server process: restarting the server forgets used grants (they still expire in 10 minutes); multiple workers multiply the rate limits.
- The lessons endpoints (`/api/lessons`) are not behind the facility token. No code can set `verified=1`, so nothing is injected today; add auth and sanitisation before building a verification pipeline.
