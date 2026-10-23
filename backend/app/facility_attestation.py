"""Signed verification claims for facility receipts, consumed by empire-command-center.

The receipt is signed only over the claims a downstream approval gate needs. The canonical message format is
fixed and must match empire-command-center server/missionEvidence.ts exactly:
    "v1|<run_id>|<task_id>|<status>|<report_ok>|<verification_trust>"   (report_ok is lowercase "true"/"false")

Signing proves the claims came from this server's verifier under the shared FACILITY_EVIDENCE_SECRET. It does NOT
make a write-task result trustworthy: verification_trust is included so the consumer can refuse
"agent_code_executed_same_user" receipts. Unset secret = no attestation (consumers fail closed).
"""
import hashlib
import hmac
import os
from typing import Any, Dict, Optional

MIN_SECRET_LEN = 32
VERSION = "v1"


def _secret() -> Optional[str]:
    value = os.environ.get("FACILITY_EVIDENCE_SECRET", "")
    return value if len(value) >= MIN_SECRET_LEN else None


def canonical_message(receipt: Dict[str, Any]) -> str:
    verify = receipt.get("verify") if isinstance(receipt.get("verify"), dict) else {}
    report_ok = "true" if verify.get("report_ok") is True else "false"
    fields = [VERSION, str(receipt.get("run_id")), str(receipt.get("task_id")), str(receipt.get("status")),
              report_ok, str(receipt.get("verification_trust"))]
    return "|".join(fields)


def attest(receipt: Dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_message(receipt).encode("utf-8"), hashlib.sha256).hexdigest()


def attestation_for(receipt: Dict[str, Any]) -> Optional[str]:
    """Attestation for the receipt's final claims, or None when no evidence secret is configured."""
    secret = _secret()
    return attest(receipt, secret) if secret else None
