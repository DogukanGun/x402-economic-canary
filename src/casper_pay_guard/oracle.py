"""Delivery oracle and four-way classifier (paper Section 4).

The oracle answers the one question no incumbent monitor asks: *did the thing I
paid for actually arrive?* It maps the observable tuple

    (schema_valid, receipt_present, gap_ms, http_status, exception)

onto exactly one label:

===============  =============================================================
``delivered``    settlement confirmed and the body validates against the
                 advertised schema
``degraded``     settlement confirmed, but the body is empty or violates the
                 schema
``stalled``      settlement confirmed on-chain, but no 2xx arrived inside
                 ``maxTimeoutSeconds`` — the money-eating attack
``unreachable``  never got far enough to pay (connection failure, no 402, ...)
===============  =============================================================

``stalled`` is the class that a liveness monitor and a reputation counter
cannot represent, because reaching it requires having paid *and* knowing what a
good body looks like.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from casper_pay_guard.x402.types import Label, ProbeResult, SettlementReceipt

#: Fallback settlement-to-response gap (ms) above which a *paid* call is judged
#: stalled, used when the provider advertised no ``maxTimeoutSeconds``.
DEFAULT_STALL_GAP_MS = 2000.0

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, expected: str) -> bool:
    py = _JSON_TYPES.get(expected)
    if py is None:
        return True  # unknown type keyword: do not fail the provider on it
    if expected in ("number", "integer") and isinstance(value, bool):
        return False  # bool is an int subclass in Python; JSON says otherwise
    return isinstance(value, py)


def validate_schema(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate ``data`` against a small, dependency-free JSON-Schema subset.

    Supports ``type``, ``required`` and per-property ``type`` — enough to catch
    the schema-violating bodies a stalling provider returns, without pulling in
    a full validator.
    """
    expected_type = schema.get("type")
    if expected_type and not _type_ok(data, expected_type):
        return False, f"expected type {expected_type}, got {type(data).__name__}"

    if isinstance(data, dict):
        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            return False, f"missing keys: {missing}"
        for key, sub in (schema.get("properties") or {}).items():
            if key not in data or not isinstance(sub, dict):
                continue
            sub_type = sub.get("type")
            if sub_type and not _type_ok(data[key], sub_type):
                return False, f"property {key!r}: expected {sub_type}, got {type(data[key]).__name__}"
    return True, None


def validate_artifact(
    body_text: str | None,
    schema: dict[str, Any] | Sequence[str] | None,
) -> tuple[bool, str | None]:
    """Return ``(schema_valid, error)`` for a returned artifact.

    ``schema`` may be a JSON-Schema subset dict, a bare sequence of required key
    names, or ``None`` (any non-empty JSON object passes).
    """
    if body_text is None:
        return False, "no response body"
    if not body_text.strip():
        return False, "empty response body"
    try:
        data = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        return False, "body is not valid JSON"

    if schema is None:
        if not isinstance(data, dict):
            return False, "artifact is not a JSON object"
        return True, None
    if isinstance(schema, dict):
        return validate_schema(data, schema)
    # bare sequence of required keys
    if not isinstance(data, dict):
        return False, "artifact is not a JSON object"
    missing = [k for k in schema if k not in data]
    if missing:
        return False, f"missing keys: {missing}"
    return True, None


def classify(
    schema_valid: bool,
    receipt_present: bool,
    gap_ms: float | None,
    http_status: int | None,
    exception: BaseException | None,
    max_timeout_ms: float = DEFAULT_STALL_GAP_MS,
) -> Label:
    """Map one probe's observable tuple to exactly one of the four labels."""
    if not receipt_present:
        # The handshake failed before settlement: refused, no 402, declined to
        # pay. No money moved, so this is not a delivery failure.
        return "unreachable"

    # From here the authorization has been consumed — the money is gone whatever
    # happens next, including if the connection then died on us.
    if http_status is None or exception is not None:
        return "stalled"

    # Payment settled. Did a 2xx arrive at all, and inside the advertised window?
    got_2xx = 200 <= http_status < 300
    timed_out = gap_ms is not None and gap_ms >= max_timeout_ms
    if not got_2xx or timed_out:
        return "stalled"

    # A 2xx arrived in time. Is the artifact actually usable?
    return "delivered" if schema_valid else "degraded"


def evaluate(
    http_status: int | None,
    body_text: str | None,
    gap_ms: float | None,
    exception: BaseException | None,
    receipt: SettlementReceipt | None,
    schema: dict[str, Any] | Sequence[str] | None = None,
    latency_ms: float = 0.0,
    max_timeout_ms: float = DEFAULT_STALL_GAP_MS,
    target_url: str | None = None,
) -> ProbeResult:
    """Run the oracle + classifier over one probe's raw signals."""
    if exception is not None:
        # An exception after settlement is a stall, not an unreachable endpoint:
        # we paid, and nothing usable came back.
        return ProbeResult(
            label=classify(False, receipt is not None, gap_ms, http_status, exception, max_timeout_ms),
            http_status=http_status,
            latency_ms=latency_ms,
            settlement_to_response_gap_ms=gap_ms,
            schema_valid=False,
            receipt=receipt,
            error=f"{type(exception).__name__}: {exception}",
            target_url=target_url,
        )

    schema_valid, err = validate_artifact(body_text, schema)
    label = classify(schema_valid, receipt is not None, gap_ms, http_status, exception, max_timeout_ms)
    return ProbeResult(
        label=label,
        http_status=http_status,
        latency_ms=latency_ms,
        settlement_to_response_gap_ms=gap_ms,
        schema_valid=schema_valid,
        receipt=receipt,
        error=None if label == "delivered" else err,
        target_url=target_url,
    )


# --------------------------------------------------------------------------- #
# Fault-injection smoke: all four labels, no network, no money.
# --------------------------------------------------------------------------- #
def _demo_receipt() -> SettlementReceipt:
    from casper_pay_guard.x402.facilitator import MeteringFacilitator
    from casper_pay_guard.x402.types import PaymentRequirements

    reqs = PaymentRequirements.model_validate(
        {
            "maxAmountRequired": "1000",
            "payTo": "0x000000000000000000000000000000000000dEaD",
            "network": "base-sepolia",
            "scheme": "exact",
            "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        }
    )
    return MeteringFacilitator(db_path=":memory:").settle(
        payer="0x0000000000000000000000000000000000CA0A17",
        requirements=reqs,
        nonce="0xdeadbeef",
    )


def fault_injection_smoke() -> tuple[bool, dict[str, str]]:
    """Drive the oracle through all four labels deterministically.

    Returns ``(ok, {expected_label: observed_label})``.
    """
    receipt = _demo_receipt()
    schema = {"type": "object", "required": ["result"], "properties": {"result": {"type": "number"}}}
    cases: Iterable[tuple[str, ProbeResult]] = [
        # settled, 200, schema-valid body
        ("delivered", evaluate(200, '{"result": 42}', 120.0, None, receipt, schema)),
        # settled, 200, schema-violating body
        ("degraded", evaluate(200, '{"oops": true}', 130.0, None, receipt, schema)),
        # settled, body dropped after status — the money-eating attack
        ("stalled", evaluate(200, None, 9000.0, None, receipt, schema)),
        # never settled
        ("unreachable", evaluate(None, None, None, ConnectionRefusedError("refused"), None, schema)),
    ]
    seen = {expected: res.label for expected, res in cases}
    ok = all(seen[k] == k for k in seen)
    return ok, seen
