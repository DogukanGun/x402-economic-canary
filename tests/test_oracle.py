"""Delivery oracle: the four-way taxonomy and its boundaries.

The distinctions asserted here are the ones the paper's Section 4 draws, and
getting them wrong is exactly how a monitor ends up blind to the attack.
"""

import pytest

from casper_pay_guard.oracle import (
    classify,
    evaluate,
    fault_injection_smoke,
    validate_artifact,
    validate_schema,
)
from casper_pay_guard.x402.facilitator import MeteringFacilitator
from casper_pay_guard.x402.types import PaymentRequirements

SCHEMA = {
    "type": "object",
    "required": ["result", "model"],
    "properties": {"result": {"type": "number"}, "model": {"type": "string"}},
}


@pytest.fixture
def receipt():
    reqs = PaymentRequirements.model_validate(
        {
            "maxAmountRequired": "1000",
            "payTo": "0x000000000000000000000000000000000000dEaD",
            "network": "base-sepolia",
            "scheme": "exact",
        }
    )
    return MeteringFacilitator().settle(payer="0xCA0A17", requirements=reqs, nonce="0xabc")


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
def test_valid_artifact_passes():
    ok, err = validate_artifact('{"result": 1.5, "model": "m"}', SCHEMA)
    assert ok and err is None


@pytest.mark.parametrize(
    "body,reason",
    [
        (None, "no response body"),
        ("", "empty response body"),
        ("   ", "empty response body"),
        ("not json", "body is not valid JSON"),
        ('{"result": 1', "body is not valid JSON"),
    ],
)
def test_unusable_bodies_are_rejected(body, reason):
    ok, err = validate_artifact(body, SCHEMA)
    assert not ok and reason in err


def test_missing_required_key_is_rejected():
    ok, err = validate_artifact('{"result": 1}', SCHEMA)
    assert not ok and "model" in err


def test_wrong_property_type_is_rejected():
    ok, err = validate_artifact('{"result": "not-a-number", "model": "m"}', SCHEMA)
    assert not ok and "result" in err


def test_bool_is_not_a_number():
    """JSON says bool is not a number, even though Python says bool is an int."""
    ok, _ = validate_schema({"result": True, "model": "m"}, SCHEMA)
    assert not ok


def test_bare_key_list_schema_is_supported():
    assert validate_artifact('{"a": 1, "b": 2}', ["a", "b"])[0]
    assert not validate_artifact('{"a": 1}', ["a", "b"])[0]


def test_no_schema_accepts_any_json_object():
    assert validate_artifact('{"anything": 1}', None)[0]
    assert not validate_artifact("[1, 2, 3]", None)[0]


# --------------------------------------------------------------------------- #
# The four-way classifier
# --------------------------------------------------------------------------- #
def test_delivered_needs_2xx_and_a_valid_body():
    assert classify(True, True, 100.0, 200, None, 2000.0) == "delivered"


def test_schema_violation_after_settlement_is_degraded():
    assert classify(False, True, 100.0, 200, None, 2000.0) == "degraded"


def test_empty_body_is_degraded_not_stalled():
    """Section 4: degraded is 'settled with a body that is empty or schema-violating'.

    An empty 200 still cost money, but the provider did answer in time. Only a
    missing 2xx inside the window is a stall.
    """
    r = evaluate(200, "", 100.0, None, _dummy_receipt(), SCHEMA, max_timeout_ms=2000.0)
    assert r.label == "degraded"


def test_non_2xx_after_settlement_is_stalled():
    assert classify(False, True, 100.0, 502, None, 2000.0) == "stalled"


def test_timeout_after_settlement_is_stalled_even_with_a_valid_body():
    """Arriving after the advertised window is a stall regardless of content."""
    assert classify(True, True, 9000.0, 200, None, 2000.0) == "stalled"


def test_paid_but_no_response_at_all_is_stalled():
    """The purest stall: money gone, not even a status line came back."""
    assert classify(False, True, 30000.0, None, None, 2000.0) == "stalled"


def test_exception_after_settlement_is_stalled_not_unreachable():
    """We paid. A dropped connection afterwards is a delivery failure, not a miss."""
    assert classify(False, True, 500.0, None, ConnectionResetError(), 2000.0) == "stalled"


def test_no_settlement_is_unreachable():
    assert classify(False, False, None, None, ConnectionRefusedError(), 2000.0) == "unreachable"
    assert classify(True, False, None, 200, None, 2000.0) == "unreachable"


def test_evaluate_carries_the_receipt_through(receipt):
    r = evaluate(200, '{"result": 1, "model": "m"}', 50.0, None, receipt, SCHEMA)
    assert r.label == "delivered"
    assert r.receipt is not None and r.receipt.price_usdc == 0.001
    assert r.error is None


def test_evaluate_reports_why_it_failed(receipt):
    r = evaluate(200, '{"result": 1}', 50.0, None, receipt, SCHEMA)
    assert r.label == "degraded"
    assert "model" in r.error


def test_fault_injection_smoke_emits_all_four_labels():
    ok, seen = fault_injection_smoke()
    assert ok, seen
    assert set(seen.values()) == {"delivered", "degraded", "stalled", "unreachable"}


def _dummy_receipt():
    reqs = PaymentRequirements.model_validate(
        {"maxAmountRequired": "1000", "payTo": "0xdEaD", "network": "base-sepolia"}
    )
    return MeteringFacilitator().settle(payer="0xCA0A17", requirements=reqs, nonce="0xfeed")
