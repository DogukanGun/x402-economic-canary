"""End-to-end: real HTTP, real EIP-712 signatures, real 402 handshake.

Nothing in this file mocks our own code. A mock ASP is spun up on an ephemeral
port and misbehaves in each of the ways a real provider can, and the canary has
to grade it correctly through the full pay-then-deliver round trip.
"""

import pytest

from casper_pay_guard.ledger import Ledger
from casper_pay_guard.mock_asp import EXPECTED_LABEL, MODES, mock_asp
from casper_pay_guard.x402.facilitator import MeteringFacilitator
from casper_pay_guard.x402.handshake import X402HandshakeClient


@pytest.fixture(scope="module")
def asp():
    with mock_asp() as server:
        yield server


@pytest.fixture
def client():
    ledger = Ledger(":memory:")
    return X402HandshakeClient(facilitator=MeteringFacilitator(ledger=ledger), ledger=ledger)


@pytest.mark.parametrize("mode", MODES)
async def test_each_provider_behaviour_gets_the_right_verdict(asp, client, mode):
    result = await client.probe(asp.url(mode), timeout_s=8.0, max_price_usdc=0.01, retries=0)
    assert result.label == EXPECTED_LABEL[mode], (
        f"{mode}: expected {EXPECTED_LABEL[mode]}, got {result.label} "
        f"(status={result.http_status}, gap={result.settlement_to_response_gap_ms})"
    )


async def test_refused_connection_is_unreachable(asp, client):
    result = await client.probe(asp.closed_port_url(), timeout_s=2.0, retries=0)
    assert result.label == "unreachable"
    assert result.receipt is None, "nothing should settle against a closed port"


async def test_honest_endpoint_delivers_a_verified_artifact(asp, client):
    r = await client.probe(asp.url("honest"), timeout_s=8.0, retries=0)
    assert r.label == "delivered"
    assert r.schema_valid
    assert r.http_status == 200
    assert r.receipt is not None and r.receipt.price_usdc == 0.001
    assert r.error is None


async def test_a_stalling_provider_still_takes_the_money(asp, client):
    """The asymmetry the whole paper is about, demonstrated over the wire."""
    r = await client.probe(asp.url("stalled_silent"), timeout_s=8.0, retries=0)
    assert r.label == "stalled"
    assert r.receipt is not None, "settlement happened"
    assert r.receipt.price_usdc == 0.001, "the money is gone"
    assert not r.schema_valid, "and nothing usable came back"


async def test_the_gap_is_measured_from_settlement_not_from_the_reply(asp, client):
    """A provider that withholds its status line must not look fast.

    Settling only once the origin replies would report a near-zero gap for
    exactly the endpoint that is stealing from you.
    """
    slow = await client.probe(asp.url("stalled_slow"), timeout_s=8.0, retries=0)
    fast = await client.probe(asp.url("honest"), timeout_s=8.0, retries=0)
    assert slow.settlement_to_response_gap_ms > 500
    assert fast.settlement_to_response_gap_ms < 500
    assert slow.settlement_to_response_gap_ms > fast.settlement_to_response_gap_ms * 5


async def test_liveness_monitoring_would_score_every_stall_as_healthy(asp):
    """The incumbent's blind spot, shown directly rather than asserted."""
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as http:
        for mode in ("honest", "degraded", "stalled_5xx", "stalled_slow", "stalled_silent"):
            unpaid = await http.get(asp.url(mode))
            # Every one of them answers with a well-formed 402 challenge.
            assert unpaid.status_code == 402
            assert unpaid.json()["accepts"][0]["maxAmountRequired"] == "1000"


async def test_canary_refuses_to_overpay(asp, client):
    r = await client.probe(asp.url("honest"), timeout_s=5.0, max_price_usdc=0.0000001, retries=0)
    assert r.label == "unreachable"
    assert "exceeds" in r.error
    assert r.receipt is None, "refusing to pay must not settle"


async def test_ledger_reconciles_spend_against_delivery(asp, client):
    for mode in ("honest", "stalled_5xx", "degraded"):
        await client.probe(asp.url(mode), timeout_s=8.0, retries=0)

    assert client.ledger.settlement_count() == 3
    assert client.ledger.total_spent_usdc() == pytest.approx(0.003)

    honest = client.ledger.reconcile(asp.url("honest"))
    assert honest.delivered == 1 and honest.wasted_usdc == pytest.approx(0.0)

    stalled = client.ledger.reconcile(asp.url("stalled_5xx"))
    assert stalled.stalled == 1 and stalled.wasted_usdc == pytest.approx(0.001)


async def test_each_probe_uses_a_fresh_nonce(asp, client):
    """Reusing a nonce would be refused as a replay, so probes must not repeat."""
    a = await client.probe(asp.url("honest"), timeout_s=8.0, retries=0)
    b = await client.probe(asp.url("honest"), timeout_s=8.0, retries=0)
    assert a.receipt.nonce != b.receipt.nonce
    assert a.receipt.tx_hash != b.receipt.tx_hash
