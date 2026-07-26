"""x402 layer: signing, settlement, receipts, ledger reconciliation."""

import pytest
from eth_account import Account

from casper_pay_guard.ledger import Ledger
from casper_pay_guard.x402.facilitator import (
    MeteringFacilitator,
    SettlementError,
    deterministic_tx_hash,
    new_nonce,
)
from casper_pay_guard.x402.signer import BASE_SEPOLIA, Eip3009Signer, load_account
from casper_pay_guard.x402.types import (
    PaymentChallenge,
    PaymentPayload,
    PaymentRequirements,
    atomic_to_usdc,
    usdc_to_atomic,
)

REQS_DICT = {
    "scheme": "exact",
    "network": "base-sepolia",
    "maxAmountRequired": "1000",
    "payTo": "0x000000000000000000000000000000000000dEaD",
    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "maxTimeoutSeconds": 30,
    "extra": {"name": "USDC", "version": "2"},
}


@pytest.fixture
def reqs():
    return PaymentRequirements.model_validate(REQS_DICT)


@pytest.fixture
def signer():
    # Fixed key so the test is deterministic; it holds nothing.
    return Eip3009Signer(
        account=Account.from_key("0x" + "11" * 32), network=BASE_SEPOLIA
    )


# --------------------------------------------------------------------------- #
# Wire types
# --------------------------------------------------------------------------- #
def test_usdc_conversions_round_trip():
    assert usdc_to_atomic(0.001) == 1000
    assert atomic_to_usdc(1000) == 0.001
    assert atomic_to_usdc("2500") == 0.0025


def test_requirements_expose_price_and_timeout(reqs):
    assert reqs.price_usdc == 0.001
    assert reqs.amount_atomic == 1000
    assert reqs.max_timeout_ms == 30_000.0


def test_challenge_picks_the_cheapest_offer():
    """An economic canary pays the minimum advertised price, per Section 4."""
    challenge = PaymentChallenge.model_validate(
        {
            "x402Version": 1,
            "accepts": [
                {**REQS_DICT, "maxAmountRequired": "50000"},
                {**REQS_DICT, "maxAmountRequired": "1000"},
                {**REQS_DICT, "maxAmountRequired": "9000"},
            ],
        }
    )
    assert challenge.cheapest().amount_atomic == 1000


def test_empty_challenge_is_an_error():
    with pytest.raises(ValueError):
        PaymentChallenge.model_validate({"x402Version": 1, "accepts": []}).cheapest()


# --------------------------------------------------------------------------- #
# Signing — real EIP-712, not a stub
# --------------------------------------------------------------------------- #
def test_signature_recovers_to_the_payer(signer, reqs):
    payload, _ = signer.sign(reqs, now=1_700_000_000)
    assert signer.recover(payload, reqs).lower() == signer.address.lower()


def test_signature_is_deterministic_for_a_fixed_nonce_and_time(signer, reqs):
    nonce = new_nonce()
    a, _ = signer.sign(reqs, nonce=nonce, now=1_700_000_000)
    b, _ = signer.sign(reqs, nonce=nonce, now=1_700_000_000)
    assert a.payload["signature"] == b.payload["signature"]


def test_different_nonces_produce_different_signatures(signer, reqs):
    a, _ = signer.sign(reqs, now=1_700_000_000)
    b, _ = signer.sign(reqs, now=1_700_000_000)
    assert a.payload["signature"] != b.payload["signature"]


def test_payment_header_round_trips(signer, reqs):
    payload, nonce = signer.sign(reqs, now=1_700_000_000)
    decoded = PaymentPayload.from_header(payload.to_header())
    assert decoded.payload["authorization"]["nonce"] == nonce
    assert decoded.payload["authorization"]["value"] == "1000"
    assert decoded.scheme == "exact"


def test_authorization_expires(signer, reqs):
    payload, _ = signer.sign(reqs, now=1_700_000_000)
    valid_before = int(payload.payload["authorization"]["validBefore"])
    assert valid_before == 1_700_000_000 + signer.validity_s


def test_ephemeral_key_is_generated_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("CANARY_PRIVATE_KEY", raising=False)
    a, b = load_account(), load_account()
    assert a.address != b.address  # fresh each time, never persisted


# --------------------------------------------------------------------------- #
# Metering facilitator
# --------------------------------------------------------------------------- #
def test_settlement_mints_a_deterministic_receipt(reqs):
    f = MeteringFacilitator()
    r = f.settle(payer="0xCA0A17", requirements=reqs, nonce="0xabc")
    assert r.tx_hash == deterministic_tx_hash("0xabc", 1000)
    assert r.price_usdc == 0.001
    assert r.asset == "USDC"
    assert r.on_chain is False, "the stub must never claim an on-chain settlement"


def test_nonce_replay_is_refused(reqs):
    """x402's replay protection, enforced by the stub too."""
    f = MeteringFacilitator()
    f.settle(payer="0xCA0A17", requirements=reqs, nonce="0xabc")
    with pytest.raises(SettlementError):
        f.settle(payer="0xCA0A17", requirements=reqs, nonce="0xabc")


def test_verify_rejects_a_mismatched_amount(signer, reqs):
    f = MeteringFacilitator()
    payload, _ = signer.sign(reqs, now=1_700_000_000)
    assert f.verify(payload, reqs)

    expensive = PaymentRequirements.model_validate({**REQS_DICT, "maxAmountRequired": "999999"})
    assert not f.verify(payload, expensive)


def test_settlement_is_recorded_in_the_ledger(reqs):
    ledger = Ledger(":memory:")
    f = MeteringFacilitator(ledger=ledger)
    f.settle(payer="0xCA0A17", requirements=reqs, nonce=new_nonce(), target_url="http://asp/x")
    assert ledger.settlement_count() == 1
    assert ledger.total_spent_usdc() == pytest.approx(0.001)


# --------------------------------------------------------------------------- #
# Reconciliation — the settled-vs-delivered join
# --------------------------------------------------------------------------- #
def test_reconcile_separates_money_spent_from_value_received(reqs):
    ledger = Ledger(":memory:")
    f = MeteringFacilitator(ledger=ledger)
    url = "http://asp/x"

    outcomes = ["delivered", "stalled", "stalled", "degraded"]
    for i, label in enumerate(outcomes):
        r = f.settle(payer="0xCA0A17", requirements=reqs, nonce=f"0x{i:064x}", target_url=url)
        ledger.record_delivery(r.tx_hash, label, observed_at=float(i), target_url=url)

    spend = ledger.reconcile(url)
    assert spend.probes == 4
    assert (spend.delivered, spend.stalled, spend.degraded) == (1, 2, 1)
    assert spend.delivery_rate == pytest.approx(0.25)
    assert spend.spent_usdc == pytest.approx(0.004)
    # Three of four probes paid for nothing usable.
    assert spend.wasted_usdc == pytest.approx(0.003)


def test_settlement_without_delivery_counts_as_waste(reqs):
    """A settlement with no matching delivery row is money that vanished."""
    ledger = Ledger(":memory:")
    f = MeteringFacilitator(ledger=ledger)
    url = "http://asp/silent"
    f.settle(payer="0xCA0A17", requirements=reqs, nonce=new_nonce(), target_url=url)

    spend = ledger.reconcile(url)
    assert spend.probes == 1
    assert spend.delivered == 0
    assert spend.wasted_usdc == pytest.approx(0.001)
