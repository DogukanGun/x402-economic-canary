"""Settlement facilitators (paper Figure 1, layer 3).

Two implementations behind one Protocol:

:class:`MeteringFacilitator`
    The paper's stub. Records the per-call price to the reconciliation ledger,
    stamps ``settled_at``, and mints a deterministic
    ``tx_hash = keccak(nonce ‖ amount_atomic)``. Never touches a chain, never
    spends anything. This reproduces the receipt semantics an agent needs for
    accounting while tests run for free.

:class:`HttpFacilitator`
    Talks to a real x402 facilitator over HTTP (``POST /verify``, ``POST
    /settle``). Wired and typed, but **not exercised** — running it requires a
    funded wallet on live rails, which is the paper's stated next step, not a
    result it claims. See README.
"""

from __future__ import annotations

import os
import secrets
import time
from typing import Any, Protocol, runtime_checkable

from eth_utils import keccak

from casper_pay_guard.ledger import Ledger
from casper_pay_guard.x402.types import (
    PaymentPayload,
    PaymentRequirements,
    SettlementReceipt,
    atomic_to_usdc,
)


class SettlementError(RuntimeError):
    """Raised when a facilitator refuses or fails to settle."""


@runtime_checkable
class Facilitator(Protocol):
    """The settlement interface the handshake client depends on."""

    def verify(self, payload: PaymentPayload, requirements: PaymentRequirements) -> bool:
        """Check the payment authorization is well-formed and acceptable."""
        ...

    def settle(
        self,
        payer: str,
        requirements: PaymentRequirements,
        nonce: str,
        payload: PaymentPayload | None = None,
        target_url: str | None = None,
    ) -> SettlementReceipt:
        """Consume the authorization and return proof it was consumed."""
        ...


def new_nonce() -> str:
    """A fresh 32-byte EIP-3009 nonce."""
    return "0x" + secrets.token_hex(32)


def deterministic_tx_hash(nonce: str, amount_atomic: int) -> str:
    """``keccak(nonce ‖ amount_atomic)`` — the paper's stub receipt hash.

    Deterministic so the same (nonce, amount) always yields the same receipt,
    which makes replay/dedup testable without a chain.
    """
    return "0x" + keccak(f"{nonce}|{amount_atomic}".encode()).hex()


class MeteringFacilitator:
    """Off-chain metering stub: real receipt semantics, zero real money."""

    def __init__(self, db_path: str = ":memory:", ledger: Ledger | None = None) -> None:
        self.ledger = ledger if ledger is not None else Ledger(db_path)
        #: Nonces already consumed — the stub enforces x402's replay protection.
        self._spent_nonces: set[str] = set()

    def verify(self, payload: PaymentPayload, requirements: PaymentRequirements) -> bool:
        """Structural verification only: this stub does not check signatures."""
        auth = (payload.payload or {}).get("authorization") or {}
        if not auth.get("from") or not auth.get("to"):
            return False
        if str(auth.get("value")) != str(requirements.max_amount_required):
            return False
        if payload.scheme != requirements.scheme:
            return False
        nonce = str(auth.get("nonce", ""))
        if nonce in self._spent_nonces:
            return False  # replay
        return True

    def settle(
        self,
        payer: str,
        requirements: PaymentRequirements,
        nonce: str,
        payload: PaymentPayload | None = None,
        target_url: str | None = None,
    ) -> SettlementReceipt:
        if nonce in self._spent_nonces:
            raise SettlementError(f"nonce already consumed: {nonce}")
        self._spent_nonces.add(nonce)

        amount = requirements.amount_atomic
        receipt = SettlementReceipt(
            tx_hash=deterministic_tx_hash(nonce, amount),
            amount_atomic=amount,
            price_usdc=atomic_to_usdc(amount),
            network=requirements.network,
            payer=payer,
            pay_to=requirements.pay_to,
            nonce=nonce,
            settled_at=time.time(),
            asset="USDC",
            on_chain=False,
        )
        self.ledger.record_settlement(receipt, target_url=target_url)
        return receipt


def facilitator_from_env(ledger: Ledger | None = None) -> Any:
    """Pick the settlement facilitator from the environment.

    Precedence (default behavior unchanged):

    1. ``CASPER_SETTLE=casper-testnet`` — :class:`~casper_pay_guard.x402.
       casper_facilitator.CasperFacilitator`: real native CSPR transfers on
       Casper Testnet via the ``casper/settle.mjs`` helper. Needs a funded
       payer PEM (``CASPER_PEM_PATH``) and a recipient
       (``CASPER_PAYEE_PUBKEY``).
    2. ``CASPER_FACILITATOR_URL`` — :class:`HttpFacilitator` against a live
       x402 facilitator (``CASPER_FACILITATOR_KEY`` optional).
    3. otherwise — the free :class:`MeteringFacilitator` stub.
    """
    mode = os.environ.get("CASPER_SETTLE", "").strip().lower()
    if mode == "casper-testnet":
        # Imported lazily: casper_facilitator imports from this module.
        from casper_pay_guard.x402.casper_facilitator import CasperFacilitator

        return CasperFacilitator(ledger=ledger)
    facilitator_url = os.environ.get("CASPER_FACILITATOR_URL")
    if facilitator_url:
        return HttpFacilitator(
            facilitator_url, ledger=ledger, api_key=os.environ.get("CASPER_FACILITATOR_KEY")
        )
    return MeteringFacilitator(ledger=ledger) if ledger is not None else MeteringFacilitator()


class HttpFacilitator:
    """Real x402 facilitator client.

    .. warning::
       **Unexercised.** No number in this repository comes from this class.
       Settling for real needs a funded payer on live rails; that is future
       work in the paper, not a claimed result. It is here so the swap is a
       config change rather than a rewrite.
    """

    def __init__(
        self,
        base_url: str,
        ledger: Ledger | None = None,
        timeout_s: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.ledger = ledger if ledger is not None else Ledger(":memory:")
        self.timeout_s = timeout_s
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}{path}", json=body, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def verify(self, payload: PaymentPayload, requirements: PaymentRequirements) -> bool:
        out = self._post(
            "/verify",
            {
                "x402Version": payload.x402_version,
                "paymentPayload": payload.model_dump(by_alias=True),
                "paymentRequirements": requirements.model_dump(by_alias=True),
            },
        )
        return bool(out.get("isValid", False))

    def settle(
        self,
        payer: str,
        requirements: PaymentRequirements,
        nonce: str,
        payload: PaymentPayload | None = None,
        target_url: str | None = None,
    ) -> SettlementReceipt:
        if payload is None:
            raise SettlementError("HttpFacilitator.settle requires the signed PaymentPayload")
        out = self._post(
            "/settle",
            {
                "x402Version": payload.x402_version,
                "paymentPayload": payload.model_dump(by_alias=True),
                "paymentRequirements": requirements.model_dump(by_alias=True),
            },
        )
        if not out.get("success", False):
            raise SettlementError(str(out.get("errorReason") or "facilitator refused to settle"))

        amount = requirements.amount_atomic
        receipt = SettlementReceipt(
            tx_hash=str(out.get("transaction") or out.get("txHash") or ""),
            amount_atomic=amount,
            price_usdc=atomic_to_usdc(amount),
            network=str(out.get("network") or requirements.network),
            payer=str(out.get("payer") or payer),
            pay_to=requirements.pay_to,
            nonce=nonce,
            settled_at=time.time(),
            asset="USDC",
            on_chain=True,
        )
        self.ledger.record_settlement(receipt, target_url=target_url)
        return receipt
