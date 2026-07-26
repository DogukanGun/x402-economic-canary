"""Casper Testnet settlement facilitator.

:class:`CasperFacilitator` implements the same :class:`~casper_pay_guard.x402.
facilitator.Facilitator` Protocol as the metering stub, but ``settle()``
broadcasts a **real native CSPR transfer** on Casper Testnet (chain
``casper-test``, casper-node 2.x "Condor") and returns a receipt carrying the
real on-chain transaction hash with ``on_chain=True``.

The chain interaction is delegated to a small Node.js helper
(``casper/settle.mjs``, casper-js-sdk v5) which builds, signs, and submits the
transfer and prints a single JSON line. This keeps the Python side free of any
Casper cryptography dependency.

Environment configuration
-------------------------
``CASPER_PEM_PATH``        ed25519 PKCS8 PEM of the funded payer
                           (default ``.keys/casper_testnet_ed25519.pem``).
``CASPER_PAYEE_PUBKEY``    recipient Casper public key hex (``01…``). Required.
``CASPER_TRANSFER_MOTES``  motes moved per settlement (default ``2500000000``,
                           the protocol minimum: 2.5 CSPR).
``CASPER_SETTLER_JS``      path to the settle script (default resolved
                           relative to the repo root).
``CASPER_RPC_URL``         JSON-RPC endpoint (script default:
                           ``https://node.testnet.casper.network/rpc``).
``CASPER_CHAIN_NAME``      chain name (script default: ``casper-test``).
``CASPER_SETTLE_DRY_RUN``  ``1`` = build + sign but do not broadcast (the
                           receipt still carries the real local tx hash but
                           ``on_chain=False``).

Semantics note: the receipt's USDC-denominated fields (``amount_atomic``,
``price_usdc``) stay the *advertised* x402 probe price so ledger accounting
remains comparable across facilitators; the value that actually moved is CSPR
motes, reflected in ``asset="CSPR"`` and the on-chain transfer itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from casper_pay_guard.ledger import Ledger
from casper_pay_guard.x402.facilitator import SettlementError
from casper_pay_guard.x402.types import (
    PaymentPayload,
    PaymentRequirements,
    SettlementReceipt,
    atomic_to_usdc,
)

#: Casper's protocol-enforced native-transfer minimum: 2.5 CSPR.
MIN_TRANSFER_MOTES = 2_500_000_000


def _repo_root() -> Path:
    # src/casper_pay_guard/x402/casper_facilitator.py -> repo root is 3 up
    # from the package dir.
    return Path(__file__).resolve().parents[3]


class CasperFacilitator:
    """Settles x402 probes by moving real CSPR on Casper Testnet."""

    def __init__(
        self,
        ledger: Ledger | None = None,
        pem_path: str | None = None,
        payee_pubkey: str | None = None,
        transfer_motes: int | None = None,
        script_path: str | None = None,
        rpc_url: str | None = None,
        chain_name: str | None = None,
        dry_run: bool | None = None,
        timeout_s: float = 90.0,
    ) -> None:
        root = _repo_root()
        self.ledger = ledger if ledger is not None else Ledger(":memory:")
        self.pem_path = pem_path or os.environ.get(
            "CASPER_PEM_PATH", str(root / ".keys" / "casper_testnet_ed25519.pem")
        )
        self.payee_pubkey = payee_pubkey or os.environ.get("CASPER_PAYEE_PUBKEY", "")
        self.transfer_motes = int(
            transfer_motes
            if transfer_motes is not None
            else os.environ.get("CASPER_TRANSFER_MOTES", str(MIN_TRANSFER_MOTES))
        )
        self.script_path = script_path or os.environ.get(
            "CASPER_SETTLER_JS", str(root / "casper" / "settle.mjs")
        )
        self.rpc_url = rpc_url or os.environ.get("CASPER_RPC_URL")
        self.chain_name = chain_name or os.environ.get("CASPER_CHAIN_NAME")
        self.dry_run = (
            dry_run
            if dry_run is not None
            else os.environ.get("CASPER_SETTLE_DRY_RUN", "0") == "1"
        )
        self.timeout_s = timeout_s
        #: Nonces already consumed — same replay protection as the stub.
        self._spent_nonces: set[str] = set()

    # ------------------------------------------------------------------ #
    def verify(self, payload: PaymentPayload, requirements: PaymentRequirements) -> bool:
        """Structural verification, mirroring :class:`MeteringFacilitator`."""
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

    # ------------------------------------------------------------------ #
    def _run_settler(self) -> dict:
        if not self.payee_pubkey:
            raise SettlementError(
                "CASPER_PAYEE_PUBKEY is not set — the Casper facilitator needs a recipient"
            )
        if not Path(self.pem_path).exists():
            raise SettlementError(f"payer key not found: {self.pem_path}")
        if not Path(self.script_path).exists():
            raise SettlementError(f"settle script not found: {self.script_path}")

        cmd = [
            "node",
            self.script_path,
            "--pem", self.pem_path,
            "--to", self.payee_pubkey,
            "--motes", str(self.transfer_motes),
        ]
        if self.chain_name:
            cmd += ["--chain", self.chain_name]
        if self.rpc_url:
            cmd += ["--rpc", self.rpc_url]
        if self.dry_run:
            cmd += ["--dry-run"]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except FileNotFoundError as exc:  # node missing
            raise SettlementError(f"cannot run node: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SettlementError(
                f"casper settle timed out after {self.timeout_s}s"
            ) from exc

        line = (proc.stdout or "").strip().splitlines()
        out: dict = {}
        if line:
            try:
                out = json.loads(line[-1])
            except json.JSONDecodeError:
                out = {}
        if not out.get("ok"):
            detail = out.get("error") or (proc.stderr or "").strip()[-500:] or "unknown error"
            raise SettlementError(f"casper settle failed: {detail}")
        if not out.get("hash"):
            raise SettlementError("casper settle returned no transaction hash")
        return out

    # ------------------------------------------------------------------ #
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

        out = self._run_settler()
        # Only burn the nonce once the transfer is actually out the door.
        self._spent_nonces.add(nonce)

        broadcast = not bool(out.get("dryRun"))
        amount = requirements.amount_atomic
        receipt = SettlementReceipt(
            tx_hash=str(out["hash"]),
            amount_atomic=amount,
            price_usdc=atomic_to_usdc(amount),
            network=str(out.get("chain") or self.chain_name or "casper-test"),
            payer=str(out.get("from") or payer),
            pay_to=self.payee_pubkey,
            nonce=nonce,
            settled_at=time.time(),
            asset="CSPR",
            on_chain=broadcast,
        )
        self.ledger.record_settlement(receipt, target_url=target_url)
        return receipt
