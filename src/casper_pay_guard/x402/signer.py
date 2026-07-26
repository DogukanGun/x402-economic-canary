"""Isolated EIP-3009 signer (paper Figure 1, layer 3).

Signs a real ``transferWithAuthorization`` EIP-712 typed-data payload with
``eth-account``. This is genuine cryptography: the signature recovers to the
payer address (``tests/test_signer.py`` asserts it). What is stubbed is
*settlement*, not signing — see :mod:`casper_pay_guard.x402.facilitator`.

Key handling: the payer key is read from ``$CANARY_PRIVATE_KEY``. If unset, an
ephemeral key is generated in memory for the process lifetime. The key is never
written to disk and never logged — only the derived address is.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_account.signers.local import LocalAccount

from casper_pay_guard.x402.facilitator import new_nonce
from casper_pay_guard.x402.types import PaymentPayload, PaymentRequirements

#: EIP-712 type definition of EIP-3009 `transferWithAuthorization`.
TRANSFER_WITH_AUTHORIZATION_TYPES = {
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ]
}

#: Authorization validity window, seconds. Short by design: a canary probe that
#: cannot settle promptly should expire rather than linger as a claimable
#: authorization.
DEFAULT_VALIDITY_S = 600


@dataclass(frozen=True)
class NetworkConfig:
    """Chain + asset parameters needed to build the EIP-712 domain."""

    name: str
    chain_id: int
    #: Deployed token contract — the EIP-712 `verifyingContract`.
    asset_address: str
    #: EIP-712 domain name/version as published by the token contract.
    domain_name: str = "USDC"
    domain_version: str = "2"


#: The paper's target network. USDC on Base Sepolia.
BASE_SEPOLIA = NetworkConfig(
    name="base-sepolia",
    chain_id=84532,
    asset_address="0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    domain_name="USDC",
    domain_version="2",
)

NETWORKS: dict[str, NetworkConfig] = {BASE_SEPOLIA.name: BASE_SEPOLIA}


def resolve_network(name: str) -> NetworkConfig:
    """Look up a network by x402 name, defaulting to Base Sepolia."""
    return NETWORKS.get(name, BASE_SEPOLIA)


def load_account(private_key: str | None = None) -> LocalAccount:
    """Load the canary's payer account.

    Order: explicit argument, then ``$CANARY_PRIVATE_KEY``, then a fresh
    ephemeral key. Only ever fund this address with dust — it exists to make
    minimum-price probes, nothing else.
    """
    key = private_key or os.environ.get("CANARY_PRIVATE_KEY")
    if key:
        return Account.from_key(key if key.startswith("0x") else "0x" + key)
    return Account.create()


class Eip3009Signer:
    """Builds and signs the ``X-PAYMENT`` header for one probe."""

    def __init__(
        self,
        account: LocalAccount | None = None,
        network: NetworkConfig | str = BASE_SEPOLIA,
        validity_s: int = DEFAULT_VALIDITY_S,
    ) -> None:
        self.account = account if account is not None else load_account()
        self.network = resolve_network(network) if isinstance(network, str) else network
        self.validity_s = validity_s

    @property
    def address(self) -> str:
        """The payer address. Safe to log; the key is not."""
        return self.account.address

    def domain(self, requirements: PaymentRequirements | None = None) -> dict:
        """EIP-712 domain, preferring the provider's advertised ``extra``."""
        net = self.network
        name, version = net.domain_name, net.domain_version
        verifying = requirements.asset if requirements and requirements.asset else net.asset_address
        if requirements and requirements.extra:
            name = str(requirements.extra.get("name", name))
            version = str(requirements.extra.get("version", version))
        return {
            "name": name,
            "version": version,
            "chainId": net.chain_id,
            "verifyingContract": verifying,
        }

    def build_authorization(
        self, requirements: PaymentRequirements, nonce: str | None = None, now: int | None = None
    ) -> dict:
        """EIP-3009 message fields for a transfer of the advertised amount."""
        now = int(time.time()) if now is None else now
        return {
            "from": self.account.address,
            "to": requirements.pay_to,
            "value": int(requirements.max_amount_required),
            "validAfter": 0,
            "validBefore": now + self.validity_s,
            "nonce": bytes.fromhex((nonce or new_nonce())[2:]),
        }

    def sign(
        self, requirements: PaymentRequirements, nonce: str | None = None, now: int | None = None
    ) -> tuple[PaymentPayload, str]:
        """Sign a transfer authorization. Returns ``(payload, nonce_hex)``."""
        nonce_hex = nonce or new_nonce()
        message = self.build_authorization(requirements, nonce=nonce_hex, now=now)
        signed = self.account.sign_typed_data(
            domain_data=self.domain(requirements),
            message_types=TRANSFER_WITH_AUTHORIZATION_TYPES,
            message_data=message,
        )
        payload = PaymentPayload(
            x402Version=1,
            scheme=requirements.scheme,
            network=requirements.network,
            payload={
                "signature": "0x" + signed.signature.hex().removeprefix("0x"),
                "authorization": {
                    "from": message["from"],
                    "to": message["to"],
                    "value": str(message["value"]),
                    "validAfter": str(message["validAfter"]),
                    "validBefore": str(message["validBefore"]),
                    "nonce": nonce_hex,
                },
            },
        )
        return payload, nonce_hex

    def recover(self, payload: PaymentPayload, requirements: PaymentRequirements) -> str:
        """Recover the signer address from a payload — inverse of :meth:`sign`."""
        auth = payload.payload["authorization"]
        message = {
            "from": auth["from"],
            "to": auth["to"],
            "value": int(auth["value"]),
            "validAfter": int(auth["validAfter"]),
            "validBefore": int(auth["validBefore"]),
            "nonce": bytes.fromhex(auth["nonce"][2:]),
        }
        signable = encode_typed_data(
            domain_data=self.domain(requirements),
            message_types=TRANSFER_WITH_AUTHORIZATION_TYPES,
            message_data=message,
        )
        return Account.recover_message(signable, signature=payload.payload["signature"])
