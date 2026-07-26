"""Typed x402 wire models (Pydantic v2).

These mirror the x402 specification's shapes: the ``402 Payment Required`` body
carrying one or more :class:`PaymentRequirements`, the base64 ``X-PAYMENT``
header carrying an EIP-3009 :class:`PaymentPayload`, and the
``X-PAYMENT-RESPONSE`` settlement receipt.

``ProbeRequest`` / ``ProbeResult`` are the MCP tool's typed I/O surface, with
exactly the fields the paper's Section 4 names.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: The four-way probe taxonomy. ``stalled`` is the class incumbent monitors
#: structurally cannot represent: it requires having paid.
Label = Literal["delivered", "degraded", "stalled", "unreachable"]

#: USDC has 6 decimals on every network we target.
USDC_DECIMALS = 6


def atomic_to_usdc(atomic: int | str) -> float:
    """Convert atomic USDC units to a decimal amount."""
    return int(atomic) / 10**USDC_DECIMALS


def usdc_to_atomic(amount: float) -> int:
    """Convert a decimal USDC amount to atomic units."""
    return int(round(amount * 10**USDC_DECIMALS))


class PaymentRequirements(BaseModel):
    """One entry of the ``accepts`` array in a 402 challenge body."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    scheme: str = "exact"
    network: str = "base-sepolia"
    max_amount_required: str = Field(alias="maxAmountRequired")
    resource: str = ""
    description: str = ""
    mime_type: str = Field(default="application/json", alias="mimeType")
    #: The provider's *advertised* output schema. Verifying the delivered
    #: artifact against this is what separates "settled" from "delivered".
    output_schema: dict[str, Any] | None = Field(default=None, alias="outputSchema")
    pay_to: str = Field(alias="payTo")
    max_timeout_seconds: int = Field(default=60, alias="maxTimeoutSeconds")
    asset: str = ""
    extra: dict[str, Any] | None = None

    @property
    def price_usdc(self) -> float:
        return atomic_to_usdc(self.max_amount_required)

    @property
    def amount_atomic(self) -> int:
        return int(self.max_amount_required)

    @property
    def max_timeout_ms(self) -> float:
        return float(self.max_timeout_seconds) * 1000.0


class PaymentChallenge(BaseModel):
    """The full body of a ``402 Payment Required`` response."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    x402_version: int = Field(default=1, alias="x402Version")
    error: str | None = None
    accepts: list[PaymentRequirements] = Field(default_factory=list)

    def cheapest(self) -> PaymentRequirements:
        """The minimum advertised price — what an economic canary pays."""
        if not self.accepts:
            raise ValueError("402 challenge carried no `accepts` entries")
        return min(self.accepts, key=lambda r: r.amount_atomic)


class EIP3009Authorization(BaseModel):
    """`transferWithAuthorization` parameters (EIP-3009)."""

    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    value: str
    valid_after: str = Field(alias="validAfter")
    valid_before: str = Field(alias="validBefore")
    nonce: str


class PaymentPayload(BaseModel):
    """What the base64 ``X-PAYMENT`` header decodes to."""

    model_config = ConfigDict(populate_by_name=True)

    x402_version: int = Field(default=1, alias="x402Version")
    scheme: str = "exact"
    network: str = "base-sepolia"
    payload: dict[str, Any]

    def to_header(self) -> str:
        raw = json.dumps(self.model_dump(by_alias=True), separators=(",", ":"))
        return base64.b64encode(raw.encode()).decode()

    @classmethod
    def from_header(cls, header: str) -> PaymentPayload:
        return cls.model_validate(json.loads(base64.b64decode(header)))


class SettlementReceipt(BaseModel):
    """Proof that the transfer authorization was consumed.

    A receipt plus a missing artifact is precisely the settled-but-stalled
    state: the money moved, the deliverable did not.
    """

    model_config = ConfigDict(populate_by_name=True)

    tx_hash: str
    amount_atomic: int
    price_usdc: float
    network: str
    payer: str
    pay_to: str
    nonce: str
    settled_at: float
    asset: str = "USDC"
    #: True when settled by a real facilitator against a chain, False for the
    #: metering stub. Never let this be silently true.
    on_chain: bool = False

    def to_response_header(self) -> str:
        raw = json.dumps(self.model_dump(), separators=(",", ":"))
        return base64.b64encode(raw.encode()).decode()


class ProbeRequest(BaseModel):
    """Typed input of the priced ``probe_asp_liveness`` MCP tool."""

    model_config = ConfigDict(populate_by_name=True)

    target_url: str = Field(description="ASP endpoint to probe with a real paid call")
    timeout_s: float = Field(default=10.0, gt=0, description="Client-side probe timeout")
    max_price_usdc: float = Field(
        default=0.01, ge=0, description="Refuse to pay more than this per probe"
    )
    expected_output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON-Schema subset the artifact must satisfy. Defaults to the "
        "provider's advertised outputSchema from the 402 challenge.",
    )
    retries: int = Field(default=1, ge=0, le=5, description="Handshake retries before giving up")


class ProbeResult(BaseModel):
    """Typed output of the priced ``probe_asp_liveness`` MCP tool.

    Field set is exactly the observable tuple the paper's delivery oracle maps
    to a label: ``(schema_valid, receipt_present, gap_ms, http_status, exception)``.
    """

    model_config = ConfigDict(populate_by_name=True)

    label: Label
    http_status: int | None = None
    latency_ms: float = 0.0
    settlement_to_response_gap_ms: float | None = None
    schema_valid: bool = False
    receipt: SettlementReceipt | None = None
    error: str | None = None
    #: Calibrated P(next paid call delivers), when a fitted predictor is loaded.
    p_delivered: float | None = None
    target_url: str | None = None
