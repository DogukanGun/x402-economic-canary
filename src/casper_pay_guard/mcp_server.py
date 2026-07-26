"""The priced ``probe_asp_liveness`` MCP tool (paper Figure 1, layer 7).

A stdio MCP server exposing one tool that spends money to earn its answer. Point
it at an x402 endpoint and it completes the full pay-then-deliver handshake at
the minimum advertised price, verifies the returned artifact against the
provider's advertised schema, and hands back a verdict plus the settlement
receipt.

    delivered   — settled, and the body validates
    degraded    — settled, but the body is empty or violates the schema
    stalled     — settled, and no 2xx arrived inside maxTimeoutSeconds
    unreachable — never got far enough to pay

Call this before committing a routed payment. A green liveness dashboard cannot
distinguish the first verdict from the third; that is the entire point.

Settlement uses the metering facilitator by default: real receipts, real
accounting, no real money. Set ``CASPER_FACILITATOR_URL`` to settle through a
live x402 facilitator instead — that path needs a funded payer and is
unexercised here.

Run:  ``python -m casper_pay_guard.mcp_server``
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from casper_pay_guard import observability
from casper_pay_guard.ledger import Ledger
from casper_pay_guard.x402.facilitator import HttpFacilitator, MeteringFacilitator
from casper_pay_guard.x402.handshake import X402HandshakeClient
from casper_pay_guard.x402.signer import Eip3009Signer
from casper_pay_guard.x402.types import ProbeResult

mcp = FastMCP("asp-probe")

_logger = observability.get_logger()
_client: X402HandshakeClient | None = None


def _ledger_path() -> str:
    return os.environ.get("CASPER_LEDGER_PATH", ":memory:")


def get_client() -> X402HandshakeClient:
    """Build the handshake client once, from the environment."""
    global _client
    if _client is not None:
        return _client

    ledger = Ledger(_ledger_path())
    facilitator_url = os.environ.get("CASPER_FACILITATOR_URL")
    if facilitator_url:
        # Live rails. Requires a funded payer; nothing in this repo exercises it.
        facilitator: Any = HttpFacilitator(
            facilitator_url, ledger=ledger, api_key=os.environ.get("CASPER_FACILITATOR_KEY")
        )
        _logger.warning(
            "live facilitator configured — probes will attempt real settlement",
            extra={"extra_fields": {"facilitator_url": facilitator_url}},
        )
    else:
        facilitator = MeteringFacilitator(ledger=ledger)

    signer = Eip3009Signer()
    _logger.info(
        "canary ready",
        extra={"extra_fields": {"payer": signer.address, "on_chain": bool(facilitator_url)}},
    )
    _client = X402HandshakeClient(facilitator=facilitator, signer=signer, ledger=ledger)
    return _client


@mcp.tool()
async def probe_asp_liveness(
    target_url: str = Field(description="x402 endpoint to probe with a real paid call"),
    timeout_s: float = Field(default=10.0, gt=0, description="Client-side probe timeout"),
    max_price_usdc: float = Field(
        default=0.01, ge=0, description="Refuse to pay more than this for one probe"
    ),
    expected_output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON-Schema subset the artifact must satisfy (`type`, `required`, "
            "`properties[].type`). Defaults to the provider's advertised outputSchema."
        ),
    ),
    retries: int = Field(default=1, ge=0, le=5, description="Handshake retries before giving up"),
) -> ProbeResult:
    """Pay an x402 endpoint its minimum price, verify what comes back, and grade it.

    Returns one of `delivered`, `degraded`, `stalled`, `unreachable`, with the
    settlement receipt, the HTTP status, and the settlement-to-response gap that
    separates a provider which paid out from one which merely got paid.
    """
    client = get_client()
    result = await client.probe(
        target_url=target_url,
        timeout_s=timeout_s,
        max_price_usdc=max_price_usdc,
        expected_schema=expected_output_schema,
        retries=retries,
    )
    observability.record(result)
    observability.log_probe(_logger, result)
    return result


@mcp.tool()
def canary_spend_report() -> dict[str, Any]:
    """Reconcile what the canary has spent against what it actually received.

    Returns per-endpoint probe counts, delivery rate and `wasted_usdc` — money
    that settled without producing a usable artifact.
    """
    return get_client().ledger.summary()


@mcp.tool()
def canary_metrics() -> str:
    """Prometheus text exposition of delivery rate, p95 latency, gap and spend."""
    return observability.snapshot()


def main() -> int:
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
