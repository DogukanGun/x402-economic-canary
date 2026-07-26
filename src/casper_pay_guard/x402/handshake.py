"""x402 pay-then-deliver handshake client (paper Figure 1, layer 2).

One faithful round trip:

1. ``GET`` the target.
2. On ``402``, decode the challenge and take the **cheapest** advertised
   ``PaymentRequirements`` — an economic canary pays the minimum, so its probes
   cost what a minimum-price production call costs.
3. Sign a real EIP-3009 transfer authorization and base64 it into ``X-PAYMENT``.
4. Retry with the header, streaming the response so the moment settlement is
   confirmed is separable from the moment the body arrives.
5. Grade the artifact with the delivery oracle.

The separation in step 4 is the whole measurement. A stalling provider confirms
settlement quickly and then never produces a body inside
``maxTimeoutSeconds``; the **settlement-to-response gap** ``g`` is what makes
that visible, and it is measured from monotonic clocks, not estimated.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from casper_pay_guard.ledger import Ledger
from casper_pay_guard.oracle import evaluate
from casper_pay_guard.x402.facilitator import Facilitator, MeteringFacilitator, SettlementError
from casper_pay_guard.x402.signer import Eip3009Signer
from casper_pay_guard.x402.types import (
    PaymentChallenge,
    PaymentRequirements,
    ProbeResult,
    SettlementReceipt,
)


class X402HandshakeClient:
    """Completes the priced handshake and returns a graded :class:`ProbeResult`."""

    def __init__(
        self,
        facilitator: Facilitator | None = None,
        signer: Eip3009Signer | None = None,
        ledger: Ledger | None = None,
        user_agent: str = "casper-pay-guard/0.1 (economic-canary)",
    ) -> None:
        self.facilitator = facilitator if facilitator is not None else MeteringFacilitator()
        self.signer = signer if signer is not None else Eip3009Signer()
        self.ledger = ledger if ledger is not None else getattr(self.facilitator, "ledger", None)
        self.user_agent = user_agent

    # ------------------------------------------------------------------ #
    async def probe(
        self,
        target_url: str,
        timeout_s: float = 10.0,
        max_price_usdc: float = 0.01,
        expected_schema: dict[str, Any] | None = None,
        retries: int = 1,
    ) -> ProbeResult:
        """Pay the minimum advertised price once, then grade what came back."""
        last: ProbeResult | None = None
        for attempt in range(retries + 1):
            last = await self._probe_once(
                target_url, timeout_s, max_price_usdc, expected_schema
            )
            # Only a transport-level miss is worth retrying; a settled probe has
            # already cost money and its verdict stands.
            if last.label != "unreachable" or last.receipt is not None:
                break
            if attempt < retries:
                await asyncio.sleep(0.05 * (attempt + 1))
        assert last is not None
        self._record(last)
        return last

    # ------------------------------------------------------------------ #
    async def _probe_once(
        self,
        target_url: str,
        timeout_s: float,
        max_price_usdc: float,
        expected_schema: dict[str, Any] | None,
    ) -> ProbeResult:
        import httpx

        t0 = time.monotonic()
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
                # -- 1. unauthenticated request: expect the 402 challenge -----
                first = await client.get(target_url, headers=headers)
                if first.status_code != 402:
                    # Never got far enough to pay. The paper's taxonomy puts
                    # "no 402" in `unreachable`, not in a delivery class.
                    return evaluate(
                        http_status=first.status_code,
                        body_text=first.text,
                        gap_ms=None,
                        exception=None,
                        receipt=None,
                        schema=expected_schema,
                        latency_ms=(time.monotonic() - t0) * 1000,
                        target_url=target_url,
                    )

                reqs = self._parse_requirements(first)
                schema = expected_schema if expected_schema is not None else reqs.output_schema

                # -- 2. refuse to overpay -------------------------------------
                if reqs.price_usdc > max_price_usdc:
                    return ProbeResult(
                        label="unreachable",
                        http_status=402,
                        latency_ms=(time.monotonic() - t0) * 1000,
                        schema_valid=False,
                        error=(
                            f"advertised price {reqs.price_usdc} USDC exceeds "
                            f"max_price_usdc {max_price_usdc}; did not pay"
                        ),
                        target_url=target_url,
                    )

                # -- 3. sign a real EIP-3009 authorization --------------------
                payload, nonce = self.signer.sign(reqs)
                if not self.facilitator.verify(payload, reqs):
                    return ProbeResult(
                        label="unreachable",
                        http_status=402,
                        latency_ms=(time.monotonic() - t0) * 1000,
                        schema_valid=False,
                        error="facilitator rejected the payment authorization",
                        target_url=target_url,
                    )

                # -- 4. settle, THEN ask for the goods ------------------------
                # Settlement is what consumes the authorization; it does not
                # wait on the origin. Measuring the gap from here — rather than
                # from whenever the origin deigns to send headers — is what
                # makes a provider that settles and then goes quiet visible.
                # A provider that stalls *before* replying would otherwise show
                # a near-zero gap and grade as healthy.
                try:
                    receipt = self.facilitator.settle(
                        payer=self.signer.address,
                        requirements=reqs,
                        nonce=nonce,
                        payload=payload,
                        target_url=target_url,
                    )
                except SettlementError as exc:
                    return ProbeResult(
                        label="unreachable",
                        http_status=402,
                        latency_ms=(time.monotonic() - t0) * 1000,
                        schema_valid=False,
                        error=f"settlement failed: {exc}",
                        target_url=target_url,
                    )

                settled_ts = time.monotonic()
                paid_headers = {**headers, "X-PAYMENT": payload.to_header()}
                budget_s = min(reqs.max_timeout_seconds, timeout_s)

                status, body_text, exc = await self._fetch_paid(
                    client, target_url, paid_headers, budget_s
                )
                gap_ms = (time.monotonic() - settled_ts) * 1000.0

                return evaluate(
                    http_status=status,
                    body_text=body_text,
                    gap_ms=gap_ms,
                    exception=exc,
                    receipt=receipt,
                    schema=schema,
                    latency_ms=(time.monotonic() - t0) * 1000,
                    max_timeout_ms=reqs.max_timeout_ms,
                    target_url=target_url,
                )

        except Exception as exc:  # refused, DNS failure, connect timeout, ...
            return evaluate(
                http_status=None,
                body_text=None,
                gap_ms=None,
                exception=exc,
                receipt=None,
                schema=expected_schema,
                latency_ms=(time.monotonic() - t0) * 1000,
                target_url=target_url,
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    async def _fetch_paid(
        client: Any, target_url: str, headers: dict[str, str], budget_s: float
    ) -> tuple[int | None, str | None, BaseException | None]:
        """Fetch the paid resource within the advertised window.

        Returns ``(status, body_text, exception)``. A ``None`` status means the
        provider took the money and never produced a response at all inside
        ``maxTimeoutSeconds`` — the purest form of the stall. The whole
        request/response is under one budget, headers included, so a provider
        that withholds its status line is caught the same as one that withholds
        its body.
        """

        async def _go() -> tuple[int | None, str | None]:
            async with client.stream("GET", target_url, headers=headers) as resp:
                body = await resp.aread()
                return resp.status_code, (body.decode(errors="replace") if body else None)

        try:
            status, text = await asyncio.wait_for(_go(), timeout=max(budget_s, 0.001))
            return status, text, None
        except asyncio.TimeoutError:
            return None, None, None
        except Exception as exc:
            return None, None, exc

    @staticmethod
    def _parse_requirements(response: Any) -> PaymentRequirements:
        """Pull the cheapest ``accepts`` entry out of a 402 body."""
        data = response.json()
        if "accepts" in data:
            return PaymentChallenge.model_validate(data).cheapest()
        # Some providers return a bare PaymentRequirements object.
        return PaymentRequirements.model_validate(data)

    def _record(self, result: ProbeResult) -> None:
        """Write the delivery outcome next to its settlement in the ledger."""
        if self.ledger is None or result.receipt is None:
            return
        self.ledger.record_delivery(
            tx_hash=result.receipt.tx_hash,
            label=result.label,
            observed_at=time.time(),
            target_url=result.target_url,
            http_status=result.http_status,
            latency_ms=result.latency_ms,
            gap_ms=result.settlement_to_response_gap_ms,
            schema_valid=result.schema_valid,
        )


async def probe_endpoint(
    target_url: str,
    timeout_s: float = 10.0,
    max_price_usdc: float = 0.01,
    expected_schema: dict[str, Any] | None = None,
    retries: int = 1,
    client: X402HandshakeClient | None = None,
) -> ProbeResult:
    """One-shot convenience wrapper around :class:`X402HandshakeClient`."""
    return await (client or X402HandshakeClient()).probe(
        target_url, timeout_s, max_price_usdc, expected_schema, retries
    )


__all__ = [
    "X402HandshakeClient",
    "probe_endpoint",
    "SettlementReceipt",
]
