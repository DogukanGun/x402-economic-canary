"""``casper-canary`` — one entry point for every part of the system."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

SUBCOMMANDS = {
    "probe": "Probe a live x402 endpoint with a real paid call",
    "reproduce": "Reproduce the published numbers (frozen path)",
    "paper-spec": "Run the Section 5 experiment as written",
    "ablations": "Compute Table 1 and Table 3",
    "figures": "Regenerate Figures 3 and 4",
    "report": "Write results/paper_delta.md",
    "validate": "Run the 14 validation criteria on every configuration",
    "serve": "Run the MCP stdio server",
    "mock-asp": "Run the misbehaving mock ASP for manual probing",
}


def _cmd_probe(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="casper-canary probe")
    ap.add_argument("url", help="x402 endpoint to probe")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-price", type=float, default=0.01, help="max USDC per probe")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args(argv)

    from casper_pay_guard.x402.facilitator import facilitator_from_env
    from casper_pay_guard.x402.handshake import X402HandshakeClient

    # Env-selected settlement: CASPER_SETTLE=casper-testnet moves real CSPR on
    # Casper Testnet; default remains the free metering stub.
    client = X402HandshakeClient(facilitator=facilitator_from_env())
    result = asyncio.run(
        client.probe(
            args.url, timeout_s=args.timeout, max_price_usdc=args.max_price, retries=args.retries
        )
    )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"verdict     {result.label}")
        print(f"http        {result.http_status}")
        print(f"latency     {result.latency_ms:.1f} ms")
        gap = result.settlement_to_response_gap_ms
        print(f"gap         {gap:.1f} ms" if gap is not None else "gap         —")
        print(f"schema ok   {result.schema_valid}")
        if result.receipt:
            print(f"paid        {result.receipt.price_usdc} USDC  tx {result.receipt.tx_hash[:18]}…")
            if result.label != "delivered":
                print(f"WASTED      {result.receipt.price_usdc} USDC — settled, nothing usable back")
        else:
            print("paid        nothing (never settled)")
        if result.error:
            print(f"error       {result.error}")
    # Non-zero exit when the endpoint took money without delivering, so shell
    # callers can gate a routing decision on it.
    return 0 if result.label == "delivered" else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: casper-canary <command> [options]\n")
        for name, desc in SUBCOMMANDS.items():
            print(f"  {name:12s} {desc}")
        print("\nRun `casper-canary <command> --help` for command options.")
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "probe":
        return _cmd_probe(rest)
    if cmd == "reproduce":
        from casper_pay_guard.repro.run import main as m

        return m()
    if cmd == "paper-spec":
        from casper_pay_guard.experiment import main as m

        return m(rest)
    if cmd == "ablations":
        from casper_pay_guard.ablation import main as m

        return m(rest)
    if cmd == "figures":
        from casper_pay_guard.figures import main as m

        return m(rest)
    if cmd == "report":
        from casper_pay_guard.report import main as m

        return m(rest)
    if cmd == "validate":
        from casper_pay_guard.validate import main as m

        return m(rest)
    if cmd == "serve":
        from casper_pay_guard.mcp_server import main as m

        return m()
    if cmd == "mock-asp":
        from casper_pay_guard.mock_asp import main as m

        return m()

    print(f"unknown command {cmd!r}", file=sys.stderr)
    print(json.dumps(list(SUBCOMMANDS), indent=2), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
