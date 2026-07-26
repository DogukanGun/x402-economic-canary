"""MCP smoke test — a real stdio client session against the shipped server.

The paper's validation criterion 14 is "the MCP smoke test passed". The original
version called a pure function in-process. This one launches the actual server
as a subprocess, speaks the MCP protocol over stdio, and drives the priced tool
against a live mock ASP — so the typed schema, the transport and the tool
registration are all exercised, not just the classifier.
"""

import json
import sys
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from casper_pay_guard.mock_asp import EXPECTED_LABEL, mock_asp

SERVER_CMD = [sys.executable, "-m", "casper_pay_guard.mcp_server"]


def _payload(result):
    """MCP returns structured content plus a text fallback; prefer the former."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    return json.loads(result.content[0].text)


@pytest.fixture(scope="module")
def asp():
    with mock_asp() as server:
        yield server


@asynccontextmanager
async def session():
    """Open a real MCP stdio session against the shipped server.

    Deliberately not a pytest fixture: `stdio_client` opens an anyio cancel
    scope that must be exited by the same task that entered it, and a generator
    fixture does not guarantee that.
    """
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            yield s


async def test_server_advertises_the_priced_tool():
    async with session() as s:
        tools = {t.name: t for t in (await s.list_tools()).tools}

    assert "probe_asp_liveness" in tools
    props = tools["probe_asp_liveness"].inputSchema["properties"]
    # The typed input surface Section 4 specifies.
    for field in ("target_url", "timeout_s", "max_price_usdc", "expected_output_schema", "retries"):
        assert field in props, f"missing input field {field}"


async def test_tool_returns_all_four_labels_under_fault_injection(asp):
    """The four-way taxonomy, end to end through the MCP transport."""
    async with session() as s:
        seen = {}
        for mode in ("honest", "degraded", "stalled_5xx", "no_402"):
            result = await s.call_tool(
                "probe_asp_liveness",
                {"target_url": asp.url(mode), "timeout_s": 8.0, "retries": 0},
            )
            seen[mode] = _payload(result)["label"]

        assert seen == {m: EXPECTED_LABEL[m] for m in seen}
        assert set(seen.values()) == {"delivered", "degraded", "stalled", "unreachable"}


async def test_settled_probes_carry_a_receipt(asp):
    async with session() as s:
        for mode in ("honest", "stalled_5xx"):
            out = _payload(
                await s.call_tool(
                    "probe_asp_liveness", {"target_url": asp.url(mode), "timeout_s": 8.0, "retries": 0}
                )
            )
            assert out["receipt"] is not None, f"{mode}: settled probe must return a receipt"
            assert out["receipt"]["price_usdc"] == 0.001
            assert out["receipt"]["on_chain"] is False


async def test_unreachable_probe_settles_nothing(asp):
    async with session() as s:
        out = _payload(
            await s.call_tool(
                "probe_asp_liveness",
                {"target_url": asp.closed_port_url(), "timeout_s": 2.0, "retries": 0},
            )
        )
        assert out["label"] == "unreachable"
        assert out["receipt"] is None


async def test_caller_supplied_schema_overrides_the_advertised_one(asp):
    """An agent can demand more than the provider advertised."""
    async with session() as s:
        strict = {"type": "object", "required": ["result", "model", "definitely_absent"]}
        out = _payload(
            await s.call_tool(
                "probe_asp_liveness",
                {
                    "target_url": asp.url("honest"),
                    "timeout_s": 8.0,
                    "retries": 0,
                    "expected_output_schema": strict,
                },
            )
        )
        assert out["label"] == "degraded"
        assert not out["schema_valid"]


async def test_spend_report_reconciles_settlement_against_delivery(asp):
    async with session() as s:
        for mode in ("honest", "stalled_5xx"):
            await s.call_tool(
                "probe_asp_liveness", {"target_url": asp.url(mode), "timeout_s": 8.0, "retries": 0}
            )

        report = _payload(await s.call_tool("canary_spend_report"))
        assert report["settlements"] >= 2
        assert report["total_spent_usdc"] > 0
        stalled = report["endpoints"][asp.url("stalled_5xx")]
        assert stalled["stalled"] >= 1
        assert stalled["wasted_usdc"] > 0, "a stall must show up as wasted money"


async def test_metrics_endpoint_exports_the_stall_signal(asp):
    async with session() as s:
        await s.call_tool(
            "probe_asp_liveness", {"target_url": asp.url("honest"), "timeout_s": 8.0, "retries": 0}
        )
        result = await s.call_tool("canary_metrics")
        text = result.content[0].text
        for metric in (
            "casper_probes_total",
            "casper_settlement_to_response_gap_ms",
            "casper_delivery_rate",
            "casper_stalled_rejection_rate",
        ):
            assert metric in text
