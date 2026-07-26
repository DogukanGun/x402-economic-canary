# casper_pay_guard

**An economic canary for pay-then-deliver agent markets.** It pays x402 endpoints
their minimum advertised price, verifies what actually comes back, and tells you
whether your next routed call will get anything for its money.

x402 is trustless at the payment layer and trust-maximal at the delivery layer.
There is no escrow, no conditional release, and nothing in the protocol that
guarantees a confirmed settlement produces usable output. A provider can accept
your EIP-3009 authorization, let the transfer confirm, and return a 502, an empty
body, or a payload that violates the schema it advertised. On-chain everything
looks perfect.

Every incumbent trust signal is blind to this:

- **Uptime monitors** see the 402 handshake and score the endpoint healthy — at
  the exact moment it is taking money and stalling.
- **Marketplace ratings** derive trust from settlement volume, which a stalling
  endpoint *accumulates*. The metric meant to build trust is the one the attacker
  farms.
- **Payment policy engines** validate that the payment is authentic. They never
  check whether the artifact exists.

None of them ever observes post-settlement delivery. The only signal a provider
cannot spoof is one obtained under the economics of a real call — so this canary
pays like a customer and checks like a skeptic.

```
                         delivered   settled, and the body validates
   pay minimum price     degraded    settled, body empty or schema-violating
   verify the artifact   stalled     settled, no 2xx inside maxTimeoutSeconds
                         unreachable never got far enough to pay
```

`stalled` is the class the incumbents structurally cannot represent: reaching it
requires having paid *and* knowing what a good body looks like.

## Quick start

```bash
make setup     # pinned venv (python 3.11, numpy 2.4.4, scipy 1.17.1, sklearn 1.8.0)
make test      # 106 tests, incl. real-HTTP e2e and a real MCP stdio session
```

See it catch a thief, end to end:

```bash
make mock-asp                                              # terminal 1
.venv/bin/python -m casper_pay_guard.cli probe \
    http://127.0.0.1:8402/asp/stalled_silent               # terminal 2
```

```
verdict     stalled
http        None
latency     1015.2 ms
gap         1001.6 ms
schema ok   False
paid        0.001 USDC  tx 0x8f2c1ab34de90f12…
WASTED      0.001 USDC — settled, nothing usable back
```

The same endpoint answers an unauthenticated liveness check with a perfectly
well-formed `402`. That is what a monitoring dashboard would be looking at.

## As an MCP tool

```bash
make serve      # stdio MCP server, name: asp-probe
```

Exposes three tools:

| tool | what it does |
|---|---|
| `probe_asp_liveness` | pays, verifies, returns a verdict + settlement receipt |
| `canary_spend_report` | reconciles money spent against artifacts received |
| `canary_metrics` | Prometheus exposition of delivery rate, p95, gap, spend |

`probe_asp_liveness` takes `target_url`, `timeout_s`, `max_price_usdc`,
`expected_output_schema` and `retries`, and returns:

```json
{
  "label": "stalled",
  "http_status": null,
  "latency_ms": 1015.2,
  "settlement_to_response_gap_ms": 1001.6,
  "schema_valid": false,
  "receipt": {"tx_hash": "0x8f2c…", "price_usdc": 0.001, "on_chain": false},
  "error": "no response body"
}
```

Call it before committing a routed payment.

## How it works

Seven layers, following the paper's Figure 1:

| module | role |
|---|---|
| `x402/handshake.py` | GET → 402 → sign → retry, streamed, monotonic clocks |
| `x402/signer.py` | real EIP-3009 / EIP-712 signing via `eth-account` |
| `x402/facilitator.py` | settlement: metering stub (default) or live HTTP facilitator |
| `oracle.py` | schema validation + the four-way classifier |
| `ledger.py` | SQLite reconciliation of settlements against deliveries |
| `features.py` / `predictor.py` | causal per-endpoint features → calibrated P(delivery) |
| `cusum.py` | change detection: flips a verdict on sustained failure |
| `mcp_server.py` | the priced tool surface |

**The measurement that matters** is the settlement-to-response gap, and *when* it
starts. Settlement consumes the authorization; it does not wait on the origin.
If you start the clock when the origin's headers arrive, a provider that takes
your money and then withholds its status line reports a near-zero gap and grades
as healthy — the exact endpoint that is robbing you looks fastest. The gap is
measured from settlement (`x402/handshake.py`), which is why `stalled_silent`
grades correctly.

**Detection** uses a CUSUM over the delivery stream, so a verdict flips after a
couple of paid observations on genuine degradation while raising a false flip
only about once per 1,465 healthy calls. It does not twitch at one bad call, and
it does not un-flip on one good one.

## Results

`make reproduce` regenerates every number the paper published, **bit-for-bit**
(stalled precision 0.994 / recall 0.983, PR-AUC 0.861, Brier 0.0029, ARL 1465,
McNemar p ≈ 1.9e-196 against liveness monitoring). `tests/test_reproduce.py`
asserts all 22 values exactly, not to a tolerance.

`make report` runs every configuration and writes `results/paper_delta.md`, a
printed-vs-computed comparison.

**Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before citing any number here.**
The paper's Methods section describes a different experiment from the one that
produced its results, three of its reported tables had no generating code, and
the sign of its economic claim depends on a routing-policy choice the text and
the code disagree about. All of it is implemented, measured, and written down —
including the two acceptance targets the honest forecast-mode configuration
misses.

## Scope and honesty

- **No money has moved.** Every number is simulated or comes from the local mock
  ASP. No mainnet or testnet transaction has been broadcast.
- **Signing is real, settlement is stubbed.** Signatures are genuine EIP-712 and
  recover to the payer address. The default facilitator mints deterministic
  receipts against a SQLite ledger and never touches a chain.
  `SettlementReceipt.on_chain` is `False` for every receipt this repository can
  produce.
- **`HttpFacilitator` is wired but unexercised.** Set `CASPER_FACILITATOR_URL` to
  route settlement through a real x402 facilitator. It needs a funded payer;
  nothing here tests that path. Taking the canary onto live rails is the paper's
  stated next step, not a claimed result.
- **Key handling.** The payer key comes from `$CANARY_PRIVATE_KEY`, or is
  generated ephemerally in memory. It is never written to disk and never logged —
  only the derived address is. Fund that address with dust and nothing more.

## Layout

```
src/casper_pay_guard/
├── repro/          FROZEN — reproduces the published numbers, do not refactor
├── x402/           types, signer, facilitator, handshake
├── metrics/        bootstrap CIs, off-policy estimators
├── oracle.py       four-way delivery classifier
├── simulate.py     paper Section 5 corpus
├── experiment.py   paper-spec runner
├── ablation.py     Table 1 and Table 3
├── report.py       results/paper_delta.md
├── mock_asp.py     a provider that misbehaves on demand
└── mcp_server.py   the priced tool
```

## Reference

Dogukan Ali Gundogan, *Economic Canaries for Pay-Then-Deliver Agent Markets:
Detecting Settled-but-Stalled x402 Service Providers Before You Route*, 2026.
[Zenodo 10.5281/zenodo.21515696](https://zenodo.org/record/21515696)
