"""
Synthetic endpoint + probe generator for the x402 economic-canary experiment.

We model a corpus of Agent Service Provider (ASP) endpoints reachable over the
x402 pay-then-deliver protocol. Each endpoint has a latent *health state*:

    healthy   -> settles AND delivers a valid artifact
    stalled   -> SETTLES on-chain (consumes nonce, takes the stablecoin) but
                 returns 5xx / empty / schema-violating bodies  <-- the target
    degraded  -> settles and delivers, but slow / partial
    down      -> unreachable, never settles

The critical property that motivates the canary: a *stalled* endpoint still
answers an unauthenticated liveness probe with a fresh HTTP 402 challenge and
still settles the payment, so free HTTP-liveness monitors and lagging
marketplace reputation cannot see the failure. Only an economic canary that
actually completes the handshake and verifies the returned artifact observes it.

Everything is synthetic numpy — no network, no chain, no torch.

FROZEN: verbatim port of experiment/simulate.py. Do not refactor.
"""

import numpy as np

# Latent health states
STATES = ["healthy", "stalled", "degraded", "down"]

# ---- per-state probe-behaviour parameters -----------------------------------
# p_reach   : endpoint answers at all (TCP/HTTP liveness)
# p_settle  : payment settles on-chain given reachable (nonce consumed, funds taken)
# p_deliver : a VALID artifact comes back given settled
# gap_mu/sd : log-normal mean/sd of settlement->response gap (ms)
STATE_PARAMS = {
    "healthy": dict(p_reach=0.995, p_settle=0.99, p_deliver=0.985, gap_mu=6.0, gap_sd=0.35),
    "stalled": dict(p_reach=0.99, p_settle=0.98, p_deliver=0.04, gap_mu=7.4, gap_sd=0.9),
    "degraded": dict(p_reach=0.97, p_settle=0.95, p_deliver=0.80, gap_mu=7.9, gap_sd=0.5),
    "down": dict(p_reach=0.10, p_settle=0.05, p_deliver=0.30, gap_mu=8.5, gap_sd=1.0),
}


def make_corpus(n_endpoints=60, seed=7):
    """Build an endpoint corpus mimicking an x402-list / Bazaar snapshot."""
    rng = np.random.default_rng(seed)
    # realistic-ish mix: mostly healthy, a meaningful minority stalled
    states = rng.choice(STATES, size=n_endpoints, p=[0.55, 0.22, 0.15, 0.08])
    endpoints = []
    for i, st in enumerate(states):
        endpoints.append(
            dict(
                endpoint_id=f"asp-{i:03d}",
                chain=int(rng.choice([196, 84532])),  # X Layer / Base Sepolia
                price_usdc=float(np.round(rng.uniform(0.0008, 0.0015), 6)),  # ~min advertised
                state=str(st),
                # lagging marketplace reputation: an EMA of past delivery + staleness noise.
                # For freshly-stalled providers reputation is still high (structural blind spot).
                reputation=float(
                    np.clip(
                        {"healthy": 0.93, "stalled": 0.82, "degraded": 0.78, "down": 0.55}[st]
                        + rng.normal(0, 0.06),
                        0,
                        1,
                    )
                ),
                buyers=int(rng.integers(3, 400)),
                settle_vol_30d=float(np.round(rng.uniform(1, 900), 2)),
            )
        )
    return endpoints


def probe(rng, ep):
    """
    Simulate ONE priced canary probe against an endpoint.

    Returns the observable record the canary logs:
      {settled, delivered(ground truth), reachable, http_status,
       artifact_valid, schema_valid, latency_ms, settlement_to_response_gap_ms}
    'delivered' is the ground-truth outcome of the paid call.
    """
    p = STATE_PARAMS[ep["state"]]
    reachable = rng.random() < p["p_reach"]
    if not reachable:
        return dict(
            reachable=False,
            settled=False,
            delivered=False,
            http_status=0,
            artifact_valid=False,
            schema_valid=False,
            latency_ms=float(rng.uniform(1500, 5000)),
            settlement_to_response_gap_ms=0.0,
            returns_402_challenge=False,
        )

    # unauthenticated liveness: reachable endpoints answer with a fresh 402
    returns_402 = True
    settled = rng.random() < p["p_settle"]
    if not settled:
        # paid call did not settle (rare) -> treated as unreachable-ish failure
        return dict(
            reachable=True,
            settled=False,
            delivered=False,
            http_status=int(rng.choice([402, 502])),
            artifact_valid=False,
            schema_valid=False,
            latency_ms=float(rng.uniform(400, 1500)),
            settlement_to_response_gap_ms=0.0,
            returns_402_challenge=returns_402,
        )

    delivered = rng.random() < p["p_deliver"]
    gap = float(np.exp(rng.normal(p["gap_mu"], p["gap_sd"])))
    if delivered:
        artifact_valid = rng.random() < 0.985  # occasional hash mismatch
        schema_valid = rng.random() < 0.99
        http_status = 200
        latency = gap + float(rng.uniform(80, 300))
    else:
        # settled-but-stalled: 5xx / empty / schema-violating body
        artifact_valid = False
        schema_valid = rng.random() < 0.05
        http_status = int(rng.choice([500, 502, 503, 200], p=[0.35, 0.2, 0.15, 0.30]))
        latency = gap + float(rng.uniform(80, 300))
    return dict(
        reachable=True,
        settled=True,
        delivered=bool(delivered),
        http_status=http_status,
        artifact_valid=bool(artifact_valid),
        schema_valid=bool(schema_valid),
        latency_ms=float(latency),
        settlement_to_response_gap_ms=float(gap),
        returns_402_challenge=returns_402,
    )


def generate_probes(endpoints, trials_per_endpoint=120, seed=11):
    """Produce a labeled probe corpus: >=100 trials/endpoint, time-ordered."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in endpoints:
        for t in range(trials_per_endpoint):
            r = probe(rng, ep)
            r.update(
                endpoint_id=ep["endpoint_id"],
                state=ep["state"],
                reputation=ep["reputation"],
                price_usdc=ep["price_usdc"],
                t=t,
            )
            rows.append(r)
    return rows
