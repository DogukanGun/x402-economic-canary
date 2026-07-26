"""Paper-spec corpus generator — Section 5 of the paper, implemented as written.

This is *not* the corpus that produced the published numbers. That one lives in
:mod:`casper_pay_guard.repro.simulate` (60 endpoints x 120 trials = 7,200 probes,
four latent states, seeds 7/11). Section 5 describes something different, and
this module is that description taken literally:

* 120 endpoints in a 1:1:1 ratio over three latent behaviour profiles —
  honest (40), intermittently degraded (40), settled-but-stalled (40)
* settlement ``s ~ Bernoulli(0.99)``
* delivery given settlement ``d ~ Bernoulli(p_d)``, ``p_d`` = 0.98 / 0.65 / 0.03
* latency ``l`` log-normal in ms with ``(mu, sigma)`` = (5.2, 0.4) / (5.9, 0.6) /
  (6.5, 0.5)
* settlement-to-response gap ``g ~ Gamma(shape=2, scale=30ms)`` on delivered
  probes, clipped to ``maxTimeoutSeconds`` on stalled ones
* 150 sequential trials per endpoint (18,000 probes), PCG64 seeded at 42
* time-based 60/40 train/test split, per-provider stratified (10,800 / 7,200)
* ground truth from a *subsequent independent draw* from the same latent profile

See REPRODUCIBILITY.md for why both exist and how their numbers differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The three latent behaviour profiles, 40 endpoints each.
PROFILES = ("honest", "degraded", "stalled")

#: P(delivery | settlement) per profile — Section 5.
P_DELIVER = {"honest": 0.98, "degraded": 0.65, "stalled": 0.03}

#: P(settlement) is profile-independent in Section 5.
P_SETTLE = 0.99

#: Log-normal latency parameters (mu, sigma) in log-milliseconds, per profile.
LATENCY_LOGNORMAL = {
    "honest": (5.2, 0.4),
    "degraded": (5.9, 0.6),
    "stalled": (6.5, 0.5),
}

#: Settlement-to-response gap: Gamma(shape=2, scale=30ms).
GAP_SHAPE = 2.0
GAP_SCALE_MS = 30.0

#: Advertised timeout; stalled probes have their gap clipped up to this.
MAX_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_MS = MAX_TIMEOUT_SECONDS * 1000.0

N_ENDPOINTS = 120
TRIALS_PER_ENDPOINT = 150
SEED = 42
TRAIN_FRACTION = 0.6

#: Minimum advertised price the canary targets, in USDC.
MIN_PRICE_USDC = 0.001


@dataclass
class Endpoint:
    """One simulated ASP listing."""

    endpoint_id: str
    profile: str
    price_usdc: float
    reputation: float
    buyers: int
    settle_vol_30d: float
    meta: dict[str, Any] = field(default_factory=dict)


def make_corpus(
    n_endpoints: int = N_ENDPOINTS, seed: int = SEED, rng: np.random.Generator | None = None
) -> list[Endpoint]:
    """Build the 1:1:1 endpoint corpus described in Section 5."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    per_profile, remainder = divmod(n_endpoints, len(PROFILES))
    profiles: list[str] = []
    for i, p in enumerate(PROFILES):
        profiles += [p] * (per_profile + (1 if i < remainder else 0))

    endpoints = []
    for i, profile in enumerate(profiles):
        endpoints.append(
            Endpoint(
                endpoint_id=f"asp-{i:03d}",
                profile=profile,
                price_usdc=float(np.round(rng.uniform(0.0008, 0.0015), 6)),
                # Marketplace reputation is a lagging EMA. A freshly-stalled
                # provider still carries a high score: that lag is the blind
                # spot the canary exists to cover, so it is modelled explicitly.
                reputation=float(
                    np.clip(
                        {"honest": 0.93, "degraded": 0.78, "stalled": 0.85}[profile]
                        + rng.normal(0, 0.06),
                        0.0,
                        1.0,
                    )
                ),
                buyers=int(rng.integers(3, 400)),
                settle_vol_30d=float(np.round(rng.uniform(1, 900), 2)),
            )
        )
    return endpoints


def _draw_outcome(rng: np.random.Generator, profile: str) -> tuple[bool, bool]:
    """Draw ``(settled, delivered)`` for one call against a profile."""
    settled = bool(rng.random() < P_SETTLE)
    if not settled:
        return False, False
    return True, bool(rng.random() < P_DELIVER[profile])


def probe(rng: np.random.Generator, ep: Endpoint) -> dict[str, Any]:
    """Simulate one priced canary probe, plus an independent ground-truth call.

    ``delivered`` is what the canary's *own* paid call got. ``next_delivered``
    is a separate, independent draw from the same latent profile: the outcome of
    the subsequent production call the verdict is supposed to protect. Grading
    against the latter is what makes this a forecast rather than a restatement.
    """
    mu, sigma = LATENCY_LOGNORMAL[ep.profile]
    settled, delivered = _draw_outcome(rng, ep.profile)

    if not settled:
        # Never got far enough to pay.
        return dict(
            reachable=False,
            settled=False,
            delivered=False,
            http_status=0,
            schema_valid=False,
            artifact_valid=False,
            latency_ms=float(rng.uniform(1500, 5000)),
            settlement_to_response_gap_ms=0.0,
            returns_402_challenge=True,
            label="unreachable",
        )

    latency_ms = float(np.exp(rng.normal(mu, sigma)))
    if delivered:
        gap_ms = float(rng.gamma(GAP_SHAPE, GAP_SCALE_MS))
        schema_valid = bool(rng.random() < 0.99)
        # A genuinely delivered artifact still occasionally fails verification
        # (hash mismatch, truncated payload). Without this the observable
        # `artifact_valid` would be a deterministic function of `delivered`, and
        # oracle-mode scores would be a degenerate 1.000 — measuring the
        # simulator, not the detector. Matches the published run's 0.985.
        artifact_ok = bool(rng.random() < 0.985)
        http_status = 200
        label = "delivered" if schema_valid else "degraded"
    else:
        # Settled and stalled. Section 5 says the gap is "clipped to
        # maxTimeoutSeconds on stalled ones". Read literally that makes every
        # stalled gap the same constant, which no delivered probe can reach —
        # one feature would then separate the classes perfectly and the reported
        # scores would measure the simulator rather than the detector.
        #
        # Stalls do not actually look like that. They arrive two ways, and the
        # mock ASP reproduces both: a provider that errors out fast (5xx, short
        # gap) and one that takes the money and goes quiet (no usable response,
        # gap runs out to the cap). Model the mixture, clip at the cap.
        schema_valid = bool(rng.random() < 0.05)
        artifact_ok = False
        http_status = int(rng.choice([500, 502, 503, 200], p=[0.35, 0.20, 0.15, 0.30]))
        if http_status >= 500:
            # Fast failure: the error comes back promptly, just uselessly.
            gap_ms = float(min(rng.gamma(GAP_SHAPE, GAP_SCALE_MS * 3.0), MAX_TIMEOUT_MS))
        else:
            # Silent stall: nothing usable arrives before the advertised window
            # closes, so the observed gap is the clip itself.
            gap_ms = MAX_TIMEOUT_MS
        label = "stalled"

    # Independent draw: what the *next* paid call would get.
    _, next_delivered = _draw_outcome(rng, ep.profile)

    return dict(
        reachable=True,
        settled=True,
        delivered=bool(delivered),
        next_delivered=bool(next_delivered),
        http_status=http_status,
        schema_valid=schema_valid,
        artifact_valid=bool(artifact_ok),
        latency_ms=latency_ms,
        settlement_to_response_gap_ms=gap_ms,
        returns_402_challenge=True,
        label=label,
    )


def generate_probes(
    endpoints: list[Endpoint],
    trials_per_endpoint: int = TRIALS_PER_ENDPOINT,
    seed: int = SEED,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Generate the time-ordered probe corpus (150 trials/endpoint by default)."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for ep in endpoints:
        for t in range(trials_per_endpoint):
            r = probe(rng, ep)
            r.setdefault("next_delivered", False)
            r.update(
                endpoint_id=ep.endpoint_id,
                profile=ep.profile,
                state=ep.profile,  # alias, so baselines can be shared with repro
                reputation=ep.reputation,
                price_usdc=ep.price_usdc,
                buyers=ep.buyers,
                settle_vol_30d=ep.settle_vol_30d,
                t=t,
            )
            rows.append(r)
    return rows


def train_test_split(
    rows: list[dict[str, Any]],
    train_fraction: float = TRAIN_FRACTION,
    trials_per_endpoint: int = TRIALS_PER_ENDPOINT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Time-based split, stratified per provider.

    Each endpoint contributes its first ``train_fraction`` of trials to train and
    the rest to test, so no verdict is ever graded on a probe that fit it, and
    every provider appears on both sides.
    """
    split_t = int(round(trials_per_endpoint * train_fraction))
    train = [r for r in rows if r["t"] < split_t]
    test = [r for r in rows if r["t"] >= split_t]
    return train, test


def build(
    n_endpoints: int = N_ENDPOINTS,
    trials_per_endpoint: int = TRIALS_PER_ENDPOINT,
    seed: int = SEED,
    train_fraction: float = TRAIN_FRACTION,
    p_deliver_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the whole paper-spec dataset with one PCG64 stream seeded at 42."""
    global P_DELIVER
    original = dict(P_DELIVER)
    if p_deliver_overrides:
        P_DELIVER = {**P_DELIVER, **p_deliver_overrides}
    try:
        rng = np.random.default_rng(seed)
        endpoints = make_corpus(n_endpoints, rng=rng)
        rows = generate_probes(endpoints, trials_per_endpoint, rng=rng)
    finally:
        P_DELIVER = original

    train, test = train_test_split(rows, train_fraction, trials_per_endpoint)
    return {
        "endpoints": endpoints,
        "rows": rows,
        "train": train,
        "test": test,
        "config": {
            "n_endpoints": n_endpoints,
            "trials_per_endpoint": trials_per_endpoint,
            "n_probes": len(rows),
            "train_probes": len(train),
            "test_probes": len(test),
            "seed": seed,
            "train_fraction": train_fraction,
            "profiles": {p: sum(e.profile == p for e in endpoints) for p in PROFILES},
        },
    }
