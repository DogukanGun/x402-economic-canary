"""The three incumbent trust signals the canary is graded against.

Each is deliberately given the *best* version of its own information, so the
comparison is about what a signal can structurally observe, not about tuning:

``always_pay``
    APEX's no-policy setting. Routes every call, so it never abstains and never
    avoids a stall. Its recall on the stalled class is exactly zero by
    construction.

``liveness_402``
    UptimeRobot-style HTTP monitoring extended to x402: an endpoint that answers
    and returns a fresh ``402`` is healthy. It flags hard-down endpoints only —
    and a stalling provider answers its 402 challenge perfectly.

``marketplace_ratings``
    Bazaar-style reputation from 30-day settlement volume and buyer counts. A
    stalling endpoint *accumulates* settlement volume while delivering nothing,
    so the metric meant to build trust is the one the attacker farms.

Every function returns ``(pred_stalled, score_healthy)`` so they can be scored
by the same harness as the canary.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: Reputation below which the marketplace baseline calls an endpoint bad.
RATINGS_THRESHOLD = 0.80


def always_pay(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Never abstains, so it can never avoid a stall."""
    return np.zeros(len(rows), dtype=int), np.full(len(rows), 0.99, dtype=float)


def liveness_402(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Scores a fresh 402 as up — healthy at the exact moment money is taken."""
    pred = np.zeros(len(rows), dtype=int)
    score = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        alive = bool(r.get("reachable")) and bool(r.get("returns_402_challenge"))
        score[i] = 0.97 if alive else 0.02
        pred[i] = 0 if alive else 1
    return pred, score


def marketplace_ratings(
    rows: Sequence[dict[str, Any]], threshold: float = RATINGS_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Lagging reputation EMA: high for a provider that only just started stalling."""
    pred = np.zeros(len(rows), dtype=int)
    score = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        rep = float(r.get("reputation", 0.0))
        score[i] = rep
        pred[i] = 1 if rep < threshold else 0
    return pred, score


#: Name -> callable, in the order the paper reports them.
BASELINES = {
    "always-pay": always_pay,
    "402-as-healthy-liveness": liveness_402,
    "marketplace-ratings": marketplace_ratings,
}


def score_all(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """Run every baseline, returning stall predictions and P(stall) scores."""
    out = {}
    for name, fn in BASELINES.items():
        pred, healthy = fn(rows)
        out[name] = {"pred_stalled": pred, "score_stalled": 1.0 - healthy}
    return out
