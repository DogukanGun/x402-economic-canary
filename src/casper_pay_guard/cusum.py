"""CUSUM change detection over the delivery stream (paper Section 3, Eq. 3).

The paper writes the statistic as

.. math::  C_t = \\max(0, C_{t-1} + (\\mu_0 - d_t - k))

with reference mean :math:`\\mu_0 = 0.98` (the honest in-control delivery rate),
slack :math:`k = 0.25` and threshold :math:`\\lambda = 1.0`, over the *delivery*
indicator :math:`d_t`.

The published run wrote the equivalent statistic over the *failure* indicator
:math:`x_t = 1 - d_t`:

.. math::  S_t = \\max(0, S_{t-1} + (x_t - \\text{target\\_ok} - k))

with ``target_ok = 0.02``, ``k = 0.25``, ``h = 1.8``. Substituting
:math:`x_t = 1 - d_t` and :math:`\\text{target\\_ok} = 1 - \\mu_0` shows the two
increments differ only in sign convention — but the thresholds do not match
(1.8 vs 1.0), so they are genuinely different detectors. Both are expressible
here; :data:`PUBLISHED_PARAMS` is the one the reported ARL of 1465 comes from.

Why it matters operationally: this is what converts a stall into a routing
decision after a couple of paid observations, while raising a spurious flip only
about once per ARL healthy calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CusumParams:
    """One-sided CUSUM configuration over a delivery stream."""

    #: In-control delivery rate. Failures are scored against ``1 - mu0``.
    mu0: float = 0.98
    #: Slack: how much sustained excess failure is tolerated per call.
    k: float = 0.25
    #: Decision threshold. Crossing it flips the verdict pay -> skip.
    threshold: float = 1.8

    @property
    def target_fail_rate(self) -> float:
        return 1.0 - self.mu0


#: The configuration behind the published ARL of 1465 and 2-call median latency.
PUBLISHED_PARAMS = CusumParams(mu0=0.98, k=0.25, threshold=1.8)

#: The configuration the paper's Section 4 prose states (lambda = 1.0).
PAPER_PARAMS = CusumParams(mu0=0.98, k=0.25, threshold=1.0)


class CusumDetector:
    """Streaming detector. Feed it delivery outcomes; it tells you when to stop.

    >>> d = CusumDetector()
    >>> any(d.update(delivered=False) for _ in range(10))
    True
    """

    def __init__(self, params: CusumParams = PUBLISHED_PARAMS) -> None:
        self.params = params
        self.statistic = 0.0
        self.alarmed = False
        self.n = 0

    def reset(self) -> None:
        self.statistic = 0.0
        self.alarmed = False
        self.n = 0

    def update(self, delivered: bool) -> bool:
        """Feed one outcome. Returns True once the verdict has flipped."""
        p = self.params
        failure = 0.0 if delivered else 1.0
        self.statistic = max(0.0, self.statistic + (failure - p.target_fail_rate - p.k))
        self.n += 1
        if self.statistic > p.threshold:
            self.alarmed = True
        return self.alarmed


def first_alarm(
    fail_stream: Iterable[float], params: CusumParams = PUBLISHED_PARAMS
) -> int | None:
    """Index of the first alarm on a stream of failure indicators, else None."""
    S = 0.0
    for t, x in enumerate(fail_stream):
        S = max(0.0, S + (float(x) - params.target_fail_rate - params.k))
        if S > params.threshold:
            return t
    return None


def average_run_length(
    rng: np.random.Generator,
    p_fail_ok: float = 0.02,
    n_streams: int = 400,
    length: int = 1500,
    params: CusumParams = PUBLISHED_PARAMS,
) -> float:
    """Mean calls to a *false* alarm on an all-healthy stream (ARL_0)."""
    runs = []
    for _ in range(n_streams):
        stream = (rng.random(length) < p_fail_ok).astype(float)
        a = first_alarm(stream, params)
        runs.append(a if a is not None else length)
    return float(np.mean(runs))


def detection_latency(
    rng: np.random.Generator,
    p_fail_ok: float = 0.02,
    p_fail_bad: float = 0.85,
    change_at: int = 8,
    length: int = 40,
    n_runs: int = 400,
    params: CusumParams = PUBLISHED_PARAMS,
) -> dict[str, float]:
    """Calls from onset of genuine failure to the pay -> skip verdict flip."""
    lats: list[float] = []
    detected = 0
    for _ in range(n_runs):
        pre = (rng.random(change_at) < p_fail_ok).astype(float)
        post = (rng.random(length - change_at) < p_fail_bad).astype(float)
        a = first_alarm(np.concatenate([pre, post]), params)
        if a is None:
            continue
        lats.append(float(max(a - change_at, 0)))
        detected += 1
    return dict(
        median_latency=float(np.median(lats)) if lats else float("inf"),
        mean_latency=float(np.mean(lats)) if lats else float("inf"),
        p95_latency=float(np.percentile(lats, 95)) if lats else float("inf"),
        detection_rate=float(detected / n_runs) if n_runs else 0.0,
    )


def verdict_stream(
    delivered: Sequence[bool], params: CusumParams = PUBLISHED_PARAMS
) -> list[bool]:
    """Per-call "should I skip this endpoint?" verdicts, once and for all.

    Sticky by design: once an endpoint has demonstrably flipped to stalling, the
    router should stop paying until something re-qualifies it.
    """
    det = CusumDetector(params)
    return [det.update(bool(d)) for d in delivered]
