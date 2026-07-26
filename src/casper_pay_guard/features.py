"""Per-endpoint feature extraction (paper Section 3 and 4).

Two feature sets, because the paper describes one and the published run used the
other. Keeping both explicit is the point — see REPRODUCIBILITY.md.

**Forecast features** (Section 3: ``z_{e,t}`` "summarizing delivery rate, p95
latency, and gap history"; Section 4: "rolling delivery rate, p95 latency, mean
gap, and last-good timestamp"). Computed *causally* from that endpoint's prior
probes only — probe ``t`` never sees its own outcome. This is what a router
actually has when deciding whether to pay for call ``t``.

**Oracle features** (what the published run used): the current probe's own
post-payment observables — ``artifact_valid``, ``schema_valid``, ``http_status``,
its own gap. These grade a call that already happened. They are excellent at
that job and useless as a forecast, because you must pay to obtain them.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: Column order of :func:`forecast_features`.
FORECAST_FEATURE_NAMES = [
    "delivery_rate",
    "p95_latency_ms",
    "mean_gap_ms",
    "calls_since_last_good",
    "n_prior_probes",
    "stall_rate",
    "recent_delivery_rate",
]

#: Column order of :func:`oracle_features` — mirrors repro.canary.features.
ORACLE_FEATURE_NAMES = [
    "settled",
    "artifact_valid",
    "schema_valid",
    "http_200",
    "http_5xx",
    "log_latency",
    "log_gap",
]

#: Window for the "recent" delivery rate, in calls.
RECENT_WINDOW = 10

#: Neutral prior used before an endpoint has any history, so the first probes of
#: a provider are not silently scored as perfect.
PRIOR_DELIVERY_RATE = 0.5


def oracle_features(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Same-probe post-payment features. Post-hoc grading, not forecasting."""
    X = np.empty((len(rows), len(ORACLE_FEATURE_NAMES)), dtype=float)
    for i, r in enumerate(rows):
        status = r.get("http_status") or 0
        X[i] = (
            1.0 if r.get("settled") else 0.0,
            1.0 if r.get("artifact_valid") else 0.0,
            1.0 if r.get("schema_valid") else 0.0,
            1.0 if status == 200 else 0.0,
            1.0 if status >= 500 else 0.0,
            np.log1p(r.get("latency_ms", 0.0)),
            np.log1p(r.get("settlement_to_response_gap_ms", 0.0)),
        )
    return X


def forecast_features(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Causal per-endpoint history features.

    ``rows`` must be time-ordered within each endpoint. Row ``i`` is described
    only by probes of the same endpoint that came strictly before it, so there
    is no leakage from the outcome being predicted.
    """
    history: dict[str, dict[str, Any]] = {}
    X = np.empty((len(rows), len(FORECAST_FEATURE_NAMES)), dtype=float)

    for i, r in enumerate(rows):
        eid = r.get("endpoint_id", "?")
        h = history.setdefault(
            eid,
            {"delivered": [], "latency": [], "gap": [], "stalled": [], "since_good": 0.0},
        )
        n = len(h["delivered"])

        if n == 0:
            delivery_rate = PRIOR_DELIVERY_RATE
            p95_latency = 0.0
            mean_gap = 0.0
            stall_rate = 0.0
            recent = PRIOR_DELIVERY_RATE
        else:
            delivery_rate = float(np.mean(h["delivered"]))
            p95_latency = float(np.percentile(h["latency"], 95)) if h["latency"] else 0.0
            mean_gap = float(np.mean(h["gap"])) if h["gap"] else 0.0
            stall_rate = float(np.mean(h["stalled"]))
            recent = float(np.mean(h["delivered"][-RECENT_WINDOW:]))

        X[i] = (
            delivery_rate,
            np.log1p(p95_latency),
            np.log1p(mean_gap),
            h["since_good"],
            float(n),
            stall_rate,
            recent,
        )

        # Now fold this probe into the endpoint's history for the *next* row.
        delivered = bool(r.get("delivered"))
        h["delivered"].append(1.0 if delivered else 0.0)
        h["latency"].append(float(r.get("latency_ms", 0.0)))
        h["gap"].append(float(r.get("settlement_to_response_gap_ms", 0.0)))
        h["stalled"].append(1.0 if (r.get("settled") and not delivered) else 0.0)
        h["since_good"] = 0.0 if delivered else h["since_good"] + 1.0

    return X


def endpoint_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-endpoint rolling stats for reporting and the MCP tool surface.

    Returns delivery rate, p95 latency, mean settlement-to-response gap and the
    index of the last good probe — the four quantities Section 4 names.
    """
    by_ep: dict[str, dict[str, Any]] = {}
    for r in rows:
        eid = r.get("endpoint_id", "?")
        acc = by_ep.setdefault(
            eid, {"delivered": [], "latency": [], "gap": [], "last_good_t": None, "n": 0}
        )
        delivered = bool(r.get("delivered"))
        acc["delivered"].append(1.0 if delivered else 0.0)
        acc["latency"].append(float(r.get("latency_ms", 0.0)))
        acc["gap"].append(float(r.get("settlement_to_response_gap_ms", 0.0)))
        acc["n"] += 1
        if delivered:
            acc["last_good_t"] = r.get("t", acc["n"] - 1)

    out: dict[str, dict[str, float]] = {}
    for eid, acc in by_ep.items():
        out[eid] = {
            "delivery_rate": float(np.mean(acc["delivered"])) if acc["n"] else 0.0,
            "p95_latency_ms": float(np.percentile(acc["latency"], 95)) if acc["n"] else 0.0,
            "mean_gap_ms": float(np.mean(acc["gap"])) if acc["n"] else 0.0,
            "last_good_t": float(acc["last_good_t"]) if acc["last_good_t"] is not None else -1.0,
            "probes": float(acc["n"]),
        }
    return out


def labels_delivered(rows: Sequence[dict[str, Any]], target: str = "delivered") -> np.ndarray:
    """Binary delivery labels.

    ``target="delivered"`` grades the probe itself (oracle mode).
    ``target="next_delivered"`` grades the *subsequent independent* paid call,
    which is what Section 5 says ground truth is and what a router cares about.
    """
    return np.asarray([1 if r.get(target) else 0 for r in rows], dtype=int)


def labels_stalled(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """A genuine stall: settled but not delivered."""
    return np.asarray(
        [1 if (r.get("settled") and not r.get("delivered")) else 0 for r in rows], dtype=int
    )
