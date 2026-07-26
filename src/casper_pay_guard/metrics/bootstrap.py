"""Paired-bootstrap confidence intervals for the headline metrics.

The paper reports 95% CIs on stalled-class precision, recall, PR-AUC and the
Brier score ("2,000-resample paired-bootstrap 95% confidence intervals"), but
nothing in the original experiment computed them — only the doubly-robust
savings CI was bootstrapped. This module supplies the missing harness.

Resampling is **paired and clustered by endpoint**. Probes from one provider are
not independent — a stalling endpoint produces a run of correlated failures — so
resampling individual probes would understate the interval. Clustering on
``endpoint_id`` resamples whole providers, which is the honest unit of
independence here. Pass ``cluster_key=None`` for the naive i.i.d. version.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 5
DEFAULT_ALPHA = 0.05


def _clusters(rows: Sequence[dict[str, Any]], key: str) -> list[np.ndarray]:
    idx: dict[Any, list[int]] = {}
    for i, r in enumerate(rows):
        idx.setdefault(r.get(key), []).append(i)
    return [np.asarray(v, dtype=int) for v in idx.values()]


def _resample(
    rng: np.random.Generator, n: int, clusters: list[np.ndarray] | None
) -> np.ndarray:
    if clusters is None:
        return rng.integers(0, n, n)
    picks = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[p] for p in picks])


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    rows: Sequence[dict[str, Any]] | None = None,
    cluster_key: str | None = "endpoint_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Percentile bootstrap CI for ``statistic(indices)``."""
    rng = np.random.default_rng(seed)
    clusters = _clusters(rows, cluster_key) if (rows is not None and cluster_key) else None

    draws: list[float] = []
    for _ in range(n_boot):
        idx = _resample(rng, n, clusters)
        try:
            val = statistic(idx)
        except ValueError:
            continue  # degenerate resample (e.g. one class absent)
        if np.isfinite(val):
            draws.append(float(val))

    if not draws:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}

    arr = np.asarray(draws)
    return {
        "point": float(statistic(np.arange(n))),
        "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_boot": len(draws),
    }


def headline_metric_cis(
    y_stalled: np.ndarray,
    pred_stalled: np.ndarray,
    score_stalled: np.ndarray,
    p_delivered: np.ndarray,
    y_delivered: np.ndarray,
    rows: Sequence[dict[str, Any]] | None = None,
    cluster_key: str | None = "endpoint_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """CIs for the four metrics the paper prints intervals for."""
    y_stalled = np.asarray(y_stalled)
    pred_stalled = np.asarray(pred_stalled)
    score_stalled = np.asarray(score_stalled)
    p_delivered = np.asarray(p_delivered, dtype=float)
    y_delivered = np.asarray(y_delivered)
    n = len(y_stalled)

    def _precision(i):
        return precision_score(y_stalled[i], pred_stalled[i], zero_division=0)

    def _recall(i):
        return recall_score(y_stalled[i], pred_stalled[i], zero_division=0)

    def _pr_auc(i):
        if len(np.unique(y_stalled[i])) < 2:
            raise ValueError("single-class resample")
        return average_precision_score(y_stalled[i], score_stalled[i])

    def _brier(i):
        return float(np.mean((p_delivered[i] - y_delivered[i]) ** 2))

    common = dict(n=n, rows=rows, cluster_key=cluster_key, n_boot=n_boot, seed=seed)
    return {
        "stalled_precision": bootstrap_ci(_precision, **common),
        "stalled_recall": bootstrap_ci(_recall, **common),
        "pr_auc": bootstrap_ci(_pr_auc, **common),
        "brier": bootstrap_ci(_brier, **common),
    }


def paired_difference_ci(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    n: int,
    rows: Sequence[dict[str, Any]] | None = None,
    cluster_key: str | None = "endpoint_id",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Paired bootstrap on ``metric_a - metric_b`` over the same resamples."""

    def _diff(idx: np.ndarray) -> float:
        return metric_a(idx) - metric_b(idx)

    out = bootstrap_ci(
        _diff, n, rows=rows, cluster_key=cluster_key, n_boot=n_boot, seed=seed, alpha=alpha
    )
    out["excludes_zero"] = bool(out["ci_low"] > 0 or out["ci_high"] < 0)
    return out
