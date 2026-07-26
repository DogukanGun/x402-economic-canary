"""
Evaluation metrics for the canary experiment.

Includes: stalled-class precision/recall/F2/PR-AUC, Brier score with the Murphy
reliability-resolution-uncertainty decomposition, CUSUM detection latency +
ARL-to-false-alarm, and the significance tests McNemar and DeLong.

FROZEN: verbatim port of experiment/metrics.py. Do not refactor.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score


# ---------------- classification metrics -------------------------------------
def stalled_metrics(y_true_stalled, y_pred_stalled, scores_stalled):
    """y_*_stalled: 1 == the probe was a genuine stall. scores: P(stall)."""
    p = precision_score(y_true_stalled, y_pred_stalled, zero_division=0)
    r = recall_score(y_true_stalled, y_pred_stalled, zero_division=0)
    f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
    ap = (
        average_precision_score(y_true_stalled, scores_stalled)
        if len(np.unique(y_true_stalled)) > 1
        else 0.0
    )
    return dict(precision=float(p), recall=float(r), f2=float(f2), pr_auc=float(ap))


# ---------------- Brier + Murphy decomposition -------------------------------
def brier_decomposition(prob_delivered, y_delivered, n_bins=10):
    """
    Brier = reliability - resolution + uncertainty  (Murphy 1973).
    prob_delivered: forecast P(delivered); y_delivered: outcome 1/0.
    """
    f = np.asarray(prob_delivered, dtype=float)
    o = np.asarray(y_delivered, dtype=float)
    N = len(o)
    obar = o.mean()
    brier = float(np.mean((f - o) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(f, edges[1:-1]), 0, n_bins - 1)
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        fk = f[m].mean()
        ok = o[m].mean()
        reliability += nk * (fk - ok) ** 2
        resolution += nk * (ok - obar) ** 2
    reliability /= N
    resolution /= N
    uncertainty = float(obar * (1 - obar))
    return dict(
        brier=brier,
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=uncertainty,
        recomposed=float(reliability - resolution + uncertainty),
    )


# ---------------- CUSUM detection latency ------------------------------------
def cusum_latency(fail_stream, target_ok=0.02, k=0.25, h=4.0):
    """
    One-sided CUSUM on a per-call FAILURE indicator (1==genuine failure).
    Returns index of first alarm (verdict flip pay->skip), or None.
    S_t = max(0, S_{t-1} + (x_t - target_ok - k)).
    """
    S = 0.0
    for t, x in enumerate(fail_stream):
        S = max(0.0, S + (x - target_ok - k))
        if S > h:
            return t
    return None


def average_run_length(rng, p_fail_ok, n_streams=400, length=1500, **cusum_kw):
    """ARL-to-false-alarm: mean calls to a (false) alarm on an all-healthy stream."""
    runs = []
    for _ in range(n_streams):
        stream = (rng.random(length) < p_fail_ok).astype(float)
        a = cusum_latency(stream, **cusum_kw)
        runs.append(a if a is not None else length)
    return float(np.mean(runs))


def detection_latency_experiment(
    rng, p_fail_ok=0.02, p_fail_bad=0.85, change_at=8, length=40, n_runs=300, **cusum_kw
):
    """
    Endpoint healthy until `change_at`, then genuinely failing. Measure calls
    from first genuine failure to the pay->skip verdict flip.
    """
    lats = []
    detected = 0
    for _ in range(n_runs):
        pre = (rng.random(change_at) < p_fail_ok).astype(float)
        post = (rng.random(length - change_at) < p_fail_bad).astype(float)
        stream = np.concatenate([pre, post])
        a = cusum_latency(stream, **cusum_kw)
        if a is not None and a >= change_at:
            lats.append(a - change_at)
            detected += 1
        elif a is not None and a < change_at:
            lats.append(0)  # early (still counts as detected quickly)
            detected += 1
    return dict(
        median_latency=float(np.median(lats)) if lats else float("inf"),
        mean_latency=float(np.mean(lats)) if lats else float("inf"),
        p95_latency=float(np.percentile(lats, 95)) if lats else float("inf"),
        detection_rate=float(detected / n_runs),
    )


# ---------------- significance tests -----------------------------------------
def mcnemar_test(correct_a, correct_b):
    """
    McNemar on paired correctness. correct_* : bool arrays (per probe correct?).
    Uses the exact binomial when discordant pairs are few, else chi-square w/ CC.
    Returns (statistic, p_value, b, c).
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum(a & ~b))  # A right, B wrong
    n10 = int(np.sum(~a & b))  # A wrong, B right
    n = n01 + n10
    if n == 0:
        return dict(statistic=0.0, p_value=1.0, b=n01, c=n10)
    from scipy.stats import binomtest, chi2

    if n < 25:
        p = binomtest(min(n01, n10), n, 0.5).pvalue
        stat = float(min(n01, n10))
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(chi2.sf(stat, 1))
    return dict(statistic=float(stat), p_value=float(p), b=n01, c=n10)


# ---- DeLong test for two correlated ROC AUCs --------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, m):
    """preds row0/1 = two predictors; first m columns positive. Returns aucs, cov."""
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    n = negative.shape[1]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(preds_sorted_transposed[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(y_true, score_a, score_b):
    """Two-sided p-value for AUC(a) == AUC(b) on the same samples (DeLong)."""
    from scipy.stats import norm

    y = np.asarray(y_true, dtype=int)
    order = (-y).argsort(kind="mergesort")
    label = y[order]
    m = int(label.sum())
    preds = np.vstack([np.asarray(score_a, float)[order], np.asarray(score_b, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        p = 1.0 if abs(aucs[0] - aucs[1]) < 1e-12 else 0.0
        z = 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * (1 - norm.cdf(abs(z))))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), z=float(z), p_value=float(p))
