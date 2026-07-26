"""
Economic-canary classifier + the three baselines, behind one harness.

The canary completes the x402 pay-then-deliver handshake and VERIFIES the
returned artifact, so it observes post-payment features the baselines cannot:
artifact_valid, schema_valid, http_status, settlement_to_response_gap. It fits a
calibrated logistic model of P(delivered) and maps each probe to one of four
labels: delivered / degraded / stalled / unreachable.

NOTE ON FRAMING: the feature vector below is drawn from the *same probe* being
labelled, so this is a post-hoc delivery **oracle**, not a forecast of the next
call. See casper_pay_guard.predictor.ForecastModePredictor for the history-only
formulation the paper's Section 3 actually defines, and REPRODUCIBILITY.md for
why both are reported.

FROZEN: verbatim port of experiment/canary.py. Do not refactor.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----- feature extraction (only the canary sees post-payment fields) ---------
def features(rows):
    X = []
    for r in rows:
        X.append(
            [
                1.0 if r["settled"] else 0.0,
                1.0 if r["artifact_valid"] else 0.0,
                1.0 if r["schema_valid"] else 0.0,
                1.0 if r["http_status"] == 200 else 0.0,
                1.0 if r["http_status"] >= 500 else 0.0,
                np.log1p(r["latency_ms"]),
                np.log1p(r["settlement_to_response_gap_ms"]),
            ]
        )
    return np.asarray(X, dtype=float)


#: Names of the columns produced by :func:`features`, in order. Used by the
#: leave-one-feature-out ablation (Table 1).
FEATURE_NAMES = [
    "settled",
    "artifact_valid",
    "schema_valid",
    "http_200",
    "http_5xx",
    "log_latency",
    "log_gap",
]


def labels_delivered(rows):
    return np.asarray([1 if r["delivered"] else 0 for r in rows], dtype=int)


class Canary:
    """Priced canary: predicts P(delivered) and emits a 4-class verdict."""

    def __init__(self, stall_threshold=0.5, degrade_gap_ms=4000.0, drop_features=()):
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=2.0),
        )
        self.stall_threshold = stall_threshold
        self.degrade_gap_ms = degrade_gap_ms
        # Column indices to zero out, for the Table 1 ablation. Empty by default,
        # so the published path is untouched.
        self.drop_features = tuple(drop_features)

    def _X(self, rows):
        X = features(rows)
        for name in self.drop_features:
            X[:, FEATURE_NAMES.index(name)] = 0.0
        return X

    def fit(self, rows):
        self.clf.fit(self._X(rows), labels_delivered(rows))
        return self

    def predict_proba_delivered(self, rows):
        return self.clf.predict_proba(self._X(rows))[:, 1]

    def verdict(self, r, p_deliver):
        """Map one probe + its delivery probability to a 4-class label."""
        if not r["reachable"] or not r["settled"]:
            return "unreachable"
        if p_deliver < self.stall_threshold:
            return "stalled"
        # delivered-ish, but flag partial/slow as degraded
        if r["settlement_to_response_gap_ms"] > self.degrade_gap_ms or not r["schema_valid"]:
            return "degraded"
        return "delivered"

    def label_stream(self, rows):
        p = self.predict_proba_delivered(rows)
        return [self.verdict(r, pi) for r, pi in zip(rows, p)], p


# ---------------------------------------------------------------------------
#  Baselines behind the identical harness.
#  Each returns (pred_stalled: 0/1, score_healthy: prob-of-delivery proxy).
# ---------------------------------------------------------------------------
def baseline_always_pay(rows):
    """APEX no_policy: always route/pay -> never predicts a stall."""
    pred_stalled = np.zeros(len(rows), dtype=int)
    score = np.ones(len(rows), dtype=float) * 0.99  # always "healthy"
    return pred_stalled, score


def baseline_402_liveness(rows):
    """
    UptimeRobot-style + x402 handshake probe. Sees only an unauthenticated
    liveness check: a reachable endpoint that returns a fresh 402 is scored
    healthy. Structurally blind to settled-but-stalled delivery failures.
    """
    pred_stalled = np.zeros(len(rows), dtype=int)
    score = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        alive = r["reachable"] and r["returns_402_challenge"]
        score[i] = 0.97 if alive else 0.02
        pred_stalled[i] = 0 if alive else 1  # only flags hard-down
    return pred_stalled, score


def baseline_reputation(rows, thresh=0.80):
    """Marketplace/reputation ratings (Bazaar-style). Lagging EMA."""
    pred_stalled = np.zeros(len(rows), dtype=int)
    score = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        rep = float(r["reputation"])
        score[i] = rep
        pred_stalled[i] = 1 if rep < thresh else 0
    return pred_stalled, score
