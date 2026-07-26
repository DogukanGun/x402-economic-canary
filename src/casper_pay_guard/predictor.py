"""Calibrated delivery-probability predictors (paper Section 4).

Two predictors behind one interface, matching the two framings the paper mixes:

:class:`OracleModePredictor`
    Logistic regression over the current probe's own post-payment observables.
    This is what the published run used and what reproduces its numbers. It
    answers "did *this* call deliver?" — accurate, but you had to pay to ask.

:class:`ForecastModePredictor`
    ``HistGradientBoostingClassifier`` wrapped in an isotonic
    ``CalibratedClassifierCV(cv=5)``, over causal per-endpoint history features.
    This is what Section 4 and Section 5 describe, and it answers the question
    the abstract actually promises: "will the *next* paid call deliver?"

Hyper-parameters are exactly those Section 5 names: ``learning_rate=0.1``,
``max_iter=100``, ``max_leaf_nodes=31``, no ``max_depth`` cap,
``min_samples_leaf=20``, ``l2_regularization=0.0``, ``random_state=42``.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from casper_pay_guard import features as F

#: Section 5's HistGradientBoostingClassifier settings.
HGB_PARAMS = dict(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)

CALIBRATION_FOLDS = 5

#: Below this predicted delivery probability, the router should not pay.
DEFAULT_STALL_THRESHOLD = 0.5

#: Settlement-to-response gap above which a delivered-looking call is degraded.
DEFAULT_DEGRADE_GAP_MS = 4000.0


class DeliveryPredictor(Protocol):
    """What the scheduler and the MCP tool depend on."""

    feature_names: list[str]

    def fit(self, rows: Sequence[dict[str, Any]]) -> DeliveryPredictor: ...
    def predict_proba_delivered(self, rows: Sequence[dict[str, Any]]) -> np.ndarray: ...


class _BasePredictor:
    """Shared verdict logic; subclasses supply features and an estimator."""

    feature_names: list[str] = []
    target: str = "delivered"

    def __init__(
        self,
        stall_threshold: float = DEFAULT_STALL_THRESHOLD,
        degrade_gap_ms: float = DEFAULT_DEGRADE_GAP_MS,
        drop_features: Sequence[str] = (),
    ) -> None:
        self.stall_threshold = stall_threshold
        self.degrade_gap_ms = degrade_gap_ms
        self.drop_features = tuple(drop_features)
        self.clf: Any = None
        self._fallback_rate: float | None = None

    # -- features ---------------------------------------------------------
    def _extract(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _X(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        X = self._extract(rows)
        for name in self.drop_features:
            if name in self.feature_names:
                # Zero the column rather than dropping it, so the ablated model
                # keeps the same shape and the comparison stays like-for-like.
                X[:, self.feature_names.index(name)] = 0.0
        return X

    # -- fit / predict ----------------------------------------------------
    def _make_estimator(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def fit(self, rows: Sequence[dict[str, Any]]) -> _BasePredictor:
        y = F.labels_delivered(rows, target=self.target)
        if len(np.unique(y)) < 2:
            # Degenerate split (everything delivered or nothing did): fall back
            # to the base rate rather than raising, so ablations stay runnable.
            self._fallback_rate = float(y.mean()) if len(y) else 0.5
            self.clf = None
            return self
        self._fallback_rate = None
        self.clf = self._make_estimator()
        self.clf.fit(self._X(rows), y)
        return self

    def predict_proba_delivered(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if self.clf is None:
            rate = self._fallback_rate if self._fallback_rate is not None else 0.5
            return np.full(len(rows), rate, dtype=float)
        return self.clf.predict_proba(self._X(rows))[:, 1]

    # -- four-way verdict -------------------------------------------------
    def verdict(self, row: dict[str, Any], p_delivered: float) -> str:
        """Map one probe plus its delivery probability to a label."""
        if not row.get("reachable", True) or not row.get("settled", False):
            return "unreachable"
        if p_delivered < self.stall_threshold:
            return "stalled"
        if (
            row.get("settlement_to_response_gap_ms", 0.0) > self.degrade_gap_ms
            or not row.get("schema_valid", True)
        ):
            return "degraded"
        return "delivered"

    def label_stream(self, rows: Sequence[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
        p = self.predict_proba_delivered(rows)
        return [self.verdict(r, pi) for r, pi in zip(rows, p)], p


class OracleModePredictor(_BasePredictor):
    """Same-probe post-payment features + logistic regression.

    Reproduces the published framing. Note what it needs to score a call: the
    artifact from that call. It cannot be consulted *before* paying.
    """

    feature_names = list(F.ORACLE_FEATURE_NAMES)
    target = "delivered"

    def _extract(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return F.oracle_features(rows)

    def _make_estimator(self) -> Any:
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=2.0))


class ForecastModePredictor(_BasePredictor):
    """Causal history features + calibrated gradient boosting — Section 4/5.

    Predicts the outcome of the *next* independent paid call from what the
    endpoint has done so far. Strictly harder than oracle mode, and the only
    formulation that can inform a routing decision before money moves.
    """

    feature_names = list(F.FORECAST_FEATURE_NAMES)
    target = "next_delivered"

    def __init__(self, *args: Any, calibrate: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calibrate = calibrate

    def _extract(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return F.forecast_features(rows)

    def _make_estimator(self) -> Any:
        base = HistGradientBoostingClassifier(**HGB_PARAMS)
        if not self.calibrate:
            return base
        return CalibratedClassifierCV(base, method="isotonic", cv=CALIBRATION_FOLDS)


def build_predictor(mode: str, **kwargs: Any) -> _BasePredictor:
    """Factory: ``"oracle"`` or ``"forecast"``."""
    if mode == "oracle":
        return OracleModePredictor(**kwargs)
    if mode == "forecast":
        return ForecastModePredictor(**kwargs)
    raise ValueError(f"unknown predictor mode {mode!r}; expected 'oracle' or 'forecast'")
