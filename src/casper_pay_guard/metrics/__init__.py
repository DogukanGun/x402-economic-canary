"""Evaluation metrics for the economic canary.

The non-trivial estimators — the Murphy reliability/resolution/uncertainty
decomposition, the DeLong covariance for correlated ROC areas, McNemar's exact
and chi-square forms, and the IPS/SNIPS/DR closed forms — are the ones the
published run used, imported from :mod:`casper_pay_guard.repro` rather than
rewritten. They are correct, they are what produced the published numbers, and
reimplementing them would only introduce drift.

New here: :mod:`~casper_pay_guard.metrics.bootstrap`, which supplies the
confidence intervals the paper prints but nothing computed.
"""

from casper_pay_guard.metrics.bootstrap import (
    bootstrap_ci,
    headline_metric_cis,
    paired_difference_ci,
)
from casper_pay_guard.repro.metrics import (
    average_run_length,
    brier_decomposition,
    cusum_latency,
    delong_test,
    detection_latency_experiment,
    mcnemar_test,
    stalled_metrics,
)
from casper_pay_guard.repro.ope import (
    bootstrap_ci_saved,
    build_bandit_feedback,
    dr,
    ips,
    saved_vs_always_pay,
    snips,
)

__all__ = [
    # classification + calibration
    "stalled_metrics",
    "brier_decomposition",
    # change detection
    "cusum_latency",
    "average_run_length",
    "detection_latency_experiment",
    # significance
    "mcnemar_test",
    "delong_test",
    # off-policy evaluation
    "build_bandit_feedback",
    "ips",
    "snips",
    "dr",
    "saved_vs_always_pay",
    "bootstrap_ci_saved",
    # confidence intervals (new)
    "bootstrap_ci",
    "headline_metric_cis",
    "paired_difference_ci",
]
