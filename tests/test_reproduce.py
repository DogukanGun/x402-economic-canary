"""Guard the published numbers.

Every value asserted here appears in the paper (Zenodo 10.5281/zenodo.21515696)
and in the original ``experiment/results.json``. They reproduce *exactly* — not
to a tolerance — under the pinned stack (numpy 2.4.4 / scipy 1.17.1 /
scikit-learn 1.8.0, Python 3.11).

If this test starts failing, something in ``casper_pay_guard.repro`` was
refactored, a dependency floated, or an RNG draw moved. Do not "fix" it by
loosening the tolerance.
"""

import pytest

from casper_pay_guard.repro.run import PUBLISHED, run


@pytest.fixture(scope="module")
def results():
    return run()


def _canary(results):
    return results["metrics"]["economic-canary"]


@pytest.mark.parametrize(
    "key",
    ["precision", "recall", "f2", "pr_auc", "brier", "reliability", "resolution", "uncertainty"],
)
def test_stalled_class_and_calibration_exact(results, key):
    assert _canary(results)[key] == PUBLISHED[key]


@pytest.mark.parametrize(
    "key,path",
    [
        ("cusum_median_latency", "median_latency_calls"),
        ("cusum_mean_latency", "mean_latency_calls"),
        ("cusum_p95_latency", "p95_latency_calls"),
        ("cusum_detection_rate", "detection_rate"),
        ("arl_to_false_alarm", "arl_to_false_alarm"),
    ],
)
def test_cusum_detection_exact(results, key, path):
    assert results["metrics"]["cusum"][path] == PUBLISHED[key]


@pytest.mark.parametrize(
    "key,path",
    [
        ("saved_ips", "ips"),
        ("saved_snips", "snips"),
        ("saved_dr", "dr"),
        ("saved_dr_ci_low", "dr_ci_low"),
        ("saved_dr_ci_high", "dr_ci_high"),
    ],
)
def test_offpolicy_savings_exact(results, key, path):
    assert results["metrics"]["usdc_saved_per_call"][path] == PUBLISHED[key]


def test_significance_exact(results):
    sig = results["metrics"]["significance"]
    assert sig["mcnemar"]["402-as-healthy-liveness"]["p_value"] == PUBLISHED["mcnemar_liveness_p"]
    assert sig["mcnemar"]["marketplace-ratings"]["p_value"] == PUBLISHED["mcnemar_ratings_p"]
    assert sig["delong"]["402-as-healthy-liveness"]["auc_canary"] == PUBLISHED["delong_auc_canary"]
    # DeLong p-values underflow double precision against both baselines.
    assert sig["delong"]["402-as-healthy-liveness"]["p_value"] == 0.0
    assert sig["delong"]["marketplace-ratings"]["p_value"] == 0.0


def test_all_fourteen_validation_criteria_pass(results):
    m = results["metrics"]
    failed = [k for k, v in m["validation_criteria"].items() if not v]
    assert not failed, f"failed criteria: {failed}"
    assert m["validation_passed"] == m["validation_total"] == PUBLISHED["validation_passed"] == 14


def test_corpus_shape_is_the_one_that_ran(results):
    """The paper's Section 5 claims 18,000 probes; the published run used 7,200.

    This asserts the *actual* shape, so the discrepancy stays visible in code
    rather than only in prose. See REPRODUCIBILITY.md.
    """
    cfg = results["config"]
    assert (cfg["n_endpoints"], cfg["trials_per_endpoint"]) == (60, 120)
    assert cfg["train_probes"] == cfg["test_probes"] == 3600
    assert cfg["classifier"] == "StandardScaler+LogisticRegression(C=2)"


def test_baselines_are_blind_to_the_stall(results):
    """The structural claim: liveness and always-pay never flag a stall."""
    for name in ("always-pay", "402-as-healthy-liveness"):
        assert results["metrics"][name]["recall"] == 0.0
        assert results["metrics"][name]["precision"] == 0.0
    # Ratings do flag some, but badly.
    assert results["metrics"]["marketplace-ratings"]["recall"] < 0.5
