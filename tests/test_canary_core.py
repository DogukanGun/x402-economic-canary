"""Simulator, causal features, CUSUM, baselines, off-policy estimators."""

import numpy as np
import pytest

from casper_pay_guard import baselines, cusum, features, simulate
from casper_pay_guard.metrics import offpolicy
from casper_pay_guard.predictor import build_predictor


# --------------------------------------------------------------------------- #
# Paper-spec simulator (Section 5)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data():
    return simulate.build()


def test_corpus_matches_the_shape_section_5_describes(data):
    cfg = data["config"]
    assert cfg["n_endpoints"] == 120
    assert cfg["trials_per_endpoint"] == 150
    assert cfg["n_probes"] == 18_000
    assert cfg["train_probes"] == 10_800
    assert cfg["test_probes"] == 7_200
    assert cfg["profiles"] == {"honest": 40, "degraded": 40, "stalled": 40}


def test_corpus_is_deterministic_under_seed_42():
    a = simulate.build()["rows"]
    b = simulate.build()["rows"]
    assert [r["delivered"] for r in a] == [r["delivered"] for r in b]


def test_delivery_rates_track_the_specified_profiles(data):
    for profile, expected in simulate.P_DELIVER.items():
        settled = [r for r in data["rows"] if r["profile"] == profile and r["settled"]]
        rate = np.mean([r["delivered"] for r in settled])
        assert rate == pytest.approx(expected, abs=0.03), f"{profile}: {rate:.3f}"


def test_stalled_gaps_are_not_a_single_constant(data):
    """A degenerate gap would separate the classes perfectly on its own.

    Section 5 says stalled gaps are "clipped to maxTimeoutSeconds"; taken
    literally that makes every stalled gap identical and unreachable by any
    delivered probe, which would measure the simulator instead of the detector.
    Stalls arrive both fast (5xx) and silent (timeout), so the gap must vary.
    """
    gaps = np.array(
        [
            r["settlement_to_response_gap_ms"]
            for r in data["rows"]
            if r["settled"] and not r["delivered"]
        ]
    )
    assert len(np.unique(gaps)) > 1
    assert gaps.min() < simulate.MAX_TIMEOUT_MS


def test_split_is_time_based_and_covers_every_provider(data):
    train_eps = {r["endpoint_id"] for r in data["train"]}
    test_eps = {r["endpoint_id"] for r in data["test"]}
    assert train_eps == test_eps, "every provider must appear on both sides"
    assert max(r["t"] for r in data["train"]) < min(r["t"] for r in data["test"])


def test_ground_truth_is_an_independent_subsequent_draw(data):
    """`next_delivered` must not be a copy of `delivered`."""
    settled = [r for r in data["rows"] if r["settled"]]
    agree = np.mean([r["delivered"] == r["next_delivered"] for r in settled])
    assert 0.5 < agree < 0.95, f"suspiciously coupled: {agree:.3f}"


# --------------------------------------------------------------------------- #
# Causal features — no leakage
# --------------------------------------------------------------------------- #
def test_forecast_features_never_see_the_current_probe():
    """The first probe of an endpoint has no history, so it must be the prior."""
    rows = [
        {"endpoint_id": "a", "delivered": True, "latency_ms": 100.0,
         "settlement_to_response_gap_ms": 10.0, "settled": True, "t": 0},
        {"endpoint_id": "a", "delivered": False, "latency_ms": 200.0,
         "settlement_to_response_gap_ms": 20.0, "settled": True, "t": 1},
    ]
    X = features.forecast_features(rows)
    i = features.FORECAST_FEATURE_NAMES.index("delivery_rate")
    assert X[0, i] == features.PRIOR_DELIVERY_RATE
    # Row 1 sees exactly one prior probe, which delivered.
    assert X[1, i] == 1.0
    assert X[1, features.FORECAST_FEATURE_NAMES.index("n_prior_probes")] == 1.0


def test_forecast_features_are_per_endpoint():
    rows = [
        {"endpoint_id": "a", "delivered": False, "latency_ms": 1.0,
         "settlement_to_response_gap_ms": 1.0, "settled": True},
        {"endpoint_id": "b", "delivered": True, "latency_ms": 1.0,
         "settlement_to_response_gap_ms": 1.0, "settled": True},
        {"endpoint_id": "a", "delivered": False, "latency_ms": 1.0,
         "settlement_to_response_gap_ms": 1.0, "settled": True},
    ]
    X = features.forecast_features(rows)
    i = features.FORECAST_FEATURE_NAMES.index("delivery_rate")
    assert X[2, i] == 0.0, "endpoint a's history must not include endpoint b"


def test_calls_since_last_good_counts_up_then_resets():
    rows = [
        {"endpoint_id": "a", "delivered": d, "latency_ms": 1.0,
         "settlement_to_response_gap_ms": 1.0, "settled": True}
        for d in (True, False, False, True)
    ]
    X = features.forecast_features(rows)
    i = features.FORECAST_FEATURE_NAMES.index("calls_since_last_good")
    assert list(X[:, i]) == [0.0, 0.0, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #
def test_sustained_failure_flips_the_verdict_quickly():
    det = cusum.CusumDetector()
    flips = [det.update(delivered=False) for _ in range(10)]
    assert any(flips)
    assert flips.index(True) <= 3, "should flip within a few calls"


def test_a_healthy_stream_does_not_flip():
    det = cusum.CusumDetector()
    assert not any(det.update(delivered=True) for _ in range(500))


def test_an_isolated_failure_does_not_flip():
    """Twitching at one bad call would make the detector useless."""
    det = cusum.CusumDetector()
    for d in [True] * 20 + [False] + [True] * 20:
        assert not det.update(delivered=d)


def test_the_verdict_is_sticky_once_flipped():
    det = cusum.CusumDetector()
    for _ in range(10):
        det.update(delivered=False)
    assert det.update(delivered=True), "must not un-flip on one good call"


def test_arl_to_false_alarm_meets_the_published_target():
    rng = np.random.default_rng(2026)
    arl = cusum.average_run_length(rng, p_fail_ok=0.02, n_streams=200, length=1500)
    assert arl >= 500


def test_detection_latency_is_a_couple_of_calls():
    rng = np.random.default_rng(2026)
    det = cusum.detection_latency(rng, n_runs=200)
    assert det["median_latency"] <= 3
    assert det["detection_rate"] == 1.0


# --------------------------------------------------------------------------- #
# Baselines — the structural blindness claim
# --------------------------------------------------------------------------- #
def test_liveness_scores_a_stalling_endpoint_as_healthy(data):
    """The paper's core structural claim, as a test."""
    stalls = [r for r in data["rows"] if r["settled"] and not r["delivered"]]
    pred, score = baselines.liveness_402(stalls)
    assert pred.sum() == 0, "liveness flags none of the stalls"
    assert np.all(score > 0.9), "and scores every one of them healthy"


def test_always_pay_never_abstains(data):
    pred, _ = baselines.always_pay(data["rows"])
    assert pred.sum() == 0


def test_ratings_stay_high_for_freshly_stalled_providers(data):
    stalled_eps = [e for e in data["endpoints"] if e.profile == "stalled"]
    mean_rep = np.mean([e.reputation for e in stalled_eps])
    assert mean_rep > baselines.RATINGS_THRESHOLD, (
        "a lagging reputation EMA must not have caught up yet — that lag is the "
        "blind spot the canary covers"
    )


# --------------------------------------------------------------------------- #
# Off-policy: the routing policy matters
# --------------------------------------------------------------------------- #
def test_expected_value_policy_pays_when_the_upside_justifies_it():
    p = np.array([0.03, 0.65, 0.98])
    pi = offpolicy.pay_probability(p, value=0.02, price=0.001, policy="ev")
    # 0.03*0.02 = 0.0006 < 0.001 -> skip; the other two clear the price.
    assert list(pi) == [0.0, 1.0, 1.0]


def test_soft_policy_abstains_on_endpoints_it_believes_in():
    """Why the soft policy loses money: it skips profitable calls."""
    p = np.array([0.65])
    soft = offpolicy.pay_probability(p, value=0.02, price=0.001, policy="soft")
    ev = offpolicy.pay_probability(p, value=0.02, price=0.001, policy="ev")
    assert soft[0] == 0.65 and ev[0] == 1.0


def test_reward_model_uses_delivery_probability_not_action_probability(data):
    """q(x, pay) must be p_hat*v - c regardless of the policy's pay rate."""
    p = np.full(len(data["rows"]), 0.5)
    soft = offpolicy.build_feedback(data["rows"], p, value=0.02, policy="soft")
    ev = offpolicy.build_feedback(data["rows"], p, value=0.02, policy="ev")
    assert np.allclose(soft["q_pay"], ev["q_pay"])
    assert not np.allclose(soft["pi_e_pay"], ev["pi_e_pay"])


def test_perfect_foresight_beats_always_pay(data):
    """Sanity floor: an oracle policy must show positive savings."""
    truth = np.array([1.0 if r["delivered"] else 0.0 for r in data["rows"]])
    bf = offpolicy.build_feedback(data["rows"], truth, value=0.02, policy="ev")
    assert offpolicy.saved_vs_always_pay(bf, "dr") > 0


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
def test_forecast_predictor_is_calibrated_and_beats_chance(data):
    pred = build_predictor("forecast").fit(data["train"])
    p = pred.predict_proba_delivered(data["test"])
    y = features.labels_delivered(data["test"], target="next_delivered")
    assert 0.0 <= p.min() and p.max() <= 1.0
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    assert brier < base, "must beat predicting the base rate"


def test_dropping_a_feature_changes_the_model(data):
    full = build_predictor("forecast").fit(data["train"])
    ablated = build_predictor("forecast", drop_features=["delivery_rate"]).fit(data["train"])
    assert not np.allclose(
        full.predict_proba_delivered(data["test"]),
        ablated.predict_proba_delivered(data["test"]),
    )
