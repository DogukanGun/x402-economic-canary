"""
Off-policy evaluation of expected stablecoin saved per routed call.

Logged policy (behaviour): always-pay with a recorded stochastic pay_probability
(propensities). Reward of a call = economic value realized minus USDC price:

    pay + delivered  -> +value - price
    pay + stalled    -> -price            (money burned on a non-delivery)
    skip             ->  0                 (avoids the wasted price on a stall,
                                            forgoes value on a real delivery)

'USDC saved per routed call' is measured vs the always-pay policy, which pays on
every call. We estimate the canary (evaluation) policy value with IPS, SNIPS and
a doubly-robust estimator, plus a paired bootstrap 95% CI on the difference.
obp's estimators reduce to these closed forms; we implement them directly so the
experiment runs with no external service.

FROZEN: verbatim port of experiment/ope.py. Do not refactor.
"""

import numpy as np


def build_bandit_feedback(rows, canary_pscore, value=0.02):
    """
    Construct logged bandit feedback.
    action a in {0=skip, 1=pay}. Logged policy pays (a=1) with prob pay_prob.
    reward computed from ground-truth delivery.
    Returns dict of arrays.
    """
    n = len(rows)
    rng = np.random.default_rng(2027)
    delivered = np.array([1 if r["delivered"] else 0 for r in rows])
    price = np.array([r["price_usdc"] for r in rows])
    # behaviour policy pay probability (recorded propensity); mostly pays
    pay_prob = np.clip(0.9 + rng.normal(0, 0.03, n), 0.6, 0.999)
    actions = (rng.random(n) < pay_prob).astype(int)
    # realized reward under the logged action
    reward = np.where(actions == 1, np.where(delivered == 1, value - price, -price), 0.0)
    # evaluation policy = canary: pay iff predicted-delivered prob high
    pi_e_pay = canary_pscore  # P(pay | canary) in [0,1]
    # propensity of the LOGGED action under behaviour policy
    pscore = np.where(actions == 1, pay_prob, 1 - pay_prob)
    # reward model \hat q(x,a): expected reward if paying (uses canary q of delivery)
    q_pay = pi_e_pay * value - price  # E[reward | pay] via delivery prob
    q_skip = np.zeros(n)
    return dict(
        actions=actions,
        reward=reward,
        pay_prob=pay_prob,
        pscore=pscore,
        pi_e_pay=pi_e_pay,
        delivered=delivered,
        price=price,
        value=value,
        q_pay=q_pay,
        q_skip=q_skip,
    )


def _policy_value_always_pay(bf):
    """Always-pay realized value per call (baseline being beaten)."""
    return float(
        np.mean(np.where(bf["delivered"] == 1, bf["value"] - bf["price"], -bf["price"]))
    )


def ips(bf):
    """Vanilla inverse propensity score estimate of canary policy value."""
    # eval policy prob of the LOGGED action
    pi_e = np.where(bf["actions"] == 1, bf["pi_e_pay"], 1 - bf["pi_e_pay"])
    w = pi_e / bf["pscore"]
    return float(np.mean(w * bf["reward"]))


def snips(bf):
    """Self-normalized IPS."""
    pi_e = np.where(bf["actions"] == 1, bf["pi_e_pay"], 1 - bf["pi_e_pay"])
    w = pi_e / bf["pscore"]
    return float(np.sum(w * bf["reward"]) / np.sum(w))


def dr(bf):
    """Doubly-robust estimate of canary policy value."""
    pi_e = np.where(bf["actions"] == 1, bf["pi_e_pay"], 1 - bf["pi_e_pay"])
    w = pi_e / bf["pscore"]
    # direct-method baseline: E_{a~pi_e}[q(x,a)]
    dm = bf["pi_e_pay"] * bf["q_pay"] + (1 - bf["pi_e_pay"]) * bf["q_skip"]
    q_logged = np.where(bf["actions"] == 1, bf["q_pay"], bf["q_skip"])
    return float(np.mean(dm + w * (bf["reward"] - q_logged)))


def saved_vs_always_pay(bf, estimator="dr"):
    est = {"ips": ips, "snips": snips, "dr": dr}[estimator]
    return est(bf) - _policy_value_always_pay(bf)


def bootstrap_ci_saved(bf, estimator="dr", n_boot=2000, seed=5, alpha=0.05):
    """Paired bootstrap 95% CI on USDC saved per call vs always-pay."""
    rng = np.random.default_rng(seed)
    n = len(bf["reward"])
    ap = np.where(bf["delivered"] == 1, bf["value"] - bf["price"], -bf["price"])
    pi_e = np.where(bf["actions"] == 1, bf["pi_e_pay"], 1 - bf["pi_e_pay"])
    w = pi_e / bf["pscore"]
    dm = bf["pi_e_pay"] * bf["q_pay"] + (1 - bf["pi_e_pay"]) * bf["q_skip"]
    q_logged = np.where(bf["actions"] == 1, bf["q_pay"], bf["q_skip"])
    dr_terms = dm + w * (bf["reward"] - q_logged)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if estimator == "dr":
            val = dr_terms[idx].mean()
        elif estimator == "ips":
            val = (w[idx] * bf["reward"][idx]).mean()
        else:
            val = (w[idx] * bf["reward"][idx]).sum() / w[idx].sum()
        diffs.append(val - ap[idx].mean())
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return dict(
        point=float(np.mean(diffs)),
        ci_low=lo,
        ci_high=hi,
        excludes_zero=bool(lo > 0 or hi < 0),
    )
