"""Off-policy evaluation with an explicit routing policy.

The frozen path (:mod:`casper_pay_guard.repro.ope`) hard-codes one evaluation
policy — *pay with probability* :math:`\\hat p` — and reuses that same
:math:`\\hat p` as the reward model :math:`\\hat q(x, \\text{pay})`. Two things
are worth separating here:

**The policy is a choice, and the soft one is usually the wrong choice.**
Section 4 says the calibrated probability exists so it "translates directly into
correct expected-value routing". The expected-value rule is *pay iff*
:math:`\\hat p \\cdot v > c` — a threshold, not a coin flip. Paying with
probability :math:`\\hat p` abstains on endpoints it believes in: at
:math:`v=\\$0.02` and :math:`c=\\$0.001` a provider that delivers 65 % of the
time is worth paying every time (:math:`0.65 \\times 0.02 = \\$0.013 \\gg
\\$0.001`), yet the soft policy skips it a third of the time and forgoes the
value. That costs more than the stalls it avoids, and the estimate goes
negative. The soft policy only looks good when :math:`\\hat p` is already
near-deterministic, which is exactly the case in the published oracle-mode run.

**The reward model and the policy are different objects.** :math:`\\hat q` must
use the delivery probability; :math:`\\pi_e` is the action probability. They
coincide only under the soft policy, which is why the frozen code can conflate
them without being wrong *there*.

Both policies are reported, so the difference is visible rather than assumed.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

Policy = Literal["soft", "ev", "threshold"]


def pay_probability(
    p_delivered: np.ndarray,
    value: float,
    price: np.ndarray | float,
    policy: Policy = "ev",
    threshold: float = 0.5,
) -> np.ndarray:
    """Action probability :math:`\\pi_e(\\text{pay} \\mid z)` under a routing policy.

    ``soft``
        Pay with probability :math:`\\hat p`. What the published run evaluated.
    ``ev``
        Pay iff :math:`\\hat p \\cdot v > c`. The expected-value rule Section 4
        motivates.
    ``threshold``
        Pay iff :math:`\\hat p > \\text{threshold}`. Price-blind, for ablation.
    """
    p = np.asarray(p_delivered, dtype=float)
    if policy == "soft":
        return p
    if policy == "ev":
        return (p * value > np.asarray(price, dtype=float)).astype(float)
    if policy == "threshold":
        return (p > threshold).astype(float)
    raise ValueError(f"unknown policy {policy!r}")


def build_feedback(
    rows: Sequence[dict[str, Any]],
    p_delivered: np.ndarray,
    value: float = 0.02,
    policy: Policy = "ev",
    threshold: float = 0.5,
    price_override: float | None = None,
    seed: int = 2027,
) -> dict[str, Any]:
    """Logged bandit feedback with the policy and reward model kept distinct."""
    n = len(rows)
    rng = np.random.default_rng(seed)
    delivered = np.array([1 if r.get("delivered") else 0 for r in rows])
    price = (
        np.full(n, float(price_override))
        if price_override is not None
        else np.array([float(r.get("price_usdc", 0.001)) for r in rows])
    )

    # Behaviour policy: always-pay with a recorded stochastic propensity.
    pay_prob = np.clip(0.9 + rng.normal(0, 0.03, n), 0.6, 0.999)
    actions = (rng.random(n) < pay_prob).astype(int)
    reward = np.where(actions == 1, np.where(delivered == 1, value - price, -price), 0.0)
    pscore = np.where(actions == 1, pay_prob, 1 - pay_prob)

    p_hat = np.asarray(p_delivered, dtype=float)
    pi_e_pay = pay_probability(p_hat, value, price, policy=policy, threshold=threshold)

    # Reward model uses the DELIVERY probability, not the action probability.
    q_pay = p_hat * value - price
    q_skip = np.zeros(n)

    return dict(
        actions=actions,
        reward=reward,
        pay_prob=pay_prob,
        pscore=pscore,
        pi_e_pay=pi_e_pay,
        p_delivered=p_hat,
        delivered=delivered,
        price=price,
        value=value,
        q_pay=q_pay,
        q_skip=q_skip,
        policy=policy,
    )


def _weights(bf: dict[str, Any]) -> np.ndarray:
    pi_e = np.where(bf["actions"] == 1, bf["pi_e_pay"], 1 - bf["pi_e_pay"])
    return pi_e / bf["pscore"]


def always_pay_value(bf: dict[str, Any]) -> float:
    """Realized per-call value of paying every time — the baseline to beat."""
    return float(np.mean(np.where(bf["delivered"] == 1, bf["value"] - bf["price"], -bf["price"])))


def ips(bf: dict[str, Any]) -> float:
    return float(np.mean(_weights(bf) * bf["reward"]))


def snips(bf: dict[str, Any]) -> float:
    w = _weights(bf)
    return float(np.sum(w * bf["reward"]) / np.sum(w))


def dr_terms(bf: dict[str, Any]) -> np.ndarray:
    """Per-sample doubly-robust contributions."""
    w = _weights(bf)
    dm = bf["pi_e_pay"] * bf["q_pay"] + (1 - bf["pi_e_pay"]) * bf["q_skip"]
    q_logged = np.where(bf["actions"] == 1, bf["q_pay"], bf["q_skip"])
    return dm + w * (bf["reward"] - q_logged)


def dr(bf: dict[str, Any]) -> float:
    return float(np.mean(dr_terms(bf)))


def saved_vs_always_pay(bf: dict[str, Any], estimator: str = "dr") -> float:
    est = {"ips": ips, "snips": snips, "dr": dr}[estimator]
    return est(bf) - always_pay_value(bf)


def bootstrap_ci_saved(
    bf: dict[str, Any],
    estimator: str = "dr",
    n_boot: int = 2000,
    seed: int = 5,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Paired bootstrap CI on USDC saved per routed call vs always-pay."""
    rng = np.random.default_rng(seed)
    n = len(bf["reward"])
    ap = np.where(bf["delivered"] == 1, bf["value"] - bf["price"], -bf["price"])
    w = _weights(bf)
    terms = dr_terms(bf)

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if estimator == "dr":
            val = terms[idx].mean()
        elif estimator == "ips":
            val = (w[idx] * bf["reward"][idx]).mean()
        else:
            val = (w[idx] * bf["reward"][idx]).sum() / w[idx].sum()
        diffs[i] = val - ap[idx].mean()

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "point": float(diffs.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def evaluate_policies(
    rows: Sequence[dict[str, Any]],
    p_delivered: np.ndarray,
    value: float = 0.02,
    policies: Sequence[Policy] = ("soft", "ev"),
    n_boot: int = 2000,
    price_override: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Estimate savings under each routing policy, with CIs."""
    out: dict[str, dict[str, Any]] = {}
    for policy in policies:
        bf = build_feedback(
            rows, p_delivered, value=value, policy=policy, price_override=price_override
        )
        saved = {e: saved_vs_always_pay(bf, e) for e in ("ips", "snips", "dr")}
        ci = bootstrap_ci_saved(bf, estimator="dr", n_boot=n_boot)
        out[policy] = {
            **saved,
            "dr_ci_low": ci["ci_low"],
            "dr_ci_high": ci["ci_high"],
            "dr_ci_excludes_zero": ci["excludes_zero"],
            "all_positive": bool(all(v > 0 for v in saved.values())),
            "pay_rate": float(np.mean(bf["pi_e_pay"])),
        }
    return out
