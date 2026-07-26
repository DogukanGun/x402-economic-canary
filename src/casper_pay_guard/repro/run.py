"""
x402 Economic Canary — end-to-end published experiment (numpy-only simulation).

Detect settled-but-stalled Agent Service Providers with an economic canary that
completes the x402 pay-then-deliver handshake and verifies the returned artifact,
versus three baselines that structurally cannot see the failure:
  - always-pay (APEX no_policy)
  - 402-as-healthy HTTP liveness
  - marketplace / reputation ratings

Pipeline: build corpus -> probe -> time-split train/test -> fit canary ->
score all systems -> stalled precision/recall/F2/PR-AUC, Brier+Murphy decomp,
CUSUM latency + ARL, IPS/SNIPS/DR USDC-saved with bootstrap CI, McNemar + DeLong
-> 14 validation checks.

FROZEN: port of experiment/main.py, restructured from a script into ``run()``
without changing a single RNG draw or its ordering. The published numbers in
:data:`PUBLISHED` must come out of this unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from casper_pay_guard.repro import canary as C
from casper_pay_guard.repro import metrics as M
from casper_pay_guard.repro import ope, simulate

PROBLEM_ID = "topic-an-agent-facing-an-x402-challenge-from-a-c20321"

# ---------------------------------------------------------------------------
# The exact values printed in the paper and recorded in the original
# experiment/results.json. tests/test_reproduce.py asserts against these.
# ---------------------------------------------------------------------------
PUBLISHED = {
    "precision": 0.9943100995732574,
    "recall": 0.9831223628691983,
    "f2": 0.9853397237101776,
    "pr_auc": 0.8610113091299968,
    "brier": 0.0028841377704232653,
    "reliability": 0.0006504973731149726,
    "resolution": 0.1990799821388987,
    "uncertainty": 0.20135524691358028,
    "cusum_median_latency": 2.0,
    "cusum_mean_latency": 2.625,
    "cusum_p95_latency": 5.0,
    "cusum_detection_rate": 1.0,
    "arl_to_false_alarm": 1465.1125,
    "saved_ips": 0.00022292979410728012,
    "saved_snips": 3.999469496261156e-05,
    "saved_dr": 0.0002782179371421615,
    "saved_dr_ci_low": 0.00026011857979146597,
    "saved_dr_ci_high": 0.0002951632071679316,
    "mcnemar_liveness_p": 1.8916719623966624e-196,
    "mcnemar_ratings_p": 5.9042090569010235e-182,
    "delong_auc_canary": 0.961870015710204,
    "validation_passed": 14,
}

# Load-bearing constants — changing any of these breaks reproduction.
N_ENDPOINTS = 60
TRIALS = 120
CORPUS_SEED = 7
PROBE_SEED = 11
CUSUM_RNG_SEED = 2026
CUSUM_KW = dict(target_ok=0.02, k=0.25, h=1.8)
STALL_THRESHOLD = 0.5
DEGRADE_GAP_MS = 4000.0
CALL_VALUE = 0.02


def run(out_dir: Path | str | None = None, smoke_ok: bool | None = None) -> dict:
    """Run the published experiment. Returns the results dict.

    Args:
        out_dir: if given, ``results.json`` is written there.
        smoke_ok: result of the MCP smoke test (validation check 14). When
            ``None`` the real four-label fault-injection smoke is executed.
            It contributes no number to the experiment, only a pass/fail.
    """
    results: dict = {
        "status": "ok",
        "problem_id": PROBLEM_ID,
        "approach": "economic-canary (priced pay-then-deliver probe + artifact verification)",
        "baselines": ["always-pay", "402-as-healthy-liveness", "marketplace-ratings"],
        "metrics": {},
        "config": {},
        "errors": [],
    }

    RNG = np.random.default_rng(CUSUM_RNG_SEED)

    # ---- 1. corpus + probes -------------------------------------------------
    endpoints = simulate.make_corpus(n_endpoints=N_ENDPOINTS, seed=CORPUS_SEED)
    rows = simulate.generate_probes(endpoints, trials_per_endpoint=TRIALS, seed=PROBE_SEED)

    # time-based train/test split (early probes train, later probes test)
    split_t = int(TRIALS * 0.5)
    train = [r for r in rows if r["t"] < split_t]
    test = [r for r in rows if r["t"] >= split_t]

    # ---- 2. fit canary, score everything on the held-out test split ---------
    can = C.Canary(stall_threshold=STALL_THRESHOLD, degrade_gap_ms=DEGRADE_GAP_MS).fit(train)
    labels, p_deliver = can.label_stream(test)
    p_stall_canary = 1.0 - p_deliver  # P(stall) score
    pred_stall_canary = np.array([1 if lbl == "stalled" else 0 for lbl in labels])

    # ground truth: a genuine stall == settled but NOT delivered
    y_stall = np.array([1 if (r["settled"] and not r["delivered"]) else 0 for r in test])
    y_deliv = np.array([1 if r["delivered"] else 0 for r in test])

    # baselines
    bp_ap, bs_ap = C.baseline_always_pay(test)
    bp_lv, bs_lv = C.baseline_402_liveness(test)
    bp_rp, bs_rp = C.baseline_reputation(test)
    # baseline "P(stall)" scores = 1 - healthy score
    ps_ap, ps_lv, ps_rp = 1 - bs_ap, 1 - bs_lv, 1 - bs_rp

    # ---- 3. stalled-class detection metrics ---------------------------------
    results["metrics"]["economic-canary"] = M.stalled_metrics(
        y_stall, pred_stall_canary, p_stall_canary
    )
    results["metrics"]["always-pay"] = M.stalled_metrics(y_stall, bp_ap, ps_ap)
    results["metrics"]["402-as-healthy-liveness"] = M.stalled_metrics(y_stall, bp_lv, ps_lv)
    results["metrics"]["marketplace-ratings"] = M.stalled_metrics(y_stall, bp_rp, ps_rp)

    # ---- 4. Brier + Murphy decomposition (canary delivery forecast) ---------
    brier = M.brier_decomposition(p_deliver, y_deliv, n_bins=10)
    results["metrics"]["economic-canary"].update(
        brier=brier["brier"],
        reliability=brier["reliability"],
        resolution=brier["resolution"],
        uncertainty=brier["uncertainty"],
    )
    results["metrics"]["brier_decomposition"] = brier

    # ---- 5. CUSUM detection latency + ARL-to-false-alarm --------------------
    # NOTE: `det` must be computed before `arl` — they share one RNG stream.
    det = M.detection_latency_experiment(
        RNG, p_fail_ok=0.02, p_fail_bad=0.85, change_at=8, length=40, n_runs=400, **CUSUM_KW
    )
    arl = M.average_run_length(RNG, p_fail_ok=0.02, n_streams=400, length=1500, **CUSUM_KW)
    results["metrics"]["cusum"] = dict(
        median_latency_calls=det["median_latency"],
        mean_latency_calls=det["mean_latency"],
        p95_latency_calls=det["p95_latency"],
        detection_rate=det["detection_rate"],
        arl_to_false_alarm=arl,
    )

    # ---- 6. OPE: expected USDC saved per routed call ------------------------
    # canary pay-probability = predicted P(delivered) on the FULL corpus
    p_all = can.predict_proba_delivered(rows)
    bf = ope.build_bandit_feedback(rows, p_all, value=CALL_VALUE)
    saved = {e: ope.saved_vs_always_pay(bf, e) for e in ("ips", "snips", "dr")}
    ci = ope.bootstrap_ci_saved(bf, estimator="dr", n_boot=2000, seed=5)
    results["metrics"]["usdc_saved_per_call"] = dict(
        ips=saved["ips"],
        snips=saved["snips"],
        dr=saved["dr"],
        dr_ci_low=ci["ci_low"],
        dr_ci_high=ci["ci_high"],
        dr_ci_excludes_zero=ci["excludes_zero"],
        all_positive=bool(all(v > 0 for v in saved.values())),
    )

    # ---- 7. significance tests vs baselines ---------------------------------
    correct_canary = pred_stall_canary == y_stall
    mcnemar_vs = {
        "402-as-healthy-liveness": M.mcnemar_test(correct_canary, (bp_lv == y_stall)),
        "marketplace-ratings": M.mcnemar_test(correct_canary, (bp_rp == y_stall)),
        "always-pay": M.mcnemar_test(correct_canary, (bp_ap == y_stall)),
    }
    delong_vs = {
        "402-as-healthy-liveness": M.delong_test(y_stall, p_stall_canary, ps_lv),
        "marketplace-ratings": M.delong_test(y_stall, p_stall_canary, ps_rp),
    }
    results["metrics"]["significance"] = dict(
        mcnemar={
            k: {"p_value": v["p_value"], "b": v["b"], "c": v["c"]} for k, v in mcnemar_vs.items()
        },
        delong={
            k: {"auc_canary": v["auc_a"], "auc_baseline": v["auc_b"], "p_value": v["p_value"]}
            for k, v in delong_vs.items()
        },
    )

    # ---- 8. price ablation (kept for parity with the published run) ---------
    # The paper's Table 3 is a full price x stall-rate grid; that lives in
    # casper_pay_guard.ablation and is NOT part of this frozen path.
    ablations = {}
    for price in (0.001, 0.005, 0.02):
        bf2 = ope.build_bandit_feedback(rows, p_all, value=CALL_VALUE)
        bf2["price"] = np.full_like(bf2["price"], price)
        # recompute reward + q under the new price
        bf2["reward"] = np.where(
            bf2["actions"] == 1,
            np.where(bf2["delivered"] == 1, bf2["value"] - price, -price),
            0.0,
        )
        bf2["q_pay"] = bf2["pi_e_pay"] * bf2["value"] - price
        ablations[f"price_{price}"] = round(ope.saved_vs_always_pay(bf2, "dr"), 6)
    results["metrics"]["ablation_dr_saved_by_price"] = ablations

    # ---- 9. MCP smoke test (validation check 14; contributes no number) -----
    if smoke_ok is None:
        from casper_pay_guard.oracle import fault_injection_smoke

        smoke_ok, smoke_seen = fault_injection_smoke()
    else:
        smoke_seen = {}
    results["metrics"]["mcp_smoke"] = dict(passed=bool(smoke_ok), labels=smoke_seen)

    # ---- config + validation-criteria check ---------------------------------
    results["config"] = dict(
        n_endpoints=N_ENDPOINTS,
        trials_per_endpoint=TRIALS,
        train_probes=len(train),
        test_probes=len(test),
        split_t=split_t,
        classifier="StandardScaler+LogisticRegression(C=2)",
        framing="oracle-mode (same-probe post-payment features)",
        chains=[196, 84532],
        min_price_usdc=0.001,
        seed=CUSUM_RNG_SEED,
        note="numpy-only simulation of the x402 pay-then-deliver canary; "
        "no torch/network/chain calls.",
    )

    c = results["metrics"]["economic-canary"]
    sig = results["metrics"]["significance"]
    checks = {
        "precision>=0.90": c["precision"] >= 0.90,
        "recall>=0.85": c["recall"] >= 0.85,
        "pr_auc>=0.80": c["pr_auc"] >= 0.80,
        "brier<=0.10": c["brier"] <= 0.10,
        "reliability<0.02": c["reliability"] < 0.02,
        "cusum_latency<=3": det["median_latency"] <= 3,
        "arl>=500": arl >= 500,
        "usdc_saved_all_positive": results["metrics"]["usdc_saved_per_call"]["all_positive"],
        "dr_ci_excludes_zero": ci["excludes_zero"],
        "mcnemar_liveness_p<0.05": sig["mcnemar"]["402-as-healthy-liveness"]["p_value"] < 0.05,
        "mcnemar_ratings_p<0.05": sig["mcnemar"]["marketplace-ratings"]["p_value"] < 0.05,
        "delong_liveness_p<0.05": sig["delong"]["402-as-healthy-liveness"]["p_value"] < 0.05,
        "delong_ratings_p<0.05": sig["delong"]["marketplace-ratings"]["p_value"] < 0.05,
        "mcp_smoke_all_four_labels": bool(smoke_ok),
    }
    results["metrics"]["validation_criteria"] = {k: bool(v) for k, v in checks.items()}
    results["metrics"]["validation_passed"] = int(sum(checks.values()))
    results["metrics"]["validation_total"] = len(checks)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    return results


def main() -> int:
    out = Path(__file__).resolve().parents[3] / "results" / "published"
    results = run(out_dir=out)
    print(f"[OK] wrote {out / 'results.json'}  status={results['status']}")
    passed = results["metrics"]["validation_passed"]
    total = results["metrics"]["validation_total"]
    print(f"[VALIDATION] {passed}/{total} criteria satisfied")
    for k, v in results["metrics"]["validation_criteria"].items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
