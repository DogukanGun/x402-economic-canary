"""Paper-spec experiment runner — Section 5 of the paper, as written.

Runs the 18,000-probe / 120-endpoint corpus that Section 5 describes, through
either predictor framing:

``--mode oracle``
    Same-probe post-payment features. Grades the call that just happened.

``--mode forecast``
    Causal history features + calibrated gradient boosting, graded against a
    *subsequent independent* paid call. This is what Sections 3–5 describe and
    the only framing that can inform a routing decision before money moves.

Numbers from here are **not** the published headline numbers, which came from a
different corpus and a different classifier — see
:mod:`casper_pay_guard.repro` and REPRODUCIBILITY.md. They are reported as
computed, including where they miss the paper's Table 2 acceptance targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from casper_pay_guard import baselines, cusum, simulate
from casper_pay_guard.features import labels_delivered, labels_stalled
from casper_pay_guard.metrics import (
    average_run_length,
    brier_decomposition,
    delong_test,
    detection_latency_experiment,
    headline_metric_cis,
    mcnemar_test,
    stalled_metrics,
)
from casper_pay_guard.metrics import offpolicy
from casper_pay_guard.predictor import build_predictor

#: Economic value an agent gets from one successfully delivered call, in USDC.
#: Matches the published run so the savings estimates stay comparable.
CALL_VALUE_USDC = 0.02

#: CUSUM harness settings, shared with the published run.
CUSUM_RNG_SEED = 2026
CUSUM_KW = dict(target_ok=0.02, k=0.25, h=1.8)

#: Table 2 acceptance targets, in the paper's own order.
TARGETS = {
    "stalled_precision": (">=", 0.90),
    "stalled_recall": (">=", 0.85),
    "pr_auc": (">=", 0.80),
    "brier": ("<=", 0.10),
    "reliability": ("<", 0.02),
    "cusum_median_latency": ("<=", 3),
    "arl_to_false_alarm": (">=", 500),
}


def run(
    mode: str = "forecast",
    n_endpoints: int = simulate.N_ENDPOINTS,
    trials_per_endpoint: int = simulate.TRIALS_PER_ENDPOINT,
    seed: int = simulate.SEED,
    out_dir: Path | str | None = None,
    n_boot: int = 2000,
    smoke_ok: bool | None = None,
) -> dict[str, Any]:
    """Run the paper-spec experiment end to end."""
    data = simulate.build(
        n_endpoints=n_endpoints, trials_per_endpoint=trials_per_endpoint, seed=seed
    )
    train, test = data["train"], data["test"]

    # -- fit + score --------------------------------------------------------
    predictor = build_predictor(mode).fit(train)
    labels, p_deliver = predictor.label_stream(test)
    p_stall = 1.0 - p_deliver
    pred_stall = np.array([1 if lbl == "stalled" else 0 for lbl in labels])

    y_stall = labels_stalled(test)
    # Grade the probability against whatever the predictor claims to forecast:
    # its own call (oracle mode) or the next independent one (forecast mode).
    y_deliv = labels_delivered(test, target=predictor.target)

    metrics: dict[str, Any] = {}
    metrics["economic-canary"] = stalled_metrics(y_stall, pred_stall, p_stall)

    base_scores = baselines.score_all(test)
    for name, b in base_scores.items():
        metrics[name] = stalled_metrics(y_stall, b["pred_stalled"], b["score_stalled"])

    # -- calibration --------------------------------------------------------
    brier = brier_decomposition(p_deliver, y_deliv, n_bins=10)
    metrics["economic-canary"].update(
        brier=brier["brier"],
        reliability=brier["reliability"],
        resolution=brier["resolution"],
        uncertainty=brier["uncertainty"],
    )
    metrics["brier_decomposition"] = brier

    # -- confidence intervals (absent from the original experiment) ---------
    metrics["confidence_intervals"] = headline_metric_cis(
        y_stalled=y_stall,
        pred_stalled=pred_stall,
        score_stalled=p_stall,
        p_delivered=p_deliver,
        y_delivered=y_deliv,
        rows=test,
        cluster_key="endpoint_id",
        n_boot=n_boot,
    )

    # -- change detection ---------------------------------------------------
    rng = np.random.default_rng(CUSUM_RNG_SEED)
    det = detection_latency_experiment(
        rng, p_fail_ok=0.02, p_fail_bad=0.85, change_at=8, length=40, n_runs=400, **CUSUM_KW
    )
    arl = average_run_length(rng, p_fail_ok=0.02, n_streams=400, length=1500, **CUSUM_KW)
    metrics["cusum"] = dict(
        median_latency_calls=det["median_latency"],
        mean_latency_calls=det["mean_latency"],
        p95_latency_calls=det["p95_latency"],
        detection_rate=det["detection_rate"],
        arl_to_false_alarm=arl,
        params=dict(cusum.PUBLISHED_PARAMS.__dict__),
    )

    # -- off-policy savings, under both routing policies ---------------------
    # `soft` is what the published run evaluated (pay with probability p-hat);
    # `ev` is the expected-value rule Section 4 motivates (pay iff p*v > c).
    # They differ a lot once p-hat is genuinely uncertain — see
    # casper_pay_guard.metrics.offpolicy.
    p_all = predictor.predict_proba_delivered(data["rows"])
    policies = offpolicy.evaluate_policies(
        data["rows"], p_all, value=CALL_VALUE_USDC, policies=("soft", "ev"), n_boot=n_boot
    )
    metrics["usdc_saved_by_policy"] = policies
    headline = policies["ev"]
    metrics["usdc_saved_per_call"] = dict(
        policy="ev",
        ips=headline["ips"],
        snips=headline["snips"],
        dr=headline["dr"],
        dr_ci_low=headline["dr_ci_low"],
        dr_ci_high=headline["dr_ci_high"],
        dr_ci_excludes_zero=headline["dr_ci_excludes_zero"],
        all_positive=headline["all_positive"],
        pay_rate=headline["pay_rate"],
    )
    ci = {"excludes_zero": headline["dr_ci_excludes_zero"]}

    # -- significance vs each incumbent -------------------------------------
    correct_canary = pred_stall == y_stall
    metrics["significance"] = dict(
        mcnemar={
            name: {
                k: v
                for k, v in mcnemar_test(
                    correct_canary, base_scores[name]["pred_stalled"] == y_stall
                ).items()
                if k in ("p_value", "b", "c")
            }
            for name in base_scores
        },
        delong={
            name: {
                "auc_canary": d["auc_a"],
                "auc_baseline": d["auc_b"],
                "p_value": d["p_value"],
            }
            for name in ("402-as-healthy-liveness", "marketplace-ratings")
            for d in [delong_test(y_stall, p_stall, base_scores[name]["score_stalled"])]
        },
    )

    # -- MCP smoke ----------------------------------------------------------
    if smoke_ok is None:
        from casper_pay_guard.oracle import fault_injection_smoke

        smoke_ok, smoke_seen = fault_injection_smoke()
    else:
        smoke_seen = {}
    metrics["mcp_smoke"] = dict(passed=bool(smoke_ok), labels=smoke_seen)

    # -- validation ---------------------------------------------------------
    c = metrics["economic-canary"]
    sig = metrics["significance"]
    checks = {
        "precision>=0.90": c["precision"] >= 0.90,
        "recall>=0.85": c["recall"] >= 0.85,
        "pr_auc>=0.80": c["pr_auc"] >= 0.80,
        "brier<=0.10": c["brier"] <= 0.10,
        "reliability<0.02": c["reliability"] < 0.02,
        "cusum_latency<=3": det["median_latency"] <= 3,
        "arl>=500": arl >= 500,
        "usdc_saved_all_positive": metrics["usdc_saved_per_call"]["all_positive"],
        "dr_ci_excludes_zero": ci["excludes_zero"],
        "mcnemar_liveness_p<0.05": sig["mcnemar"]["402-as-healthy-liveness"]["p_value"] < 0.05,
        "mcnemar_ratings_p<0.05": sig["mcnemar"]["marketplace-ratings"]["p_value"] < 0.05,
        "delong_liveness_p<0.05": sig["delong"]["402-as-healthy-liveness"]["p_value"] < 0.05,
        "delong_ratings_p<0.05": sig["delong"]["marketplace-ratings"]["p_value"] < 0.05,
        "mcp_smoke_all_four_labels": bool(smoke_ok),
    }
    metrics["validation_criteria"] = {k: bool(v) for k, v in checks.items()}
    metrics["validation_passed"] = int(sum(checks.values()))
    metrics["validation_total"] = len(checks)

    results = {
        "status": "ok",
        "spec": "paper Section 5",
        "approach": f"economic-canary ({mode} mode)",
        "baselines": list(base_scores),
        "metrics": metrics,
        "config": {
            **data["config"],
            "mode": mode,
            "predictor": type(predictor).__name__,
            "features": predictor.feature_names,
            "target": predictor.target,
            "call_value_usdc": CALL_VALUE_USDC,
            "min_price_usdc": simulate.MIN_PRICE_USDC,
            "network": "base-sepolia",
            "asset": "USDC",
        },
        "errors": [],
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"results_{mode}.json").write_text(json.dumps(results, indent=2, default=float))

    return results


def _print_report(results: dict[str, Any]) -> None:
    m = results["metrics"]
    c = m["economic-canary"]
    ci = m["confidence_intervals"]
    cfg = results["config"]

    print(f"\n=== paper-spec ({cfg['mode']} mode) ===")
    print(
        f"{cfg['n_probes']} probes / {cfg['n_endpoints']} endpoints "
        f"({cfg['train_probes']} train, {cfg['test_probes']} test), seed {cfg['seed']}"
    )
    print(f"predictor: {cfg['predictor']}  target: {cfg['target']}")

    def _ci(key):
        b = ci.get(key)
        return f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]" if b else ""

    print(f"\n  stalled precision  {c['precision']:.4f}  {_ci('stalled_precision')}")
    print(f"  stalled recall     {c['recall']:.4f}  {_ci('stalled_recall')}")
    print(f"  F2                 {c['f2']:.4f}")
    print(f"  PR-AUC             {c['pr_auc']:.4f}  {_ci('pr_auc')}")
    print(f"  Brier              {c['brier']:.4f}  {_ci('brier')}")
    print(f"  reliability        {c['reliability']:.5f}   resolution {c['resolution']:.4f}")
    print("\n  USDC saved per routed call, by routing policy:")
    for policy, s in m["usdc_saved_by_policy"].items():
        tag = "pay w.p. p-hat" if policy == "soft" else "pay iff p*v > c"
        print(
            f"    {policy:5s} ({tag:15s}) DR {s['dr']:+.6g} "
            f"[{s['dr_ci_low']:+.6g}, {s['dr_ci_high']:+.6g}]  pay_rate {s['pay_rate']:.3f}"
        )
    cu = m["cusum"]
    print(
        f"  CUSUM              median {cu['median_latency_calls']:.0f} calls, "
        f"ARL {cu['arl_to_false_alarm']:.0f}, detection {cu['detection_rate']:.0%}"
    )

    print("\n  baselines (stalled class):")
    for name in results["baselines"]:
        b = m[name]
        print(f"    {name:26s} P={b['precision']:.3f} R={b['recall']:.3f} PR-AUC={b['pr_auc']:.3f}")

    passed, total = m["validation_passed"], m["validation_total"]
    print(f"\n  [VALIDATION] {passed}/{total}")
    for k, v in m["validation_criteria"].items():
        if not v:
            print(f"     FAIL  {k}")
    if passed == total:
        print("     all criteria satisfied")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the paper-spec (Section 5) experiment")
    ap.add_argument("--mode", choices=("forecast", "oracle", "both"), default="both")
    ap.add_argument("--endpoints", type=int, default=simulate.N_ENDPOINTS)
    ap.add_argument("--trials", type=int, default=simulate.TRIALS_PER_ENDPOINT)
    ap.add_argument("--seed", type=int, default=simulate.SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results" / "paper_spec"))
    args = ap.parse_args(argv)

    modes = ("oracle", "forecast") if args.mode == "both" else (args.mode,)
    for mode in modes:
        results = run(
            mode=mode,
            n_endpoints=args.endpoints,
            trials_per_endpoint=args.trials,
            seed=args.seed,
            out_dir=args.out,
            n_boot=args.n_boot,
        )
        _print_report(results)
    print(f"\nwrote {args.out}/results_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
