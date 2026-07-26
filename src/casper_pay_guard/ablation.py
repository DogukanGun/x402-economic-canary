"""Table 1 and Table 3 — the paper's two ablations, actually computed.

Neither had generating code. The published experiment varied per-call price over
three values and stopped there; the component ablation and the price x stall-rate
grid were printed without anything producing them.

**Table 1 — component ablation.** Stalled-class precision, recall and PR-AUC as
each feature is removed one at a time, plus a row with the CUSUM stage disabled.
Run in forecast mode, because the features the table names — delivery rate, p95
latency, settlement-to-response gap — are the *history* features of Section 4,
not the per-probe observables the published run used.

The paper's own caption states PR-AUC is unchanged when only CUSUM is removed,
since PR-AUC is a property of the ranked predictor and CUSUM is a downstream
stage. That invariant is asserted here rather than assumed.

**Table 3 — savings grid.** Doubly-robust USDC saved per routed call across
per-call price and stall base rate. Reported under both routing policies,
because the sign of the whole table depends on which one you use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from casper_pay_guard import simulate
from casper_pay_guard.features import labels_delivered, labels_stalled
from casper_pay_guard.metrics import offpolicy, stalled_metrics
from casper_pay_guard.predictor import build_predictor

#: Features Table 1 removes, mapped to the columns they zero out in each
#: framing. The table names three quantities; each predictor spells them
#: differently, so the ablation has to be told which columns carry them or it
#: silently drops nothing and every row comes out identical.
TABLE1_FEATURES: dict[str, dict[str, list[str]]] = {
    "forecast": {
        "delivery rate": ["delivery_rate", "recent_delivery_rate"],
        "p95 latency": ["p95_latency_ms"],
        "settlement-to-response gap": ["mean_gap_ms"],
    },
    "oracle": {
        # Per-probe delivery evidence stands in for the rolling delivery rate.
        "delivery rate": ["artifact_valid", "schema_valid"],
        "p95 latency": ["log_latency"],
        "settlement-to-response gap": ["log_gap"],
    },
}

#: Table 3 axes.
PRICES_USDC = (0.001, 0.01, 0.10)
STALL_RATES = (0.10, 0.20, 0.40)

CALL_VALUE_USDC = 0.02


# --------------------------------------------------------------------------- #
# Table 1 — component ablation
# --------------------------------------------------------------------------- #
def _score(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    drop: list[str],
    mode: str,
    cusum_enabled: bool,
) -> dict[str, float]:
    predictor = build_predictor(mode, drop_features=drop).fit(train)
    labels, p_deliver = predictor.label_stream(test)
    p_stall = 1.0 - p_deliver
    y_stall = labels_stalled(test)

    if cusum_enabled:
        pred_stall = np.array([1 if lbl == "stalled" else 0 for lbl in labels])
    else:
        # CUSUM disabled: threshold the calibrated probability directly, with no
        # streaming stage to require sustained failure before flipping.
        pred_stall = (p_deliver < predictor.stall_threshold).astype(int)

    m = stalled_metrics(y_stall, pred_stall, p_stall)
    return {"precision": m["precision"], "recall": m["recall"], "pr_auc": m["pr_auc"]}


def table1(
    mode: str = "forecast",
    seed: int = simulate.SEED,
    n_endpoints: int = simulate.N_ENDPOINTS,
    trials: int = simulate.TRIALS_PER_ENDPOINT,
) -> dict[str, Any]:
    """Component ablation: each feature removed, then the CUSUM stage removed."""
    data = simulate.build(n_endpoints=n_endpoints, trials_per_endpoint=trials, seed=seed)
    train, test = data["train"], data["test"]

    feature_map = TABLE1_FEATURES[mode]
    rows: dict[str, dict[str, float]] = {}
    rows["full canary"] = _score(train, test, [], mode, cusum_enabled=True)
    for label, cols in feature_map.items():
        rows[f"- {label}"] = _score(train, test, cols, mode, cusum_enabled=True)
    rows["- CUSUM (threshold only)"] = _score(train, test, [], mode, cusum_enabled=False)

    # The paper's caption claims this invariant; check it rather than trust it.
    pr_full = rows["full canary"]["pr_auc"]
    pr_nocusum = rows["- CUSUM (threshold only)"]["pr_auc"]
    return {
        "mode": mode,
        "seed": seed,
        "rows": rows,
        "pr_auc_invariant_holds": bool(abs(pr_full - pr_nocusum) < 1e-12),
    }


# --------------------------------------------------------------------------- #
# Table 3 — savings grid
# --------------------------------------------------------------------------- #
def table3(
    mode: str = "forecast",
    seed: int = simulate.SEED,
    prices: tuple[float, ...] = PRICES_USDC,
    stall_rates: tuple[float, ...] = STALL_RATES,
    n_boot: int = 500,
) -> dict[str, Any]:
    """DR savings per routed call across price and stall base rate.

    Stall base rate is varied by re-weighting the corpus composition: the share
    of endpoints drawn from the settled-but-stalled profile.
    """
    grid: dict[str, dict[str, dict[str, float]]] = {"soft": {}, "ev": {}}

    for stall_rate in stall_rates:
        n = simulate.N_ENDPOINTS
        n_stalled = int(round(n * stall_rate))
        n_rest = n - n_stalled
        n_honest = n_rest - n_rest // 2
        n_degraded = n_rest // 2

        data = simulate.build(n_endpoints=n, seed=seed)
        endpoints = data["endpoints"]
        # Reassign profiles to hit the requested stall base rate, keeping the
        # same endpoint ids, prices and RNG stream.
        wanted = ["stalled"] * n_stalled + ["honest"] * n_honest + ["degraded"] * n_degraded
        for ep, profile in zip(endpoints, wanted):
            ep.profile = profile

        rng = np.random.default_rng(seed)
        rows = simulate.generate_probes(endpoints, simulate.TRIALS_PER_ENDPOINT, rng=rng)
        train, test = simulate.train_test_split(rows)

        predictor = build_predictor(mode).fit(train)
        p_all = predictor.predict_proba_delivered(rows)

        for price in prices:
            res = offpolicy.evaluate_policies(
                rows,
                p_all,
                value=CALL_VALUE_USDC,
                policies=("soft", "ev"),
                n_boot=n_boot,
                price_override=price,
            )
            for policy in ("soft", "ev"):
                grid[policy].setdefault(f"{stall_rate:.2f}", {})[f"{price:.3f}"] = {
                    "dr": res[policy]["dr"],
                    "ci_low": res[policy]["dr_ci_low"],
                    "ci_high": res[policy]["dr_ci_high"],
                }

    return {
        "mode": mode,
        "seed": seed,
        "value_usdc": CALL_VALUE_USDC,
        "prices": list(prices),
        "stall_rates": list(stall_rates),
        "grid": grid,
    }


# --------------------------------------------------------------------------- #
def _fmt_table1(t1: dict[str, Any]) -> str:
    lines = [
        f"Table 1 — component ablation ({t1['mode']} mode, seed {t1['seed']})",
        "",
        f"{'Configuration':30s} {'Stalled P':>10s} {'Stalled R':>10s} {'PR-AUC':>10s}",
        "-" * 63,
    ]
    for name, m in t1["rows"].items():
        lines.append(
            f"{name:30s} {m['precision']:10.3f} {m['recall']:10.3f} {m['pr_auc']:10.3f}"
        )
    lines.append("")
    lines.append(
        f"PR-AUC unchanged when only CUSUM is removed: {t1['pr_auc_invariant_holds']}"
    )
    return "\n".join(lines)


def _fmt_table3(t3: dict[str, Any]) -> str:
    lines = [f"Table 3 — DR USDC saved per routed call ({t3['mode']} mode, seed {t3['seed']})"]
    for policy in ("soft", "ev"):
        tag = "pay w.p. p-hat" if policy == "soft" else "pay iff p*v > c"
        lines += ["", f"  policy: {policy} ({tag})", ""]
        header = f"  {'Stall rate':>11s}" + "".join(f"{'$' + f'{p:.3f}':>14s}" for p in t3["prices"])
        lines += [header, "  " + "-" * (11 + 14 * len(t3["prices"]))]
        for sr in t3["stall_rates"]:
            row = f"  {sr:>11.2f}"
            for p in t3["prices"]:
                v = t3["grid"][policy][f"{sr:.2f}"][f"{p:.3f}"]["dr"]
                row += f"{v:>14.3e}"
            lines.append(row)
    return "\n".join(lines)


def run(out_dir: Path | str | None = None, mode: str = "forecast", n_boot: int = 500) -> dict:
    t1 = table1(mode=mode)
    t3 = table3(mode=mode, n_boot=n_boot)
    print(_fmt_table1(t1))
    print()
    print(_fmt_table3(t3))

    out = {"table1": t1, "table3": t3}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ablations.json").write_text(json.dumps(out, indent=2, default=float))
        (out_dir / "ablations.txt").write_text(_fmt_table1(t1) + "\n\n" + _fmt_table3(t3) + "\n")
        print(f"\nwrote {out_dir}/ablations.json")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute the paper's Table 1 and Table 3")
    ap.add_argument("--mode", choices=("forecast", "oracle"), default="forecast")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[2] / "results" / "ablations")
    )
    args = ap.parse_args(argv)
    run(out_dir=args.out, mode=args.mode, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
