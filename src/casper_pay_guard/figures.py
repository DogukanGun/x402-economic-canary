"""Figures 3 and 4 from the paper.

Figure 3 — distribution of key metrics across bootstrap resamples.
Figure 4 — the canary against the three incumbent baselines.

Both are regenerated from computed results, so they move when the numbers move.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CANARY_COLOR = "#2a9d8f"
BASELINE_COLORS = ["#e76f51", "#e9c46a", "#a8763e"]


def figure3(results: dict[str, Any], out_path: Path) -> Path:
    """Metric distributions with their bootstrap confidence intervals."""
    m = results["metrics"]
    c = m["economic-canary"]
    ci = m.get("confidence_intervals", {})

    keys = [
        ("stalled_precision", "stalled\nprecision", c["precision"]),
        ("stalled_recall", "stalled\nrecall", c["recall"]),
        ("pr_auc", "PR-AUC", c["pr_auc"]),
        ("brier", "Brier\n(lower better)", c["brier"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(keys))
    points, lows, highs, labels = [], [], [], []
    for key, label, point in keys:
        b = ci.get(key, {})
        lo = b.get("ci_low", point)
        hi = b.get("ci_high", point)
        points.append(point)
        lows.append(max(point - lo, 0))
        highs.append(max(hi - point, 0))
        labels.append(label)

    ax.errorbar(
        xs,
        points,
        yerr=[lows, highs],
        fmt="o",
        color=CANARY_COLOR,
        capsize=6,
        markersize=9,
        linewidth=2,
    )
    for x, p in zip(xs, points):
        ax.annotate(f"{p:.3f}", (x, p), textcoords="offset points", xytext=(12, 0), fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    cfg = results["config"]
    ax.set_title(
        f"Economic canary — headline metrics with 95% CIs\n"
        f"{cfg.get('n_probes', '?')} probes, {cfg.get('n_endpoints', '?')} endpoints "
        f"({cfg.get('mode', 'published')} mode)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def figure4(results: dict[str, Any], out_path: Path) -> Path:
    """Canary vs the three incumbents on the stalled class."""
    m = results["metrics"]
    systems = ["economic-canary"] + list(results["baselines"])
    display = ["economic\ncanary", "always-pay\n(APEX)", "402-as-healthy\nliveness", "marketplace\nratings"]
    metrics = [("precision", "precision"), ("recall", "recall"), ("pr_auc", "PR-AUC")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        vals = [m[s].get(key, 0.0) for s in systems]
        colors = [CANARY_COLOR] + BASELINE_COLORS[: len(systems) - 1]
        bars = ax.bar(range(len(systems)), vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}",
                (bar.get_x() + bar.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(display[: len(systems)], fontsize=8)
        ax.set_title(f"stalled-class {title}", fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")

    sig = m.get("significance", {}).get("mcnemar", {})
    p_live = sig.get("402-as-healthy-liveness", {}).get("p_value")
    note = ""
    if p_live is not None:
        note = f"  (McNemar vs liveness p = {p_live:.1e})"
    fig.suptitle(
        "Only a signal that pays can see the settled-but-stalled class" + note, fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Regenerate Figures 3 and 4")
    ap.add_argument("--mode", choices=("forecast", "oracle"), default="forecast")
    ap.add_argument("--out", default=str(root / "results" / "figures"))
    args = ap.parse_args(argv)

    from casper_pay_guard import experiment

    results = experiment.run(mode=args.mode)
    out = Path(args.out)
    p3 = figure3(results, out / f"figure3_metrics_{args.mode}.png")
    p4 = figure4(results, out / f"figure4_baselines_{args.mode}.png")
    print(f"wrote {p3}\nwrote {p4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
