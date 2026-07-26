"""The 14 validation criteria, run against every configuration.

These are the paper's Table 2 acceptance targets plus the significance tests and
the MCP smoke check. The published configuration passes all 14 — it is the
configuration they were chosen for. The paper-spec configuration is run against
the same bar and its misses are printed, not hidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CRITERIA_ORDER = [
    "precision>=0.90",
    "recall>=0.85",
    "pr_auc>=0.80",
    "brier<=0.10",
    "reliability<0.02",
    "cusum_latency<=3",
    "arl>=500",
    "usdc_saved_all_positive",
    "dr_ci_excludes_zero",
    "mcnemar_liveness_p<0.05",
    "mcnemar_ratings_p<0.05",
    "delong_liveness_p<0.05",
    "delong_ratings_p<0.05",
    "mcp_smoke_all_four_labels",
]


def run_all(n_boot: int = 2000) -> dict[str, dict[str, Any]]:
    """Run every configuration and collect its validation verdict."""
    from casper_pay_guard import experiment
    from casper_pay_guard.repro.run import run as run_published

    return {
        "published (frozen)": run_published()["metrics"],
        "paper-spec / oracle": experiment.run(mode="oracle", n_boot=n_boot)["metrics"],
        "paper-spec / forecast": experiment.run(mode="forecast", n_boot=n_boot)["metrics"],
    }


def format_report(results: dict[str, dict[str, Any]]) -> str:
    names = list(results)
    width = max(len(c) for c in CRITERIA_ORDER) + 2
    cols = [max(len(n), 8) for n in names]

    lines = ["Validation criteria (paper Table 2 + significance + MCP smoke)", ""]
    lines.append("criterion".ljust(width) + "".join(n.center(c + 3) for n, c in zip(names, cols)))
    lines.append("-" * (width + sum(c + 3 for c in cols)))

    for crit in CRITERIA_ORDER:
        row = crit.ljust(width)
        for name, c in zip(names, cols):
            v = results[name].get("validation_criteria", {}).get(crit)
            mark = "PASS" if v else ("FAIL" if v is not None else "—")
            row += mark.center(c + 3)
        lines.append(row)

    lines.append("-" * (width + sum(c + 3 for c in cols)))
    total = "TOTAL".ljust(width)
    for name, c in zip(names, cols):
        m = results[name]
        total += f"{m.get('validation_passed', 0)}/{m.get('validation_total', 14)}".center(c + 3)
    lines.append(total)
    lines.append("")

    for name in names:
        failed = [k for k, v in results[name].get("validation_criteria", {}).items() if not v]
        if failed:
            lines.append(f"{name} misses: {', '.join(failed)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Run the 14 validation criteria on all configs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=str(root / "results" / "validation.json"))
    args = ap.parse_args(argv)

    results = run_all(n_boot=args.n_boot)
    report = format_report(results)
    print(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                name: {
                    "criteria": m.get("validation_criteria", {}),
                    "passed": m.get("validation_passed"),
                    "total": m.get("validation_total"),
                }
                for name, m in results.items()
            },
            indent=2,
        )
    )
    out.with_suffix(".txt").write_text(report + "\n")
    print(f"\nwrote {out}")

    # The frozen path is the contract: it must be perfect.
    published = results["published (frozen)"]
    return 0 if published["validation_passed"] == published["validation_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
