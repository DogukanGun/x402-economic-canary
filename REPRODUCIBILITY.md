# Reproducibility

This repository backs the paper *"Economic Canaries for Pay-Then-Deliver Agent
Markets: Detecting Settled-but-Stalled x402 Service Providers Before You Route"*
([Zenodo 10.5281/zenodo.21515696](https://zenodo.org/record/21515696)).

Two configurations ship, deliberately:

| | `casper_pay_guard.repro` | `casper_pay_guard.experiment` |
|---|---|---|
| purpose | reproduce the published numbers | implement Section 5 as written |
| corpus | 60 endpoints x 120 trials = **7,200** probes | 120 endpoints x 150 trials = **18,000** probes |
| seeds | 7 (corpus), 11 (probes), 2026 (CUSUM), 2027 (bandit), 5 (bootstrap) | 42 |
| profiles | 4 states @ 55/22/15/8 % | 3 profiles @ 40/40/40 |
| classifier | `StandardScaler` + `LogisticRegression(C=2)` | `HistGradientBoostingClassifier` + isotonic `CalibratedClassifierCV(cv=5)` |
| split | 50/50 at `t=60` | time-based 60/40, per-provider stratified |
| status | **frozen — do not refactor** | the maintained implementation |

## The published numbers reproduce exactly

Not to a tolerance — bit-for-bit, all 22 values. `make reproduce`, asserted by
`tests/test_reproduce.py`:

| metric | value |
|---|---|
| stalled precision / recall | 0.9943100995732574 / 0.9831223628691983 |
| F2 / PR-AUC | 0.9853397237101776 / 0.8610113091299968 |
| Brier / reliability / resolution | 0.0028841377704232653 / 0.0006504973731149726 / 0.1990799821388987 |
| CUSUM median / mean / p95 / rate | 2.0 / 2.625 / 5.0 / 1.0 |
| ARL to false alarm | 1465.1125 |
| DR saved [CI] | 0.0002782179371421615 [0.00026011857979146597, 0.0002951632071679316] |
| IPS / SNIPS | 0.00022292979410728012 / 3.999469496261156e-05 |
| McNemar p (liveness / ratings) | 1.8916719623966624e-196 / 5.9042090569010235e-182 |
| DeLong canary AUC | 0.961870015710204 |
| validation | 14/14 |

This requires the exact stack the original run used, which `requirements.txt`
pins: **Python 3.11, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0**. The
`repro/` package preserves RNG call ordering; moving a single `rng` draw shifts
the PCG64 stream and the numbers change.

## Where the paper's text and its code disagree

The paper's Methods section describes a different experiment from the one that
produced its numbers. Both are now implemented; neither is hidden.

### 1. The corpus and classifier (Section 5)

Section 5 describes 18,000 probes over 120 endpoints at seed 42, with gradient
boosting and isotonic calibration. The code that produced every published
figure used 7,200 probes over 60 endpoints at seeds 7/11, with logistic
regression. The table at the top of this file lists every difference. The paper
also says USDC throughout while the original code said USDT; this repository is
USDC on Base Sepolia (chain 84532) end to end.

### 2. Post-hoc oracle vs forecast (Sections 3–4)

The published predictor's features come from **the probe being labelled**:
`artifact_valid`, `schema_valid`, `http_status`, that probe's own gap
(`repro/canary.py:16-28`). That grades a call which already happened.

The abstract promises something else — "an honest probability that your **next**
paid call will deliver" — and Section 3 defines features `z_{e,t}` as *history*
(rolling delivery rate, p95 latency, gap history). Both are implemented:

| framing | features | target | stalled P | stalled R | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| oracle | same-probe observables | this call | 1.000 | 1.000 | 0.998 | 0.0000 |
| forecast | causal endpoint history | next call | 0.944 | 0.723 | 0.898 | 0.110 |

Oracle mode reaches a perfect 1.000 on the paper-spec corpus. That is not a
better detector; it is close to a tautology, because obtaining its features *is*
the paid call. It cannot be consulted before routing. Forecast mode is the only
framing that can, and it is strictly harder.

**Forecast mode misses two of the paper's Table 2 targets:** `recall >= 0.85`
(0.723) and `brier <= 0.10` (0.110). Reported as computed. Nothing was tuned to
close them.

### 3. Three results had no generating code at all

Table 1, Table 3, and the 95% CIs on precision/recall/PR-AUC/Brier are printed
in the paper, but nothing in the original experiment computed them — only the
doubly-robust savings CI was bootstrapped. All three are now implemented
(`metrics/bootstrap.py`, `ablation.py`) and compared against the printed values
in `results/paper_delta.md`.

What survives: **Table 1's ranking reproduces in oracle mode** — removing the
settlement-to-response gap is the only ablation that moves precision or recall,
as the paper claims — though the magnitude is much smaller than the printed drop
to 0.762 PR-AUC. **Table 3's structure reproduces**: savings scale roughly
linearly in both price and stall base rate, and every cell is positive under the
expected-value policy.

The bootstrap CIs here are **clustered by endpoint**. Probes from one provider
are correlated — a stalling endpoint emits runs of failures — so resampling
individual probes would understate the intervals.

### 4. The routing policy decides the sign of the savings

The published run evaluated one policy: *pay with probability p̂*. Section 4
motivates a different one — "a well-calibrated probability translates directly
into correct expected-value routing" — which is the threshold rule *pay iff
p̂ · value > price*.

The difference is not cosmetic. At value \$0.02 and price \$0.001, a provider
that delivers 65 % of the time is worth paying **every** time
(0.65 × 0.02 = \$0.013 ≫ \$0.001), yet the soft policy skips it a third of the
time and forgoes more value than the stalls it avoids:

| policy | paper-spec / forecast, DR saved per routed call |
|---|---|
| soft (pay w.p. p̂) | **−1.57e-03** |
| ev (pay iff p̂·v > c) | **+9.96e-05**, CI [6.22e-05, 1.37e-04] |

The paper's qualitative claim — routing on canary verdicts beats always-pay —
holds, but only under the decision rule Section 4 describes, not the one the
code evaluated. In the published run the two agree, because oracle-mode
probabilities sit near 0 or 1 and the soft policy rarely abstains.

A related conflation: the frozen code reuses `p̂` as both the action probability
`π_e` and the reward model `q̂(x, pay)`. Those are different objects; they
coincide only under the soft policy. `metrics/offpolicy.py` separates them.

### 5. Two modelling artifacts found while implementing Section 5

Both would have inflated the reported scores, and both are corrected in
`simulate.py` with the reasoning inline:

- **"Gap clipped to `maxTimeoutSeconds` on stalled ones"**, read literally, makes
  every stalled gap one identical constant that no delivered probe can reach.
  One feature would then separate the classes perfectly and the scores would
  measure the simulator. Real stalls arrive two ways — a fast 5xx and a silent
  timeout — so the mixture is modelled and clipped at the cap.
- **`artifact_valid` as a deterministic function of `delivered`** leaks the label
  outright. A genuinely delivered artifact still occasionally fails verification
  (hash mismatch, truncation); the published simulator used 0.985, and so does
  this one.

## What is not claimed

- **No live rails.** Every number here is simulated or comes from the local mock
  ASP. No mainnet or testnet transaction has been broadcast.
- **`HttpFacilitator` is unexercised.** It is wired, typed, and reachable via
  `CASPER_FACILITATOR_URL`, but settling for real needs a funded payer. That is
  the paper's stated next step, not a result it claims.
- **Signing is real; settlement is stubbed.** `Eip3009Signer` produces genuine
  EIP-712 signatures that recover to the payer address
  (`tests/test_x402.py::test_signature_recovers_to_the_payer`). The default
  facilitator mints deterministic receipts against a SQLite ledger and never
  touches a chain. `SettlementReceipt.on_chain` is `False` for every receipt in
  this repository, and the stub can never set it `True`.

## Reproducing everything

```bash
make setup       # pinned venv: python 3.11, numpy 2.4.4, scipy 1.17.1, sklearn 1.8.0
make reproduce   # published numbers, bit-for-bit, 14/14
make report      # runs every configuration -> results/paper_delta.md
make validate    # 14 criteria x 3 configurations
make test        # 106 tests incl. real-HTTP e2e and MCP stdio smoke
```
