"""FROZEN reproduction path — do not refactor.

This package is a near-verbatim port of the original experiment that produced
every number published in the paper:

    thinker/outputs/topic-an-agent-facing-an-x402-challenge-from-a-c20321/experiment/
        simulate.py  canary.py  metrics.py  ope.py  main.py

Its only job is bit-for-bit reproduction. The RNG call order, the seeds, the
50/50 split, the LogisticRegression hyper-parameters and the CUSUM constants are
all load-bearing: reorder a single ``rng`` draw and the PCG64 stream shifts and
the published figures stop matching.

The clean, refactored architecture the paper *describes* lives one level up
(``casper_pay_guard.x402``, ``.oracle``, ``.predictor``, ...). Improvements go
there. Nothing in this package should be "tidied".

See REPRODUCIBILITY.md for the full paper-text-vs-code delta.
"""
from casper_pay_guard.repro import canary, metrics, ope, simulate  # noqa: F401
from casper_pay_guard.repro.run import PUBLISHED, run  # noqa: F401

__all__ = ["simulate", "canary", "metrics", "ope", "run", "PUBLISHED"]
