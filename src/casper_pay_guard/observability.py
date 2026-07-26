"""Prometheus counters and structured logging (paper Section 4).

Exports delivery rate, p95 latency, settlement-to-response gap and the
stalled-rejection rate, so an agent operator can see the thing the whole system
exists to surface: *money settled that produced nothing*.

Metrics are process-global and registered once; importing this module twice does
not double-register.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from casper_pay_guard.x402.types import ProbeResult

REGISTRY = CollectorRegistry()

PROBES = Counter(
    "casper_probes_total",
    "Priced canary probes completed, by verdict.",
    ["label", "endpoint"],
    registry=REGISTRY,
)
SPEND = Counter(
    "casper_spend_usdc_total",
    "USDC settled by the canary, by verdict. Everything but `delivered` is waste.",
    ["label", "endpoint"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "casper_probe_latency_ms",
    "End-to-end probe latency in milliseconds.",
    ["endpoint"],
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
    registry=REGISTRY,
)
GAP = Histogram(
    "casper_settlement_to_response_gap_ms",
    "Milliseconds between settlement and a usable answer. The stall signal.",
    ["endpoint"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
    registry=REGISTRY,
)
DELIVERY_RATE = Gauge(
    "casper_delivery_rate",
    "Rolling fraction of probes that actually delivered, per endpoint.",
    ["endpoint"],
    registry=REGISTRY,
)
STALLED_REJECTION_RATE = Gauge(
    "casper_stalled_rejection_rate",
    "Fraction of probes rejected as stalled, per endpoint.",
    ["endpoint"],
    registry=REGISTRY,
)
DELIVERY_PROBABILITY = Gauge(
    "casper_predicted_delivery_probability",
    "Calibrated P(next paid call delivers), per endpoint.",
    ["endpoint"],
    registry=REGISTRY,
)

_counts: dict[str, dict[str, int]] = {}


def _endpoint_label(url: str | None) -> str:
    """Strip query strings so cardinality stays bounded."""
    if not url:
        return "unknown"
    return url.split("?", 1)[0]


def record(result: ProbeResult, p_delivered: float | None = None) -> None:
    """Fold one probe result into the exported metrics."""
    ep = _endpoint_label(result.target_url)
    PROBES.labels(label=result.label, endpoint=ep).inc()

    if result.receipt is not None:
        SPEND.labels(label=result.label, endpoint=ep).inc(result.receipt.price_usdc)
    if result.latency_ms:
        LATENCY.labels(endpoint=ep).observe(result.latency_ms)
    if result.settlement_to_response_gap_ms is not None:
        GAP.labels(endpoint=ep).observe(result.settlement_to_response_gap_ms)

    c = _counts.setdefault(ep, {"n": 0, "delivered": 0, "stalled": 0})
    c["n"] += 1
    c["delivered"] += result.label == "delivered"
    c["stalled"] += result.label == "stalled"
    DELIVERY_RATE.labels(endpoint=ep).set(c["delivered"] / c["n"])
    STALLED_REJECTION_RATE.labels(endpoint=ep).set(c["stalled"] / c["n"])

    if p_delivered is not None:
        DELIVERY_PROBABILITY.labels(endpoint=ep).set(p_delivered)


def snapshot() -> str:
    """Render the current metrics in Prometheus text exposition format."""
    from prometheus_client import generate_latest

    return generate_latest(REGISTRY).decode()


# --------------------------------------------------------------------------- #
# Structured logging
# --------------------------------------------------------------------------- #
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if extra := getattr(record, "extra_fields", None):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str = "casper_pay_guard", level: int = logging.INFO) -> logging.Logger:
    """A logger that writes JSON to **stderr**.

    stderr matters: an MCP stdio server owns stdout for protocol frames, and a
    stray log line there corrupts the session.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_probe(logger: logging.Logger, result: ProbeResult) -> None:
    logger.info(
        f"probe {result.label}",
        extra={
            "extra_fields": {
                "label": result.label,
                "target_url": result.target_url,
                "http_status": result.http_status,
                "latency_ms": round(result.latency_ms, 2),
                "gap_ms": (
                    round(result.settlement_to_response_gap_ms, 2)
                    if result.settlement_to_response_gap_ms is not None
                    else None
                ),
                "schema_valid": result.schema_valid,
                "price_usdc": result.receipt.price_usdc if result.receipt else 0.0,
                "tx_hash": result.receipt.tx_hash if result.receipt else None,
            }
        },
    )
