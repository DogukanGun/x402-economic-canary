"""Settlement-vs-output reconciliation ledger (paper Figure 1, layer 5).

Every priced probe writes two rows: one when the payment settles, one when (or
whether) the artifact arrives. Joining them is what makes "settled" and
"delivered" separable numbers rather than the same number, which is the entire
premise of the canary.

SQLite so a running canary fleet has durable spend accounting; ``:memory:`` for
tests.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from casper_pay_guard.x402.types import SettlementReceipt

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settlements (
    tx_hash       TEXT PRIMARY KEY,
    target_url    TEXT,
    payer         TEXT NOT NULL,
    pay_to        TEXT NOT NULL,
    network       TEXT NOT NULL,
    asset         TEXT NOT NULL,
    amount_atomic INTEGER NOT NULL,
    price_usdc    REAL NOT NULL,
    nonce         TEXT NOT NULL,
    settled_at    REAL NOT NULL,
    on_chain      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS deliveries (
    tx_hash      TEXT PRIMARY KEY,
    target_url   TEXT,
    label        TEXT NOT NULL,
    http_status  INTEGER,
    latency_ms   REAL,
    gap_ms       REAL,
    schema_valid INTEGER NOT NULL DEFAULT 0,
    observed_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_settlements_url ON settlements(target_url);
CREATE INDEX IF NOT EXISTS idx_deliveries_url  ON deliveries(target_url);
"""


@dataclass(frozen=True)
class EndpointSpend:
    """Reconciled spend for one endpoint: what we paid vs what we got."""

    target_url: str
    probes: int
    delivered: int
    stalled: int
    degraded: int
    unreachable: int
    spent_usdc: float
    wasted_usdc: float  # settled but not delivered

    @property
    def delivery_rate(self) -> float:
        return self.delivered / self.probes if self.probes else 0.0


class Ledger:
    """Thread-safe SQLite ledger of settlements and their delivery outcomes."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path not in (":memory:", ""):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes ------------------------------------------------------------
    def record_settlement(
        self, receipt: SettlementReceipt, target_url: str | None = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO settlements "
                "(tx_hash, target_url, payer, pay_to, network, asset, amount_atomic, "
                " price_usdc, nonce, settled_at, on_chain) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt.tx_hash,
                    target_url,
                    receipt.payer,
                    receipt.pay_to,
                    receipt.network,
                    receipt.asset,
                    receipt.amount_atomic,
                    receipt.price_usdc,
                    receipt.nonce,
                    receipt.settled_at,
                    int(receipt.on_chain),
                ),
            )
            self._conn.commit()

    def record_delivery(
        self,
        tx_hash: str,
        label: str,
        observed_at: float,
        target_url: str | None = None,
        http_status: int | None = None,
        latency_ms: float | None = None,
        gap_ms: float | None = None,
        schema_valid: bool = False,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO deliveries "
                "(tx_hash, target_url, label, http_status, latency_ms, gap_ms, "
                " schema_valid, observed_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    tx_hash,
                    target_url,
                    label,
                    http_status,
                    latency_ms,
                    gap_ms,
                    int(schema_valid),
                    observed_at,
                ),
            )
            self._conn.commit()

    # -- reads -------------------------------------------------------------
    def total_spent_usdc(self) -> float:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(price_usdc), 0) AS s FROM settlements").fetchone()
        return float(row["s"])

    def settlement_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM settlements").fetchone()
        return int(row["n"])

    def reconcile(self, target_url: str) -> EndpointSpend:
        """Join settlements to deliveries for one endpoint.

        ``wasted_usdc`` is the money that settled without producing a usable
        artifact — the number the whole system exists to surface.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.price_usdc AS price, d.label AS label "
                "FROM settlements s LEFT JOIN deliveries d USING (tx_hash) "
                "WHERE s.target_url = ?",
                (target_url,),
            ).fetchall()

        counts = {"delivered": 0, "stalled": 0, "degraded": 0, "unreachable": 0}
        spent = wasted = 0.0
        for r in rows:
            label = r["label"] or "unreachable"
            counts[label] = counts.get(label, 0) + 1
            spent += float(r["price"])
            if label != "delivered":
                wasted += float(r["price"])
        return EndpointSpend(
            target_url=target_url,
            probes=len(rows),
            delivered=counts["delivered"],
            stalled=counts["stalled"],
            degraded=counts["degraded"],
            unreachable=counts["unreachable"],
            spent_usdc=round(spent, 9),
            wasted_usdc=round(wasted, 9),
        )

    def endpoints(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT target_url FROM settlements WHERE target_url IS NOT NULL"
            ).fetchall()
        return [r["target_url"] for r in rows]

    def summary(self) -> dict[str, Any]:
        return {
            "settlements": self.settlement_count(),
            "total_spent_usdc": round(self.total_spent_usdc(), 9),
            "endpoints": {u: vars(self.reconcile(u)) for u in self.endpoints()},
        }
