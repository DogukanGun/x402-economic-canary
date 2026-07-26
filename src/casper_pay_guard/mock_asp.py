"""A local Agent Service Provider that misbehaves on demand.

This is the fault-injection rig. It speaks real HTTP and a real x402 challenge,
so the handshake client, the signer and the delivery oracle are exercised over
the wire — no mocking of our own code — while costing nothing.

Each behaviour is a path::

    /asp/honest         402 -> 200 + schema-valid artifact
    /asp/degraded       402 -> 200 + schema-violating artifact
    /asp/degraded_empty 402 -> 200 + empty body
    /asp/stalled_5xx    402 -> 502                    (settled, error back)
    /asp/stalled_slow   402 -> 200, but only after maxTimeoutSeconds elapses
    /asp/stalled_silent 402 -> takes the money, never answers at all
    /asp/no_402         200 straight away, never prices the call

``unreachable`` needs no route: point the canary at a closed port.

Note which bucket the empty body lands in. The paper puts "settled, 2xx, empty
body" in ``degraded`` — the money is still lost, but the provider did answer.
``stalled`` is reserved for settled calls with no 2xx inside
``maxTimeoutSeconds``. The two are separated deliberately (Section 4).

Every priced path returns a well-formed 402 first, which is exactly why an
HTTP-liveness monitor scores all of them healthy.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

#: The artifact schema every priced path advertises in its 402 challenge.
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["result", "model"],
    "properties": {"result": {"type": "number"}, "model": {"type": "string"}},
}

#: Minimum advertised price: 1000 atomic units = $0.001 USDC.
PRICE_ATOMIC = "1000"

#: Deliberately short so a stalled_slow probe resolves in about a second.
MAX_TIMEOUT_SECONDS = 1

MODES = (
    "honest",
    "degraded",
    "degraded_empty",
    "stalled_5xx",
    "stalled_slow",
    "stalled_silent",
    "no_402",
)

#: What each mode *should* be graded as by the delivery oracle.
EXPECTED_LABEL = {
    "honest": "delivered",
    "degraded": "degraded",
    "degraded_empty": "degraded",
    "stalled_5xx": "stalled",
    "stalled_slow": "stalled",
    "stalled_silent": "stalled",
    "no_402": "unreachable",
}


def challenge_body(resource: str) -> dict:
    """A spec-shaped ``402 Payment Required`` body."""
    return {
        "x402Version": 1,
        "error": "X-PAYMENT header is required",
        "accepts": [
            {
                "scheme": "exact",
                "network": "base-sepolia",
                "maxAmountRequired": PRICE_ATOMIC,
                "resource": resource,
                "description": "mock inference call",
                "mimeType": "application/json",
                "outputSchema": OUTPUT_SCHEMA,
                "payTo": "0x000000000000000000000000000000000000dEaD",
                "maxTimeoutSeconds": MAX_TIMEOUT_SECONDS,
                "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                "extra": {"name": "USDC", "version": "2"},
            }
        ],
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Keep the test output clean.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        return

    def handle_one_request(self) -> None:
        # The canary hangs up on providers it has given up on, so resets and
        # broken pipes are the expected outcome of the stalled modes, not faults.
        with contextlib.suppress(OSError):
            super().handle_one_request()
            return
        self.close_connection = True

    # -- helpers ---------------------------------------------------------
    # A stalling provider is one the canary gives up on, so the client hangs up
    # mid-response by design. Writing to that closed socket is expected, not an
    # error; swallow it so it does not look like a bug in the test output.
    def _send_json(self, status: int, obj: object) -> None:
        raw = json.dumps(obj).encode()
        with contextlib.suppress(OSError):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    def _send_empty(self, status: int) -> None:
        with contextlib.suppress(OSError):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "0")
            self.end_headers()

    # -- routing ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        mode = self.path.rsplit("/", 1)[-1].split("?")[0]
        if mode not in MODES:
            self._send_json(404, {"error": f"unknown mode {mode!r}", "modes": list(MODES)})
            return

        # An endpoint that never prices the call: the canary never gets to pay.
        if mode == "no_402":
            self._send_json(200, {"result": 1.0, "model": "free-tier"})
            return

        paid = self.headers.get("X-PAYMENT")
        if not paid:
            # The signal every liveness monitor sees, and scores as healthy.
            self._send_json(402, challenge_body(f"http://{self.headers.get('Host', '')}{self.path}"))
            return

        # Payment authorization presented. From here the provider has been paid
        # regardless of what it does next — that asymmetry is the attack.
        if mode == "honest":
            self._send_json(200, {"result": 42.0, "model": "mock-embed-v1"})
        elif mode == "degraded":
            # 2xx, in time, but violates the schema it advertised.
            self._send_json(200, {"oops": True, "model": 7})
        elif mode == "degraded_empty":
            self._send_empty(200)
        elif mode == "stalled_5xx":
            self._send_json(502, {"error": "upstream unavailable"})
        elif mode == "stalled_slow":
            # Answers eventually, but past the window it advertised.
            time.sleep(MAX_TIMEOUT_SECONDS + 1.0)
            self._send_json(200, {"result": 42.0, "model": "mock-embed-v1"})
        elif mode == "stalled_silent":
            # Holds the connection open and never sends a status line at all.
            time.sleep(MAX_TIMEOUT_SECONDS + 5.0)
            self._send_empty(200)


class MockASP:
    """A running mock ASP. Use :func:`mock_asp` unless you need manual control."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> MockASP:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[0], self._server.server_address[1]
        return f"http://{host}:{port}"

    def url(self, mode: str) -> str:
        """URL for one behaviour, e.g. ``url("stalled_empty")``."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        return f"{self.base_url}/asp/{mode}"

    def closed_port_url(self) -> str:
        """A URL that refuses connections — the ``unreachable`` case."""
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        return f"http://127.0.0.1:{free_port}/asp/honest"


@contextlib.contextmanager
def mock_asp(host: str = "127.0.0.1", port: int = 0) -> Iterator[MockASP]:
    """Run a mock ASP on an ephemeral port for the duration of the block."""
    server = MockASP(host, port).start()
    try:
        yield server
    finally:
        server.stop()


def main() -> int:
    """Run the mock ASP in the foreground for manual probing."""
    import argparse

    ap = argparse.ArgumentParser(description="Run a misbehaving mock x402 ASP")
    ap.add_argument("--port", type=int, default=8402)
    args = ap.parse_args()

    server = MockASP(port=args.port).start()
    print(f"mock ASP listening on {server.base_url}")
    for mode in MODES:
        print(f"  {EXPECTED_LABEL[mode]:12s} <- {server.url(mode)}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
