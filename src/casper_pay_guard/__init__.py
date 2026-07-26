"""casper_pay_guard — an economic canary for pay-then-deliver x402 agent markets.

x402 is trustless at the payment layer and trust-maximal at the delivery layer:
a confirmed settlement carries no guarantee that usable output follows. This
package detects the resulting *settled-but-stalled* failure by paying like a
customer and checking like a skeptic — completing the full handshake at the
minimum advertised price, then verifying the returned artifact against the
provider's advertised schema.

Reference: "Economic Canaries for Pay-Then-Deliver Agent Markets: Detecting
Settled-but-Stalled x402 Service Providers Before You Route" (2026),
https://zenodo.org/record/21515696

See REPRODUCIBILITY.md for how the published numbers map onto this code.
"""

__version__ = "0.1.0"
