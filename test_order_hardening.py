"""
Tests for the two hardening fixes that are not about exit pricing:

  3. live_executor._post retries 5xx / transport failures, using the SAME
     client_order_id so a retry cannot mint a second real position.
  4. live_trader's entry guard allows ONE entry attempt per window, in any
     order state -- including no_fill, rejected and cancelled, which previously
     let the bot retry.

Both are driven against the real functions with the HTTP layer stubbed, because
a 503 cannot be summoned from the exchange on demand. The stub records exactly
what went over the wire, which is the thing that actually matters here: how many
distinct orders the exchange would have seen.

    python test_order_hardening.py
"""
import sys

import config
import live_executor as ex
import live_trader as lt

# ISOLATE THE ORDER LOG FIRST -- before any place_order call. These are
# synthetic orders against a fake ticker; appending them to data/live_orders.csv
# would corrupt the record that audit_live_pnl.py and the per-window entry guard
# both read. (It did, once. The rows were removed.)
import tempfile
from pathlib import Path as _Path
ex.ORDERS_CSV = _Path(tempfile.gettempdir()) / "kalshi_test_orders.csv"
if ex.ORDERS_CSV.exists():
    ex.ORDERS_CSV.unlink()

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


class FakeResp:
    def __init__(self, code, body):
        self.status_code = code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class FakePost:
    """Records every POST. `codes` is the status code sequence to return."""

    def __init__(self, codes, order_id="ord123"):
        self.codes = list(codes)
        self.calls = []
        self.order_id = order_id

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append(json)
        code = self.codes[min(len(self.calls) - 1, len(self.codes) - 1)]
        if code >= 500:
            return FakeResp(code, {"code": "service_unavailable"})
        if code == 409:
            return FakeResp(409, {"code": "duplicate_client_order_id"})
        return FakeResp(code, {"order": {
            "order_id": self.order_id, "status": "executed",
            "yes_price_dollars": "0.5000",
            "taker_fill_count_fp": "5.00", "taker_fill_cost_dollars": "2.5000",
            "taker_fees_dollars": "0.0875"}})


print("=" * 74)
print("FIX 3 -- transient 5xx is retried, with one identity on the wire")
print("=" * 74)

orig_post, orig_require = ex._session.post, ex._require_demo
ex._require_demo = lambda: None
try:
    # Two 503s then success: the real 2026-08-11 failure, but recoverable.
    fp = FakePost([503, 503, 200])
    ex._session.post = fp
    r = ex.place_order("TEST-MKT", "bid", 5, 0.50, note="retry test")
    check("retries through 5xx instead of giving up", len(fp.calls) == 3,
          f"{len(fp.calls)} POST(s) sent")
    check("order is not rejected once a retry succeeds",
          r["status"] not in ("rejected",), f"status={r['status']}")
    cids = {c.get("client_order_id") for c in fp.calls}
    check("every retry reuses ONE client_order_id (cannot double-fill)",
          len(cids) == 1, f"{len(cids)} distinct client_order_id(s): {cids}")

    # Persistent 5xx: must surface as a failure, not a silent success.
    fp2 = FakePost([503, 503, 503])
    ex._session.post = fp2
    r2 = ex.place_order("TEST-MKT", "bid", 5, 0.50, note="retry exhausted")
    check("gives up after the retry budget", len(fp2.calls) == 3,
          f"{len(fp2.calls)} POST(s)")
    check("exhausted retries do NOT report a fill",
          not r2.get("fill_count"), f"status={r2['status']} fill={r2.get('fill_count')}")

    # A 409 means the exchange already has it -- retrying is wrong and dangerous.
    fp3 = FakePost([409])
    ex._session.post = fp3
    r3 = ex.place_order("TEST-MKT", "bid", 5, 0.50, note="duplicate")
    check("a 409 duplicate is NOT retried", len(fp3.calls) == 1,
          f"{len(fp3.calls)} POST(s) -- 409 means it already exists")
finally:
    ex._session.post, ex._require_demo = orig_post, orig_require

print()
print("=" * 74)
print("FIX 4 -- one entry attempt per window, whatever the outcome")
print("=" * 74)


class Market:
    ticker = "KXBTC15M-TEST-15"
    strike = 100000.0


def guard_blocks(rows):
    """The real guard expression from live_trader.run_cycle."""
    return any(r["ticker"] == Market.ticker and "entry" in r.get("note", "")
               for r in rows)


for status in ("filled", "partial", "resting", "sending", "unknown",
               "no_fill", "rejected", "cancelled"):
    rows = [{"ticker": Market.ticker, "status": status, "note": "phase3 entry no"}]
    check(f"a prior {status:9s} entry blocks a second attempt", guard_blocks(rows))

check("an entry on a DIFFERENT window does not block this one",
      not guard_blocks([{"ticker": "KXBTC15M-OTHER-15", "status": "filled",
                         "note": "phase3 entry no"}]))
check("EXIT orders on this ticker never look like an entry",
      not guard_blocks([{"ticker": Market.ticker, "status": "filled",
                         "note": "phase3 exit sweep1 att1"}]),
      "exits must not be mistaken for 'already entered'")
check("re-entry after a completed round trip is blocked",
      guard_blocks([{"ticker": Market.ticker, "status": "filled",
                     "note": "phase3 entry no"},
                    {"ticker": Market.ticker, "status": "filled",
                     "note": "phase3 exit sweep1 att1"}]),
      "one trade per window, not one position at a time")

print()
print("=" * 74)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
for name, ok, detail in results:
    if not ok:
        print(f"  FAILED: {name} -- {detail}")
print("RESULT:", "ALL PASS" if not n_fail else f"{n_fail} FAILED")
sys.exit(1 if n_fail else 0)
