"""
Deterministic tests for persistent exit completion in live_trader.run_cycle().

These drive the REAL run_cycle -- not a reimplementation of it -- against a
stubbed exchange whose book is deliberately thinner than the position, which is
the condition that produced the live failure on 2026-08-11:

    +54.3% -> ordered 25, filled  5   (20 abandoned)
    +44.4% -> ordered 20, filled  3   (17 abandoned)
    +49.2% -> ordered 17, filled  0   (17 rode to settlement)

The bug was NOT "the order didn't fill". Partial fills are normal and fine. The
bug was that the next cycle re-decided the exit from scratch, saw the price had
ticked back under +35%, and silently walked away from a position it had already
committed to closing. Test 1 encodes exactly that sequence.

    python test_exit_completion.py

No network, no credentials, no orders. Real demo execution is exercised
separately by live_exit_demo_test.py.
"""
import sys
from datetime import datetime, timedelta, timezone

import config
import live_trader as lt

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


class FakeBook:
    """An exchange with a fixed depth per order, like the demo book at size 25.

    depth=3 means any IOC for N contracts fills min(N, 3) and cancels the rest,
    which is what a real immediate-or-cancel does against a thin touch.
    """

    def __init__(self, contracts, avg_cost, depth=3, mins_left=8.0,
                 prices=None, ticker="TEST-MKT", stale_quote=None):
        self.ticker = ticker
        self.contracts = float(contracts)       # signed: >0 long YES, <0 short YES
        self.avg_cost = avg_cost
        self.depth = depth
        self.mins_left = mins_left
        self.prices = list(prices or [])        # TRUE book yes-price per cycle
        self.stale_quote = stale_quote          # what /markets/{t} wrongly says
        self.cycle = 0
        self.orders = []

    # --- the surface run_cycle actually touches ---------------------------
    def load_orders(self):
        return []

    def get_resting_orders(self):
        return []

    def reconcile(self):
        return {"in_sync": True, "positions_not_in_local_log": [],
                "local_fills_without_position": [], "local_unresolved": []}

    def position_summary(self):
        if self.contracts == 0:
            return []
        return [{"ticker": self.ticker, "contracts": self.contracts,
                 "exposure": abs(self.contracts) * self.avg_cost,
                 "fees_paid": 0.0, "realized_pnl": 0.0}]

    def place_order(self, ticker, side, count, price, time_in_force=None,
                    client_order_id=None, post_only=False, note=""):
        assert 0 < count <= config.LIVE_MAX_CONTRACTS, f"size gate breached: {count}"
        assert 0.0 < price < 1.0, f"bad price {price}"
        filled = min(count, self.depth)
        self.orders.append({"count": count, "filled": filled, "side": side,
                            "price": price, "note": note, "cycle": self.cycle})
        # reduce toward flat, respecting sign
        if self.contracts > 0:
            self.contracts = max(0.0, self.contracts - filled)
        else:
            self.contracts = min(0.0, self.contracts + filled)
        return {"status": "filled" if filled == count else
                          ("partial" if filled else "cancelled"),
                "fill_count": filled, "avg_fill_price": price,
                "fee_paid": 0.0, "order_id": "x", "client_order_id": "y"}

    def book(self, ticker=None, retries=None):
        """The touch. `stale_quote` lets a test make market_quote disagree with
        the book, the way the real feeds did on 2026-08-11."""
        px = self.prices[min(self.cycle, len(self.prices) - 1)]
        return {"yes_bid": px, "yes_bid_size": float(self.depth),
                "yes_ask": round(px + 0.01, 4), "yes_ask_size": float(self.depth)}

    def quote(self, ticker=None, retries=None):
        px = (self.stale_quote if self.stale_quote is not None
              else self.prices[min(self.cycle, len(self.prices) - 1)])
        closes = datetime.now(timezone.utc) + timedelta(minutes=self.mins_left)
        return {"status": "active", "ticker": self.ticker,
                "yes_bid_dollars": str(px), "yes_ask_dollars": str(round(px + 0.01, 4)),
                "close_time": closes.strftime("%Y-%m-%dT%H:%M:%SZ")}


def run(book, cycles, session=None, entries_blocked=True):
    """Drive N real run_cycle() calls against `book`."""
    session = session or {"utc_date": "2026-08-11", "realized_pnl": 0.0,
                          "entries": 0, "exits": 0, "halted_reason": ""}
    saved = (lt.ex, lt.market_quote, lt.book_touch, lt.kill_switch_active,
             lt.kalshi_auth._request, config.EXIT_SWEEP_PAUSE_SECONDS)
    lt.ex = book
    lt.market_quote = book.quote
    lt.book_touch = book.book
    lt.kill_switch_active = lambda: entries_blocked
    lt.kalshi_auth._request = lambda path, **kw: {}      # no settlements/positions
    config.EXIT_SWEEP_PAUSE_SECONDS = 0.0
    try:
        for i in range(cycles):
            book.cycle = i
            lt.run_cycle(session)
    finally:
        (lt.ex, lt.market_quote, lt.book_touch, lt.kill_switch_active,
         lt.kalshi_auth._request, config.EXIT_SWEEP_PAUSE_SECONDS) = saved
    return session


# ---------------------------------------------------------------------------
print("=" * 74)
print("TEST 1 -- THE REGRESSION: partial fill, then price falls back under target")
print("=" * 74)
# Entry at 0.40. Cycle 0 the bid is 0.62 (+55%, target is +35%) so the exit
# commits. Book only gives 3 per order. By cycle 1 the bid has slumped to 0.45
# (+12.5%, well under target) -- the OLD code stopped here and left 22 open.
book = FakeBook(contracts=25, avg_cost=0.40, depth=3,
                prices=[0.62, 0.45, 0.44, 0.43, 0.46, 0.44, 0.45, 0.43])
run(book, cycles=8)
print(f"  orders sent: {len(book.orders)}  fills: {sum(o['filled'] for o in book.orders)}"
      f"  contracts left: {book.contracts:.0f}")
check("keeps working the exit after price drops below target",
      len(book.orders) > 3, f"{len(book.orders)} orders across 8 cycles")
check("position reaches flat", book.contracts == 0,
      f"{book.contracts:.0f} contracts remaining")
committed_after_drop = [o for o in book.orders if o["cycle"] >= 1]
check("orders were placed AFTER the price fell under target",
      len(committed_after_drop) > 0,
      f"{len(committed_after_drop)} orders in cycles 1+ at gain ~+12%")
check("every order respected the size gate",
      all(0 < o["count"] <= config.LIVE_MAX_CONTRACTS for o in book.orders))
check("sweeps ran inside a single cycle",
      sum(1 for o in book.orders if o["cycle"] == 0) == config.EXIT_SWEEPS_PER_CYCLE,
      f"{sum(1 for o in book.orders if o['cycle'] == 0)} orders in cycle 0")

print()
print("=" * 74)
print("TEST 2 -- the OLD behaviour, to prove the test would have caught it")
print("=" * 74)
# Same book, but exit only if gain >= target at this instant (the old rule).
book2 = FakeBook(contracts=25, avg_cost=0.40, depth=3,
                 prices=[0.62, 0.45, 0.44, 0.43, 0.46, 0.44, 0.45, 0.43])
for i in range(8):
    book2.cycle = i
    q = book2.quote()
    if book2.contracts == 0:
        break
    gain = (float(q["yes_bid_dollars"]) - book2.avg_cost) / book2.avg_cost
    if gain >= config.PAPER_TRADE_EXIT_TARGET:
        book2.place_order(book2.ticker, "ask", int(abs(book2.contracts)),
                          float(q["yes_bid_dollars"]), note="old-style")
print(f"  orders sent: {len(book2.orders)}  contracts left: {book2.contracts:.0f}")
check("old logic strands the position (regression is real, not hypothetical)",
      book2.contracts > 0, f"{book2.contracts:.0f} of 25 left open, "
                           f"{len(book2.orders)} order(s) ever sent")

print()
print("=" * 74)
print("TEST 3 -- circuit breaker: window closing")
print("=" * 74)
book3 = FakeBook(contracts=25, avg_cost=0.40, depth=0, mins_left=1.0,
                 prices=[0.62] * 6)
run(book3, cycles=6)
check("stops trying inside EXIT_GIVE_UP_MINUTES", len(book3.orders) == 0,
      f"{len(book3.orders)} orders with {book3.mins_left}min to close "
      f"(cutoff {config.EXIT_GIVE_UP_MINUTES}min)")

print()
print("=" * 74)
print("TEST 4 -- circuit breaker: max attempts, and it STICKS")
print("=" * 74)
# Zero liquidity forever, price stays above target: without a sticky breaker the
# bot would re-commit every cycle and hammer the exchange indefinitely.
book4 = FakeBook(contracts=25, avg_cost=0.40, depth=0, mins_left=9.0,
                 prices=[0.62] * 25)
s4 = run(book4, cycles=25)
attempts = s4.get("exiting", {}).get(book4.ticker, {}).get("attempts", 0)
check("gives up after EXIT_MAX_ATTEMPTS cycles",
      attempts <= config.EXIT_MAX_ATTEMPTS,
      f"attempts={attempts}, cap={config.EXIT_MAX_ATTEMPTS}")
check("give-up is recorded, not re-decided next cycle",
      s4.get("exiting", {}).get(book4.ticker, {}).get("gave_up") == "max attempts",
      f"state={s4.get('exiting', {}).get(book4.ticker)}")
check("stops sending orders once given up",
      len(book4.orders) <= config.EXIT_MAX_ATTEMPTS * config.EXIT_SWEEPS_PER_CYCLE,
      f"{len(book4.orders)} orders over 25 cycles")

print()
print("=" * 74)
print("TEST 5 -- never commits when the target was never reached")
print("=" * 74)
book5 = FakeBook(contracts=25, avg_cost=0.40, depth=5, mins_left=9.0,
                 prices=[0.41, 0.44, 0.39, 0.52, 0.30, 0.45])   # max +30%
s5 = run(book5, cycles=6)
check("no exit orders below the +35% target", len(book5.orders) == 0,
      f"{len(book5.orders)} orders, best gain "
      f"{(max(book5.prices) - 0.40) / 0.40 * 100:+.0f}%")
check("no exit state created", not s5.get("exiting"))

print()
print("=" * 74)
print("TEST 6 -- NO-side (short YES) position exits with the right side/price")
print("=" * 74)
# Short 20 YES at 0.30 in NO terms => yes-cost 0.70... expressed the way
# position_summary reports it: contracts -20, exposure 20 * 0.30.
# NO value = 1 - yes_ask. yes_ask 0.55 => NO value 0.45 => +50% on cost 0.30.
book6 = FakeBook(contracts=-20, avg_cost=0.30, depth=4, mins_left=9.0,
                 prices=[0.54, 0.60, 0.62, 0.61, 0.63, 0.60, 0.64])
run(book6, cycles=7)
check("buys YES back (side=bid) to close a short", book6.orders and
      all(o["side"] == "bid" for o in book6.orders),
      f"sides={set(o['side'] for o in book6.orders)}")
check("orders at the YES ASK, not the NO price", book6.orders and
      all(o["price"] > 0.5 for o in book6.orders),
      f"prices={sorted(set(round(o['price'], 2) for o in book6.orders))}")
check("short position reaches flat", book6.contracts == 0,
      f"{book6.contracts:.0f} left")

print()
print("=" * 74)
print("TEST 7 -- exits still run while the kill switch blocks entries")
print("=" * 74)
book7 = FakeBook(contracts=10, avg_cost=0.40, depth=3, mins_left=9.0,
                 prices=[0.62] * 6)
s7 = run(book7, cycles=6, entries_blocked=True)
check("kill switch does not trap an open position", book7.contracts == 0,
      f"{len(book7.orders)} exit orders sent with entries halted")
check("no entries were opened", s7["entries"] == 0)

print()
print("=" * 74)
print("TEST 8 -- the stale market quote must not win over the real book")
print("=" * 74)
# The measured 2026-08-11 failure: /markets/{ticker} advertised bid 0.58 while
# the book had already collapsed to 0.10. Ordering at 0.58 fills nothing, and
# the gain looks like +45% when the position is actually deep under water.
book8 = FakeBook(contracts=25, avg_cost=0.40, depth=5, mins_left=9.0,
                 prices=[0.10] * 6, stale_quote=0.58)
s8 = run(book8, cycles=6)
check("no exit fires on a stale quote the book does not support",
      len(book8.orders) == 0,
      f"{len(book8.orders)} orders; quote said {book8.stale_quote} "
      f"(+45%), book said {book8.prices[0]} (-75%)")
check("no phantom commitment recorded", not s8.get("exiting"))

book8b = FakeBook(contracts=25, avg_cost=0.40, depth=5, mins_left=9.0,
                  prices=[0.62] * 6, stale_quote=0.20)
run(book8b, cycles=6)
check("exit DOES fire when the book supports it and the quote lags low",
      len(book8b.orders) > 0 and all(abs(o["price"] - 0.62) < 1e-9
                                     for o in book8b.orders),
      f"{len(book8b.orders)} orders, all priced at the book bid 0.62")

print()
print("=" * 74)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")
print("RESULT:", "ALL PASS" if not n_fail else f"{n_fail} FAILED")
sys.exit(1 if n_fail else 0)
