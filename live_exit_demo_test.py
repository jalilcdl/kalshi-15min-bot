"""
REAL demo-exchange test of persistent exit completion, with real partial fills.

test_exit_completion.py proves the state machine against a stubbed book. This
proves it against the actual exchange, which is where the bug appeared: the
demo touch holds ~5 contracts, so any position bigger than that CANNOT be
closed by one IOC, and closing it requires exactly the cross-cycle persistence
being tested.

WHAT IT DOES
  1. finds the live KXBTC15M market with enough time left
  2. accumulates a position deliberately larger than the touch depth
  3. drives the REAL live_trader.run_cycle() until the position is flat or a
     circuit breaker records a give-up
  4. reconciles the result against /portfolio/positions and /portfolio/fills

TWO CONFIG OVERRIDES, and why they are honest:
  PAPER_TRADE_EXIT_TARGET -> -9.0    a +35% move cannot be summoned on demand;
  EXIT_COMMIT_FLOOR       -> -9.0    this forces the exit path to engage NOW.
The target rule itself is tested separately (test_exit_completion.py TEST 5 --
"never commits when the target was never reached"). What is under test here is
whether a committed exit actually finishes against a real thin book. Everything
else -- order construction, sides, sweeps, breakers, reconciliation -- is the
real shipped code, unmodified.

DEMO ONLY. Refuses to run otherwise. Costs real demo money: it crosses a wide
spread on purpose, which is the price of a genuine partial-fill test.

    python live_exit_demo_test.py
"""
import sys
import time
from datetime import datetime, timezone

import requests

import config
import kalshi_auth
import live_executor as ex
import live_trader as lt
from singleton import AlreadyRunning, SingleInstance

FLOOR_MODE = "--floor" in sys.argv   # keep the real floor; expect zero fills
TARGET_SIZE = 15          # ~3x the observed touch depth of 5
MAX_BUILD_TRIES = 10
CYCLES = 8
CYCLE_GAP = 20            # seconds; faster than the 60s production loop so a
                          # full test fits inside one 15-minute window


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def mins_left(m):
    ct = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
    return (ct - datetime.now(timezone.utc)).total_seconds() / 60.0


def pick_market():
    """A market with time left AND liquidity on both sides of the real book.

    Both sides matter: you enter against one and exit against the other. Near
    close this demo book degenerates to a single side (measured: yes bids
    5 @ $0.001, no bids empty, 0.9min to go), which makes an exit test
    impossible for reasons that have nothing to do with the code under test.
    Liquidity is judged from the ORDERBOOK, never from the quote fields --
    those lag by up to 29c (see live_trader.book_touch).
    """
    r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets",
                     params={"series_ticker": config.KALSHI_SERIES_TICKER,
                             "status": "open", "limit": 20}, timeout=20).json()
    # Never build on top of a position that is already open. Another diagnostic
    # holding 1 contract in the same market would be folded into this test's
    # average cost and its own settlement measurement would be corrupted by
    # this test's fills -- two experiments silently invalidating each other.
    busy = {p["ticker"] for p in ex.position_summary() if p["contracts"] != 0}
    best = None
    for m in r.get("markets", []) or []:
        if m["ticker"] in busy:
            continue
        ml = mins_left(m)
        if ml < 5.0 or ml > 20.0:
            continue
        b = lt.book_touch(m["ticker"])
        if not (b.get("yes_bid") and b.get("yes_ask")):
            continue
        if not (0.0 < b["yes_ask"] < 1.0 and 0.0 < b["yes_bid"] < 1.0):
            continue
        if best is None or ml > mins_left(best):
            best = m
    return best


def depth(ticker):
    return lt.book_touch(ticker)


def main():
    if config.KALSHI_ENV != "demo":
        print(f"REFUSING: KALSHI_ENV={config.KALSHI_ENV!r}, this test is demo-only")
        return 2

    print("=" * 74)
    print(f"LIVE EXIT-COMPLETION TEST -- env={config.KALSHI_ENV} "
          f"cap={config.LIVE_MAX_CONTRACTS}")
    print("=" * 74)

    m = pick_market()
    if not m:
        print("no suitable market open right now (need 6-20 min left). Retry soon.")
        return 3
    tkr = m["ticker"]
    print(f"market   {tkr}  {mins_left(m):.1f}min left")
    print(f"quote    yes_bid={m.get('yes_bid_dollars')} yes_ask={m.get('yes_ask_dollars')}")
    print(f"book     {depth(tkr)}")

    start_bal = f(kalshi_auth.get_balance().get("balance_dollars"))
    print(f"balance  ${start_bal:.4f}")

    # --- 1. build a position bigger than the touch ---------------------------
    # Build on whichever side the book can actually fill, priced off the touch.
    b0 = depth(tkr)
    long_yes = (b0.get("yes_ask_size") or 0) >= (b0.get("yes_bid_size") or 0)
    build_side = "bid" if long_yes else "ask"
    print(f"\n--- BUILDING POSITION (IOC {'buy YES' if long_yes else 'sell YES (long NO)'}) "
          + "-" * 22)
    have = 0.0
    for i in range(MAX_BUILD_TRIES):
        if have >= TARGET_SIZE:
            break
        b = depth(tkr)
        px = b.get("yes_ask") if long_yes else b.get("yes_bid")
        if not px or not (0.0 < px < 1.0):
            print(f"  no liquidity on that side right now (position {have:.0f})")
            time.sleep(6)
            continue
        want = int(min(TARGET_SIZE - have, config.LIVE_MAX_CONTRACTS))
        r = ex.place_order(tkr, build_side, want, px,
                           time_in_force="immediate_or_cancel", note="EXIT-TEST build")
        got = f(r.get("fill_count"))
        have += got
        print(f"  {'buy ' if long_yes else 'sell'} {want:2d} @ {px:.4f} -> "
              f"{r['status']:9s} filled {got:.0f} (position {have:.0f}/{TARGET_SIZE})")
        if have < TARGET_SIZE:
            time.sleep(6)

    pos = [p for p in ex.position_summary() if p["ticker"] == tkr]
    if not pos or pos[0]["contracts"] == 0:
        print("\nFAILED TO BUILD A POSITION -- nothing to test. "
              "The demo book gave no fills at all.")
        return 4
    p0 = pos[0]
    avg0 = abs(p0["exposure"]) / abs(p0["contracts"])
    print(f"\nposition {p0['contracts']:+.0f} @ avg {avg0:.4f} "
          f"(exposure ${abs(p0['exposure']):.4f})")
    touch = depth(tkr)
    print(f"book     {touch}")
    if abs(p0["contracts"]) <= 5:
        print("NOTE: position is not clearly larger than the touch; the partial-fill "
              "path may not be exercised hard.")

    # --- 2. force the exit path and drive REAL run_cycle ---------------------
    print("\n--- DRIVING live_trader.run_cycle() " + "-" * 38)
    saved = (config.PAPER_TRADE_EXIT_TARGET, config.EXIT_COMMIT_FLOOR)
    config.PAPER_TRADE_EXIT_TARGET = -9.0        # force the commit in both modes
    if FLOOR_MODE:
        # Keep the REAL floor. The position was built by crossing the spread, so
        # it is underwater from birth -- a correct floor must refuse to sell it
        # at all. ANY fill in this mode is the 2026-08-11 bug reappearing.
        print(f"mode: FLOOR TEST -- commit forced, EXIT_COMMIT_FLOOR left REAL "
              f"({config.EXIT_COMMIT_FLOOR:+.2f}); expecting ZERO exit fills")
    else:
        config.EXIT_COMMIT_FLOOR = -9.0
        print("mode: COMPLETION TEST -- commit and floor both forced "
              "(see module docstring)")

    session = {"utc_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
               "realized_pnl": 0.0, "entries": 0, "exits": 0, "halted_reason": ""}
    trace = []
    try:
        for c in range(CYCLES):
            live = [x for x in ex.position_summary() if x["ticker"] == tkr]
            held = live[0]["contracts"] if live else 0.0
            st = session.get("exiting", {}).get(tkr, {})
            trace.append({"cycle": c, "held": held, "attempts": st.get("attempts"),
                          "gave_up": st.get("gave_up")})
            print(f"\n[cycle {c}] holding {held:+.0f}  "
                  f"exit_state={ {k: st[k] for k in ('attempts', 'gave_up') if k in st} }")
            if held == 0 and c > 0:
                print("  position is FLAT -- exit completed")
                break
            if st.get("gave_up"):
                print(f"  circuit breaker recorded: {st['gave_up']} -- stopping")
                break
            lt.run_cycle(session)
            if c < CYCLES - 1:
                time.sleep(CYCLE_GAP)
    finally:
        config.PAPER_TRADE_EXIT_TARGET, config.EXIT_COMMIT_FLOOR = saved

    # --- 3. reconcile against the exchange -----------------------------------
    print("\n--- RECONCILIATION " + "-" * 55)
    live = [x for x in ex.position_summary() if x["ticker"] == tkr]
    final = live[0]["contracts"] if live else 0.0
    fills = kalshi_auth.get_fills(limit=100, ticker=tkr)
    sells = [x for x in fills if x.get("action") == "sell"]
    buys = [x for x in fills if x.get("action") == "buy"]
    print(f"exchange position now : {final:+.0f}  (started {p0['contracts']:+.0f})")
    print(f"exchange fills        : {len(buys)} buy, {len(sells)} sell")
    for x in sells[:12]:
        print(f"    sell {f(x.get('count_fp')):.0f} @ yes {x.get('yes_price_dollars')} "
              f"fee ${f(x.get('fee_cost')):.4f} taker={x.get('is_taker')}")
    exit_orders = [r for r in ex.load_orders()
                   if r["ticker"] == tkr and "exit" in r.get("note", "")]
    print(f"bot exit orders       : {len(exit_orders)}")
    partials = [r for r in exit_orders if r["status"] == "partial"]
    zeros = [r for r in exit_orders if f(r.get("fill_count")) == 0]
    print(f"    partial fills     : {len(partials)}   zero fills: {len(zeros)}")
    end_bal = f(kalshi_auth.get_balance().get("balance_dollars"))
    print(f"balance               : ${start_bal:.4f} -> ${end_bal:.4f} "
          f"({end_bal - start_bal:+.4f}, test cost -- it crosses a wide spread on purpose)")

    print("\ncycle trace:")
    for t in trace:
        print(f"    cycle {t['cycle']}: held {t['held']:+.0f} "
              f"attempts={t['attempts']} gave_up={t['gave_up']}")

    print("\n" + "=" * 74)
    ok_flat = final == 0
    ok_multi = len(exit_orders) > 1
    st = session.get("exiting", {}).get(tkr, {})
    ok_breaker = bool(st.get("gave_up"))
    if FLOOR_MODE:
        sold = sum(f(r.get("fill_count")) for r in exit_orders)
        floor_ok = sold == 0 and final == p0["contracts"]
        gv = session.get("exiting", {}).get(tkr, {}).get("gave_up")
        print(f"[{'PASS' if floor_ok else 'FAIL'}] floor refused to sell an "
              f"underwater position ({sold:.0f} contract(s) sold, position "
              f"{p0['contracts']:+.0f} -> {final:+.0f})")
        print(f"[{'PASS' if gv else 'FAIL'}] recorded why it refused ({gv})")
        print("RESULT:", "PASS" if (floor_ok and gv) else "FAIL")
        return 0 if (floor_ok and gv) else 1
    print(f"[{'PASS' if ok_multi else 'FAIL'}] exit persisted across more than one order "
          f"({len(exit_orders)} exit orders)")
    print(f"[{'PASS' if ok_flat or ok_breaker else 'FAIL'}] ended flat, or stopped for a "
          f"recorded reason (flat={ok_flat}, gave_up={st.get('gave_up')})")
    print(f"[{'PASS' if len(partials) or len(zeros) else 'INFO'}] real partial/zero fills "
          f"were encountered and handled ({len(partials)} partial, {len(zeros)} zero)")
    verdict = ok_multi and (ok_flat or ok_breaker)
    print("RESULT:", "PASS" if verdict else "FAIL")
    if final != 0 and not ok_breaker:
        print(f"WARNING: {final:+.0f} contracts still open with no breaker recorded.")
    return 0 if verdict else 1


if __name__ == "__main__":
    try:
        with SingleInstance("live_trader"):
            sys.exit(main())
    except AlreadyRunning as e:
        print(f"live_trader is running ({e}). Stop it before running this test -- "
              "two processes trading the same account is the bug singleton.py exists "
              "to prevent.")
        sys.exit(5)
