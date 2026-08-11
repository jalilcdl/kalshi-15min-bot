"""
Does the demo exchange pay out on a LOSING contract?

The day's balance runs $25.48 above a forward reconstruction of every fill,
netting event and settlement. $25.00 of that is exactly the 25 NO contracts
that LOST on KXBTC15M-26AUG111015-15 -- as if each had been paid $1.00 anyway.

That is not a rounding question. If the demo credits losing contracts, then
demo balance, demo "profit", and any conclusion drawn from a demo run are
wrong in the most dangerous possible direction: losses look like wins.

METHOD: buy ONE contract of the side the market thinks is UNLIKELY (cheap
side), hold it through settlement, and compare the balance change across
settlement against what the settlement record says the contract was worth.

  a losing contract SHOULD move the balance by exactly $0.00 at settlement
  if the balance instead rises by $1.00, the demo is crediting losers

Cost: at most the price of one cheap contract. DEMO ONLY.

    python diagnose_settlement_credit.py
"""
import time
from datetime import datetime, timezone

import requests

import config
import kalshi_auth
import live_executor as ex
import live_trader as lt


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def bal():
    return f(kalshi_auth.get_balance().get("balance_dollars"))


def main():
    if config.KALSHI_ENV != "demo":
        print(f"REFUSING: KALSHI_ENV={config.KALSHI_ENV!r}")
        return 2

    # A market with enough time to fill but not so long we wait forever.
    mkt = None
    for _ in range(40):
        r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets",
                         params={"series_ticker": config.KALSHI_SERIES_TICKER,
                                 "status": "open", "limit": 20}, timeout=20).json()
        now = datetime.now(timezone.utc)
        for m in r.get("markets", []) or []:
            ct = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
            ml = (ct - now).total_seconds() / 60.0
            if not (2.0 < ml < 14.0):
                continue
            b = lt.book_touch(m["ticker"])
            if b.get("yes_bid") and b.get("yes_ask"):
                mkt = (m, ml, b)
                break
        if mkt:
            break
        time.sleep(20)
    if not mkt:
        print("no suitable market")
        return 3

    m, ml, book = mkt
    tkr = m["ticker"]
    # Take the side the market prices as UNLIKELY, so it probably loses.
    yes_ask, yes_bid = book["yes_ask"], book["yes_bid"]
    if yes_ask <= 0.5:
        side, px, label = "bid", yes_ask, "YES"          # buy cheap YES
    else:
        side, px, label = "ask", yes_bid, "NO"           # buy cheap NO
    cost = px if side == "bid" else 1.0 - px

    print(f"market {tkr}  {ml:.1f}min left  book yes {yes_bid:.4f}/{yes_ask:.4f}")
    print(f"buying 1 {label} for ~${cost:.4f} -- the side the market thinks LOSES")

    b0 = bal()
    r = ex.place_order(tkr, side, 1, px, time_in_force="immediate_or_cancel",
                       note="SETTLE-CREDIT diag")
    if f(r.get("fill_count")) < 1:
        print(f"no fill ({r['status']}); rerun")
        return 4
    time.sleep(3)
    b1 = bal()
    print(f"balance {b0:.4f} -> {b1:.4f} after buying (delta {b1-b0:+.4f}, "
          f"fee ${f(r.get('fee_paid')):.4f})")

    print(f"holding to settlement (~{ml:.0f}min)...")
    deadline = time.time() + (ml + 8) * 60
    while time.time() < deadline:
        time.sleep(30)
        st = [x for x in (kalshi_auth._request(
            "/portfolio/settlements", params={"limit": 50}).get("settlements") or [])
            if x.get("ticker") == tkr]
        if st:
            time.sleep(5)
            b2 = bal()
            x = st[0]
            yc, nc = f(x["yes_count_fp"]), f(x["no_count_fp"])
            res = x.get("market_result")
            won = (label == "YES" and res == "yes") or (label == "NO" and res == "no")
            expected = 1.00 if won else 0.00
            actual = b2 - b1
            print(f"\nSETTLED result={res}  our side={label}  -> "
                  f"{'WON' if won else 'LOST'}")
            print(f"  settlement record: yes={yc} no={nc} "
                  f"cost=${f(x['yes_total_cost_dollars'])+f(x['no_total_cost_dollars']):.4f}")
            print(f"  balance across settlement: {b1:.4f} -> {b2:.4f} "
                  f"({actual:+.4f})")
            print(f"  expected for a {'winner' if won else 'LOSER'}: {expected:+.4f}")
            if abs(actual - expected) < 0.02:
                print("VERDICT: settlement credit is CORRECT.")
            elif not won and abs(actual - 1.00) < 0.02:
                print("VERDICT: *** DEMO PAYS OUT ON LOSING CONTRACTS *** "
                      "-- demo balance and any demo P&L are not trustworthy.")
            else:
                print(f"VERDICT: unexplained -- off by {actual - expected:+.4f}")
            return 0
    print("timed out waiting for settlement")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
