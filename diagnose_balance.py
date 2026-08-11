"""
Pin down how the demo exchange books a SHORT-YES position in `balance`.

WHY THIS EXISTS
On 2026-08-11 the account balance ended ~$25 higher than a forward
reconstruction of the day's cash flows predicted. Two incompatible models fit
the fills equally well:

  A. "sell YES" = BUY NO. Cash falls by count * (1 - price) at open. At
     settlement you own NO contracts; if NO loses you simply get nothing more.
  B. "sell YES" = SHORT YES. Cash RISES by count * price at open, and at
     settlement a losing short pays out count * $1.00.

Both produce the same P&L. They produce very different balances mid-trade, and
under B a naive reading of `balance` looks like free money until settlement
claws it back. Guessing between them is not acceptable when the number is being
used to sanity-check P&L, so this measures it with one contract.

Arithmetic cannot separate them from the fill log alone -- only observing the
balance across an open, and then across settlement, can.

DEMO ONLY, 1 contract, worst case ~$1.

    python diagnose_balance.py open     # open a 1-contract short YES, measure
    python diagnose_balance.py check    # re-measure later / after settlement
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
import kalshi_auth
import live_executor as ex
import live_trader as lt

STATE = Path(__file__).parent / "data" / "balance_diagnostic.json"


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def bal():
    return f(kalshi_auth.get_balance().get("balance_dollars"))


def positions():
    return {p["ticker"]: p for p in ex.position_summary()}


def pick():
    r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets",
                     params={"series_ticker": config.KALSHI_SERIES_TICKER,
                             "status": "open", "limit": 20}, timeout=20).json()
    now = datetime.now(timezone.utc)
    best = None
    for m in r.get("markets", []) or []:
        ct = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
        ml = (ct - now).total_seconds() / 60.0
        if ml < 3.0 or ml > 20.0:
            continue
        b = lt.book_touch(m["ticker"])
        if b.get("yes_bid") and 0.0 < b["yes_bid"] < 1.0:
            if best is None or ml > best[1]:
                best = (m, ml, b)
    return best


def do_open():
    if config.KALSHI_ENV != "demo":
        print(f"REFUSING: KALSHI_ENV={config.KALSHI_ENV!r}")
        return 2
    got = pick()
    if not got:
        print("no suitable market right now; retry in a minute")
        return 3
    m, ml, book = got
    tkr = m["ticker"]
    px = book["yes_bid"]                     # sell YES into the bid

    b0 = bal()
    print(f"market   {tkr}  {ml:.1f}min left   yes_bid={px:.4f}")
    print(f"balance BEFORE           ${b0:.4f}")
    print(f"\nPREDICTIONS for selling 1 YES @ {px:.4f}:")
    print(f"  model A (buy NO)   balance -> ${b0 - (1 - px):.4f}   "
          f"(cash out 1-price = ${1-px:.4f})")
    print(f"  model B (short YES) balance -> ${b0 + px:.4f}   "
          f"(cash in price = ${px:.4f})")

    r = ex.place_order(tkr, "ask", 1, px, time_in_force="immediate_or_cancel",
                       note="BALANCE-DIAG open")
    print(f"\norder: {r['status']} fill={r.get('fill_count')} "
          f"avg={r.get('avg_fill_price')} fee={r.get('fee_paid')}")
    if f(r.get("fill_count")) < 1:
        print("no fill -- nothing measured. Retry.")
        return 4

    time.sleep(3)
    b1 = bal()
    delta = b1 - b0
    p = positions().get(tkr, {})
    print(f"balance AFTER            ${b1:.4f}   delta {delta:+.4f}")
    print(f"position                 contracts={p.get('contracts')} "
          f"exposure={p.get('exposure')} fees={p.get('fees_paid')}")

    fee = f(r.get("fee_paid"))
    a_pred, b_pred = -(1 - px) - fee, px - fee
    verdict = ("A: sell YES == BUY NO (cash out 1-price)"
               if abs(delta - a_pred) < abs(delta - b_pred)
               else "B: sell YES == SHORT YES (cash in price, owes $1 if it loses)")
    print(f"\n  predicted A {a_pred:+.4f} | predicted B {b_pred:+.4f} | actual {delta:+.4f}")
    print(f"VERDICT AT OPEN -> {verdict}")

    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({
        "ticker": tkr, "price": px, "fee": fee,
        "balance_before": b0, "balance_after_open": b1, "delta_open": delta,
        "opened": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"\nstate saved. Run `python diagnose_balance.py check` after "
          f"{tkr} settles (~{ml:.0f}min) to measure the settlement leg.")
    return 0


def do_check():
    if not STATE.exists():
        print("no diagnostic state; run `open` first")
        return 1
    s = json.loads(STATE.read_text())
    tkr = s["ticker"]
    b_now = bal()
    print(f"ticker            {tkr}")
    print(f"balance at open   ${s['balance_before']:.4f}")
    print(f"after opening     ${s['balance_after_open']:.4f}  (delta {s['delta_open']:+.4f})")
    print(f"balance NOW       ${b_now:.4f}  (delta since open "
          f"{b_now - s['balance_after_open']:+.4f})")

    p = positions().get(tkr)
    print(f"position now      {p if p else 'gone (settled)'}")

    settle = [x for x in (kalshi_auth._request(
        "/portfolio/settlements", params={"limit": 200}).get("settlements") or [])
        if x.get("ticker") == tkr]
    if not settle:
        print("not settled yet -- rerun after the window closes")
        return 0
    x = settle[0]
    yc, nc = f(x["yes_count_fp"]), f(x["no_count_fp"])
    cost = f(x["yes_total_cost_dollars"]) + f(x["no_total_cost_dollars"])
    res = x.get("market_result")
    rev = (yc if res == "yes" else nc) * 1.0
    pnl = rev - cost - f(x.get("fee_cost"))
    print(f"\nSETTLEMENT: result={res} yes={yc} no={nc} cost=${cost:.4f} "
          f"fee=${f(x.get('fee_cost')):.4f}")
    print(f"  booked P&L      ${pnl:+.4f}")

    total = b_now - s["balance_before"]
    print(f"\nTOTAL balance change across the whole round trip: {total:+.4f}")
    print(f"settlement-derived P&L for the same trade:          {pnl:+.4f}")
    if abs(total - pnl) < 0.01:
        print("=> BALANCE AND P&L AGREE end-to-end. The mid-trade balance is "
              "simply not spendable cash; only the round trip is meaningful.")
    else:
        print(f"=> STILL DISAGREE by {total - pnl:+.4f} -- something beyond "
              "position accounting is moving the balance.")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "open"
    sys.exit(do_open() if cmd == "open" else do_check())
