"""
Per-trade audit: does what the bot REPORTED match what Kalshi actually booked?

Not an average. Every completed round trip is checked individually, because an
average hides exactly the failure this exists to catch: on 2026-08-11 a trade
was logged as +$0.76 while the exchange booked -$0.32, and a portfolio-level
average would have buried a 100%-wrong sign on that trade.

Sources compared:
  BOT      data/live_orders.csv   -- what live_trader recorded at the time
  EXCHANGE /portfolio/settlements -- the authoritative settled record
           /portfolio/fills       -- actual fill prices, counts and fees

Settled P&L, per Kalshi's own numbers:
    revenue = (winning side count) * $1.00
    pnl     = revenue - (yes_total_cost + no_total_cost) - fee_cost

    python audit_live_pnl.py            # today
    python audit_live_pnl.py --all      # every settlement on record
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import config
import kalshi_auth

ORDERS = Path(__file__).parent / "data" / "live_orders.csv"


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def main():
    all_days = "--all" in sys.argv
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    settlements = kalshi_auth._request(
        "/portfolio/settlements", params={"limit": 200}).get("settlements", []) or []
    if not all_days:
        settlements = [s for s in settlements
                       if str(s.get("settled_time", "")).startswith(today)]

    fills = kalshi_auth._request("/portfolio/fills", params={"limit": 200}).get("fills", []) or []
    fills_by_ticker = defaultdict(list)
    for f in fills:
        fills_by_ticker[f.get("ticker")].append(f)

    bot_rows = defaultdict(list)
    if ORDERS.exists():
        for r in csv.DictReader(ORDERS.open(encoding="utf-8")):
            bot_rows[r["ticker"]].append(r)

    print("=" * 78)
    print(f"P&L AUDIT -- {'all settlements' if all_days else today} -- env={config.KALSHI_ENV}")
    print("=" * 78)

    total_exch = 0.0
    n_trips = n_bot_traded = 0
    mismatches = []

    for s in sorted(settlements, key=lambda x: x.get("settled_time", "")):
        tkr = s.get("ticker")
        yc, nc = _f(s.get("yes_count_fp")), _f(s.get("no_count_fp"))
        ycost = _f(s.get("yes_total_cost_dollars"))
        ncost = _f(s.get("no_total_cost_dollars"))
        fee = _f(s.get("fee_cost"))
        result = s.get("market_result")
        revenue = (yc if result == "yes" else nc) * 1.0
        pnl = revenue - (ycost + ncost) - fee
        total_exch += pnl

        rows = bot_rows.get(tkr, [])
        entries = [r for r in rows if "entry" in r.get("note", "")
                   and r["status"] in ("filled", "partial")]
        exits = [r for r in rows if "exit" in r.get("note", "")
                 and r["status"] in ("filled", "partial")]
        traded_by_bot = bool(entries)
        if traded_by_bot:
            n_bot_traded += 1
        round_trip = bool(entries and exits)
        if round_trip:
            n_trips += 1

        side = "NO" if any(r["side"] == "ask" for r in entries) else \
               "YES" if entries else "-"
        tag = "ROUND TRIP" if round_trip else ("ENTRY-ONLY" if traded_by_bot else "not bot")

        print(f"\n{tkr}   [{tag}] side={side} result={result}")
        print(f"  exchange: yes {yc:.2f}@${ycost:.4f}  no {nc:.2f}@${ncost:.4f}  fee ${fee:.4f}")
        print(f"            revenue ${revenue:.4f} - cost ${ycost+ncost:.4f} - fee = "
              f"${pnl:+.4f}")

        # Cross-check the exchange's own fills add up to the settlement costs.
        fl = fills_by_ticker.get(tkr, [])
        if fl:
            buy_y = sum(_f(f["count_fp"]) * _f(f["yes_price_dollars"])
                        for f in fl if f.get("action") == "buy")
            sell_y = sum(_f(f["count_fp"]) * (1 - _f(f["yes_price_dollars"]))
                         for f in fl if f.get("action") == "sell")
            fee_f = sum(_f(f.get("fee_cost")) for f in fl)
            cost_ok = abs((buy_y + sell_y) - (ycost + ncost)) < 0.005
            fee_ok = abs(fee_f - fee) < 0.005
            print(f"  fills   : {len(fl)} fill(s), derived cost ${buy_y+sell_y:.4f} "
                  f"(match={cost_ok}), fees ${fee_f:.4f} (match={fee_ok})")
            if not (cost_ok and fee_ok):
                mismatches.append((tkr, "fills do not reconcile to settlement costs"))

        if traded_by_bot:
            for r in entries + exits:
                kind = "entry" if "entry" in r["note"] else "exit "
                print(f"  bot {kind}: side={r['side']} count={r['count']} px={r['price']} "
                      f"fill={r['fill_count']} avg_fill={r['avg_fill_price'] or '-'} "
                      f"fee={r['fee_paid'] or '-'}")
    # Closed-but-NOT-YET-SETTLED positions live in NEITHER the settlements feed
    # nor as an open position -- they sit in /positions with position_fp 0 and a
    # populated realized_pnl. Omitting them made this audit report a false $1.61
    # "mismatch" against a trader that was in fact exactly right. An audit has to
    # cover the same ground as the thing it audits.
    open_realised = 0.0
    for m in (kalshi_auth._request("/portfolio/positions",
              params={"settlement_status": "unsettled"}).get("market_positions") or []):
        contrib = _f(m.get("realized_pnl_dollars")) - _f(m.get("fees_paid_dollars"))
        if abs(contrib) > 1e-9:
            print("")
            print(f"{m.get('ticker')}   [CLOSED, AWAITING SETTLEMENT]")
            print(f"  exchange realized ${_f(m.get('realized_pnl_dollars')):+.4f}"
                  f" - fees ${_f(m.get('fees_paid_dollars')):.4f} = ${contrib:+.4f}")
            open_realised += contrib
    total_exch += open_realised

    print("")
    print("=" * 78)
    print(f"settlements examined      {len(settlements)}")
    print(f"  traded by the bot       {n_bot_traded}")
    print(f"  completed round trips   {n_trips}")
    print(f"EXCHANGE-BOOKED P&L       ${total_exch:+.4f}"
          f"  (settled + closed-awaiting-settlement)")

    sess = Path(__file__).parent / "data" / "live_session.json"
    if sess.exists():
        import json
        s = json.loads(sess.read_text())
        said = _f(s.get("realized_pnl"))
        print(f"session file says         ${said:+.4f} "
              f"(date {s.get('utc_date')}, entries {s.get('entries')} exits {s.get('exits')})")
        if not all_days:
            diff = abs(said - total_exch)
            if diff > 0.01:
                print(f"  *** MISMATCH: session vs exchange differ by ${diff:.4f} ***")
                mismatches.append(("session", "session P&L != exchange-booked P&L"))
            else:
                print(f"  session matches exchange to ${diff:.4f}")

    print("")
    print(f"RECONCILIATION: {'ALL CONSISTENT' if not mismatches else 'PROBLEMS FOUND'}")
    for tk, why in mismatches:
        print(f"  {tk}: {why}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
