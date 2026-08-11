"""
Phase 2b: exercise the fill paths that $0.00 balance previously blocked.

Real orders against the demo matching engine -- resting GTC, full fill, and a
genuine PARTIAL fill engineered by ordering past the remaining book depth.
Also compares Kalshi's actual charged fee against fees.py, which is the first
time this project has been able to check that prediction against ground truth.

    python check_kalshi_fills.py

Demo-only (live_executor enforces it). Every order is capped at
config.LIVE_MAX_CONTRACTS. Any resting order created here is cancelled before
exit; filled positions are reported, not force-liquidated (there is no bid to
sell into on this book, and demo positions settle on their own).
"""
import sys
import time

import requests

import config
import kalshi_auth
import live_executor as ex
from fees import kalshi_fee


def book(ticker: str) -> dict:
    r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets/{ticker}", timeout=20)
    return r.json().get("market", {})


def pick_market() -> dict | None:
    r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets",
                     params={"series_ticker": "KXBTC15M", "status": "open", "limit": 20},
                     timeout=20)
    for m in r.json().get("markets", []):
        if m.get("status") != "active":
            continue
        try:
            ask = float(m.get("yes_ask_dollars") or 0)
            size = float(m.get("yes_ask_size_fp") or 0)
        except (TypeError, ValueError):
            continue
        if 0 < ask < 1 and size >= 1:
            return m
    return None


def main():
    if config.KALSHI_ENV != "demo":
        print("REFUSING: demo-only.")
        return 1

    bal0 = kalshi_auth.get_balance()
    print("=" * 72)
    print(f"env={config.KALSHI_ENV}  balance=${bal0.get('balance_dollars')}  "
          f"max size={config.LIVE_MAX_CONTRACTS}")

    m = pick_market()
    if not m:
        print("No open KXBTC15M market with offered size on demo right now.")
        return 1
    ticker = m["ticker"]
    ask = float(m["yes_ask_dollars"])
    depth = float(m.get("yes_ask_size_fp") or 0)
    print(f"market {ticker}\n  ask=${ask:.4f}  offered={depth:.2f} contracts  closes {m.get('close_time')}")

    results, resting_ids = {}, []

    # --- 1. RESTING GTC (previously blocked by $0 balance) -------------------
    print("\n[1] RESTING GTC -- bid below the ask so it cannot match, then cancel")
    r = ex.place_order(ticker, "bid", 1, 0.001, time_in_force="good_till_canceled",
                       note="phase2b resting")
    print(f"   status={r['status']}  http={r.get('http')}  order_id={str(r.get('order_id'))[:8]}")
    if r["status"] == "resting":
        resting_ids.append(r["order_id"])
        onexch = ex.get_resting_orders()
        print(f"   exchange confirms {len(onexch)} resting order(s)")
        c = ex.cancel_order(r["order_id"])
        print(f"   cancel -> http={c['http']} ok={c['ok']}")
        after = ex.get_resting_orders()
        print(f"   after cancel: {len(after)} resting")
        results["resting_gtc"] = c["ok"] and len(after) < len(onexch)
        if c["ok"]:
            resting_ids.remove(r["order_id"])
    else:
        print(f"   error: {str(r.get('error'))[:150]}")
        results["resting_gtc"] = False

    # --- 2. FULL FILL --------------------------------------------------------
    b = book(ticker)
    ask = float(b.get("yes_ask_dollars") or ask)
    depth = float(b.get("yes_ask_size_fp") or 0)
    n_full = int(min(config.LIVE_MAX_CONTRACTS, max(depth - 1, 1)))
    print(f"\n[2] FULL FILL -- buy {n_full} at the ask ${ask:.4f} (offered {depth:.2f})")
    r = ex.place_order(ticker, "bid", n_full, ask, time_in_force="immediate_or_cancel",
                       note="phase2b full fill")
    print(f"   status={r['status']}  fill={r.get('fill_count')} remaining={r.get('remaining_count')}")
    print(f"   avg_fill_price={r.get('avg_fill_price')}  fee_paid={r.get('fee_paid')}")
    results["full_fill"] = r["status"] == "filled"
    filled_info = r if r["status"] in ("filled", "partial") else None

    # --- 3. PARTIAL FILL -----------------------------------------------------
    time.sleep(1.5)
    b = book(ticker)
    ask2 = float(b.get("yes_ask_dollars") or 0)
    depth2 = float(b.get("yes_ask_size_fp") or 0)
    print(f"\n[3] PARTIAL FILL -- book now ask=${ask2:.4f} offered={depth2:.2f}")
    if 0 < depth2 < config.LIVE_MAX_CONTRACTS and 0 < ask2 < 1:
        n_over = config.LIVE_MAX_CONTRACTS
        print(f"   ordering {n_over} against {depth2:.2f} offered -> expect partial")
        r = ex.place_order(ticker, "bid", n_over, ask2, time_in_force="immediate_or_cancel",
                           note="phase2b partial fill")
        print(f"   status={r['status']}  fill={r.get('fill_count')} remaining={r.get('remaining_count')}")
        print(f"   avg_fill_price={r.get('avg_fill_price')}  fee_paid={r.get('fee_paid')}")
        results["partial_fill"] = r["status"] == "partial"
        if r["status"] in ("filled", "partial"):
            filled_info = filled_info or r
    else:
        print(f"   cannot engineer a partial: depth={depth2} vs cap={config.LIVE_MAX_CONTRACTS}")
        results["partial_fill"] = f"skipped (depth {depth2})"

    # --- 4. FEE CHECK vs fees.py --------------------------------------------
    print("\n[4] FEE GROUND TRUTH -- Kalshi's charge vs fees.py prediction")
    fills = kalshi_auth.get_fills(limit=20, ticker=ticker)
    print(f"   exchange reports {len(fills)} fill(s) on this ticker")
    ok_fee = None
    for f in fills[:5]:
        cnt = float(f.get("count") or 0)
        px = f.get("yes_price_dollars") or f.get("yes_price")
        try:
            pxf = float(px) if px is not None else None
            if pxf and pxf > 1:      # cents
                pxf /= 100.0
        except (TypeError, ValueError):
            pxf = None
        actual = f.get("fee_paid_dollars") or f.get("fee_paid")
        pred = kalshi_fee(cnt, pxf) if (cnt and pxf) else None
        print(f"   count={cnt} price={pxf} actual_fee={actual} predicted={pred}")
        if pred is not None and actual not in (None, ""):
            try:
                a = float(actual)
                if a > 1:            # cents -> dollars
                    a /= 100.0
                ok_fee = abs(a - pred) <= 0.01
            except (TypeError, ValueError):
                pass
    results["fee_matches_model"] = ok_fee if ok_fee is not None else "no comparable fill"

    # --- 5. RECONCILIATION with a REAL position ------------------------------
    print("\n[5] RECONCILIATION -- with an actual position on the books")
    rec = ex.reconcile()
    for k in ("exchange_resting_orders", "local_rows", "local_filled_or_partial",
              "local_unresolved", "positions_not_in_local_log",
              "local_fills_without_position", "in_sync"):
        print(f"   {k:32s} {rec[k]}")
    pos = kalshi_auth.get_positions()
    for p in pos:
        print(f"   POSITION {p.get('ticker')}  position={p.get('position')}  "
              f"exposure={p.get('market_exposure')}  fees={p.get('fees_paid')}")
    results["reconcile_in_sync"] = rec["in_sync"]

    # --- cleanup -------------------------------------------------------------
    for oid in resting_ids:
        print(f"\nCLEANUP cancel {oid[:8]} -> {ex.cancel_order(oid)}")
    left = ex.get_resting_orders()
    bal1 = kalshi_auth.get_balance()
    print(f"\nresting orders left: {len(left)} (want 0)")
    print(f"balance: ${bal0.get('balance_dollars')} -> ${bal1.get('balance_dollars')}")

    print("\n" + "=" * 72)
    for k, v in results.items():
        print(f"  {k:22s} {v}")
    hard_fail = [k for k, v in results.items() if v is False]
    print("\nPHASE 2b: " + ("PASS" if not hard_fail else f"FAIL on {hard_fail}"))
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    sys.exit(main())
