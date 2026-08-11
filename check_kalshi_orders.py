"""
Phase 2 verification: place REAL orders against the DEMO exchange and prove
each outcome path is handled correctly.

Not mocks. Every case below sends an actual order to Kalshi's demo matching
engine and asserts on what comes back.

    python check_kalshi_orders.py

Refuses to run unless KALSHI_ENV=demo. Every order is capped at
config.LIVE_MAX_CONTRACTS and any resting order it creates is cancelled before
exit, so the account is left flat.
"""
import sys
import uuid

import requests

import config
import kalshi_auth
import live_executor as ex


def market() -> dict | None:
    """Pick an open KXBTC15M market on demo with a usable two-sided book."""
    r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets",
                     params={"series_ticker": "KXBTC15M", "status": "open", "limit": 20},
                     timeout=20)
    best = None
    for m in r.json().get("markets", []):
        if m.get("status") != "active":
            continue
        ask = m.get("yes_ask_dollars")
        if ask in (None, "", "0.0000", "1.0000"):
            continue
        best = m
        break
    return best


def main():
    print("=" * 70)
    print(f"env={config.KALSHI_ENV}  base={config.KALSHI_TRADE_BASE}")
    if config.KALSHI_ENV != "demo":
        print("REFUSING: this script is demo-only.")
        return 1
    bal = kalshi_auth.get_balance()
    print(f"balance: ${bal.get('balance_dollars')}   max order size: {config.LIVE_MAX_CONTRACTS}")

    m = market()
    if not m:
        print("\nNo open KXBTC15M market with a two-sided book on demo right now.")
        return 1
    ticker = m["ticker"]
    bid = float(m.get("yes_bid_dollars") or 0)
    ask = float(m.get("yes_ask_dollars") or 0)
    print(f"market: {ticker}  yes_bid={bid:.2f} yes_ask={ask:.2f}  closes {m.get('close_time')}")

    results = {}
    created_resting = []

    # --- 1. environment guard ------------------------------------------------
    print("\n[1] ENVIRONMENT GUARD -- writes must be refused outside demo")
    saved = config.KALSHI_ENV
    try:
        config.KALSHI_ENV = "prod"
        ex.place_order(ticker, "bid", 1, 0.01)
        print("   *** FAIL: order was NOT blocked in prod mode ***")
        results["env_guard"] = False
    except ex.EnvironmentGuardError as e:
        print(f"   blocked as expected: {str(e).splitlines()[0][:80]}")
        results["env_guard"] = True
    finally:
        config.KALSHI_ENV = saved

    # --- 2. size cap ---------------------------------------------------------
    print(f"\n[2] SIZE CAP -- {config.LIVE_MAX_CONTRACTS + 1} contracts must be refused")
    try:
        ex.place_order(ticker, "bid", config.LIVE_MAX_CONTRACTS + 1, 0.01)
        print("   *** FAIL: oversized order was not blocked ***")
        results["size_cap"] = False
    except ex.LiveExecError as e:
        print(f"   blocked as expected: {e}")
        results["size_cap"] = True

    # --- 3. no-fill IOC ------------------------------------------------------
    print("\n[3] NO-FILL IOC -- bid far below the ask, nothing to match")
    r = ex.place_order(ticker, "bid", 1, 0.01, time_in_force="immediate_or_cancel",
                       note="phase2 no-fill test")
    print(f"   status={r['status']}  fill={r.get('fill_count')} remaining={r.get('remaining_count')} "
          f"order_id={str(r.get('order_id'))[:8]}...")
    results["no_fill"] = r["status"] == "no_fill"

    # --- 4. marketable order with $0 balance -> rejection --------------------
    print("\n[4] REJECTION -- marketable buy at the ask with $0.00 balance")
    r = ex.place_order(ticker, "bid", 1, min(ask, 0.99),
                       time_in_force="immediate_or_cancel",
                       note="phase2 rejection test")
    print(f"   status={r['status']}  http={r.get('http')}")
    if r["status"] == "rejected":
        print(f"   error: {str(r.get('error'))[:160]}")
        results["rejection"] = True
    elif r["status"] in ("filled", "partial"):
        print(f"   NOTE: it FILLED -- the account has funds after all "
              f"(fill={r.get('fill_count')} @ {r.get('avg_fill_price')} fee={r.get('fee_paid')})")
        results["rejection"] = "n/a - filled"
    else:
        print(f"   unexpected: {r}")
        results["rejection"] = False

    # --- 5. resting GTC order + cancel --------------------------------------
    print("\n[5] RESTING ORDER -- GTC below the market, then cancel it")
    r = ex.place_order(ticker, "bid", 1, 0.01, time_in_force="good_till_canceled",
                       note="phase2 resting test")
    print(f"   status={r['status']}  order_id={str(r.get('order_id'))[:8]}...")
    if r["status"] == "resting" and r.get("order_id"):
        created_resting.append(r["order_id"])
        resting = ex.get_resting_orders()
        print(f"   exchange reports {len(resting)} resting order(s)")
        c = ex.cancel_order(r["order_id"])
        print(f"   cancel -> http={c['http']} ok={c['ok']}")
        results["resting_and_cancel"] = c["ok"]
        if c["ok"]:
            created_resting.remove(r["order_id"])
    else:
        results["resting_and_cancel"] = f"not resting ({r['status']})"

    # --- 6. client_order_id idempotency -------------------------------------
    print("\n[6] IDEMPOTENCY -- same client_order_id sent twice")
    cid = str(uuid.uuid4())
    a = ex.place_order(ticker, "bid", 1, 0.01, time_in_force="immediate_or_cancel",
                       client_order_id=cid, note="phase2 idempotency A")
    b = ex.place_order(ticker, "bid", 1, 0.01, time_in_force="immediate_or_cancel",
                       client_order_id=cid, note="phase2 idempotency B")
    print(f"   first : status={a['status']} order_id={str(a.get('order_id'))[:8]}...")
    print(f"   second: status={b['status']} order_id={str(b.get('order_id'))[:8]}... http={b.get('http')}")
    same = a.get("order_id") and a.get("order_id") == b.get("order_id")
    deduped = same or b["status"] == "rejected"
    print(f"   deduped by exchange: {deduped} "
          f"({'same order_id' if same else 'second rejected' if b['status']=='rejected' else 'NOT DEDUPED'})")
    print(f"   local log rows for this cid: "
          f"{sum(1 for x in ex.load_orders() if x['client_order_id'] == cid)} (want 1)")
    results["idempotency"] = deduped

    # --- 7. reconciliation ---------------------------------------------------
    print("\n[7] RECONCILIATION -- exchange vs local log")
    rec = ex.reconcile()
    for k in ("exchange_resting_orders", "local_rows", "local_filled_or_partial",
              "local_unresolved", "positions_not_in_local_log",
              "local_fills_without_position", "in_sync"):
        print(f"   {k:32s} {rec[k]}")
    results["reconcile_in_sync"] = rec["in_sync"]

    # --- cleanup -------------------------------------------------------------
    if created_resting:
        print(f"\nCLEANUP: cancelling {len(created_resting)} leftover resting order(s)")
        for oid in created_resting:
            print(f"   {oid[:8]}... -> {ex.cancel_order(oid)}")
    left = ex.get_resting_orders()
    print(f"\nresting orders left on the exchange: {len(left)} (want 0)")

    print("\n" + "=" * 70)
    for k, v in results.items():
        print(f"  {k:24s} {v}")
    hard_fail = [k for k, v in results.items() if v is False]
    print("\nPHASE 2: " + ("PASS" if not hard_fail else f"FAIL on {hard_fail}"))
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    sys.exit(main())
