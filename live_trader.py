"""
Phase 3: autonomous entry + exit on the DEMO exchange, with safety rails.

Wires the ALREADY-VALIDATED decision logic to the Phase 2 execution primitives.
It does not reimplement any of it:

    entry decision : paper_trader.evaluate_trade()   (same call bot.py + the
                     dashboard use -- single source of truth, and the reason
                     the dashboard/Telegram drift bug cannot recur here)
    exit target    : config.PAPER_TRADE_EXIT_TARGET  (the deployed +35%)
    execution      : live_executor.place_order()     (Phase 2, demo-guarded)

=== SEPARATION FROM SIMULATED TRADING ===
    paper_trader.py -> data/trade_log.csv       paper_trader.log    simulated
    live_trader.py  -> data/live_orders.csv     live_trader.log     REAL orders
                       data/live_session.json
No shared state, no shared log, no shared vocabulary.

=== SAFETY RAILS ===
1. DEMO ONLY. live_executor._require_demo() raises on every write path when
   KALSHI_ENV != "demo". This module additionally refuses to start.
2. SIZE CAP. config.LIVE_MAX_CONTRACTS (4), enforced inside place_order().
3. DAILY LOSS CAP. config.LIVE_DAILY_LOSS_CAP. On breach: NO NEW ENTRIES for
   the rest of the UTC day. Existing positions still ride and still exit --
   halting exits on a loss cap would strand capital in the exact scenario the
   cap exists to protect against.
4. KILL SWITCH. A file (config.LIVE_KILL_SWITCH_FILE). Checked EVERY cycle,
   before anything else. Its whole point is to work when the process is wedged
   or unreachable, so it is a filesystem check with no dependency on this
   program's own state, network, or a clean shutdown. Halts new entries; lets
   open positions exit.
5. RECONCILE EVERY CYCLE. reconcile() + refresh_order_status() run on every
   pass, not just at startup. Phase 2b proved why: a resting sell sat at
   fill=0 for 24s and then filled 1 while the local log still said "resting".
   The exchange is the truth; local state is a cache that can be stale.

Usage:
    python live_trader.py --once      one cycle, then exit
    python live_trader.py             continuous loop
    python live_trader.py --status    print state and exit, touch nothing
"""
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
import kalshi_auth
from singleton import AlreadyRunning, SingleInstance
import live_executor as ex
from coinbase_feed import fetch_1min_candles, fetch_spot_price
from fees import kalshi_fee
from indicators import compute_signals
from kalshi_feed import get_active_market
from paper_trader import evaluate_trade

SESSION_FILE = Path(__file__).parent / "data" / "live_session.json"
LOG_FILE = Path(__file__).parent / "live_trader.log"


def _setup_console_logging():
    """Same pattern as bot.py/paper_trader.py, and for the same reason: called
    from __main__ only, never at import. See the 2026-08-06 incident."""
    try:
        if sys.stdout is None or sys.stderr is None:
            log = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = log
            return
        enc = getattr(sys.stdout, "encoding", None)
        rec = getattr(sys.stdout, "reconfigure", None)
        if enc and enc.lower() != "utf-8" and callable(rec):
            rec(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg):
    print(f"[{_stamp()}] {msg}", flush=True)


# --- session state (daily loss cap) -----------------------------------------

def load_session() -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if SESSION_FILE.exists():
        try:
            s = json.loads(SESSION_FILE.read_text())
            if s.get("utc_date") == today:
                return s
        except Exception:
            pass
    return {"utc_date": today, "realized_pnl": 0.0, "entries": 0, "exits": 0,
            "halted_reason": ""}


def save_session(s: dict):
    SESSION_FILE.parent.mkdir(exist_ok=True)
    SESSION_FILE.write_text(json.dumps(s, indent=1))


def kill_switch_active() -> bool:
    return Path(config.LIVE_KILL_SWITCH_FILE).exists()


def entries_allowed(session: dict) -> tuple[bool, str]:
    if kill_switch_active():
        return False, f"kill switch present ({config.LIVE_KILL_SWITCH_FILE})"
    if session["realized_pnl"] <= -abs(config.LIVE_DAILY_LOSS_CAP):
        return False, (f"daily loss cap hit: realized {session['realized_pnl']:+.2f} "
                       f"<= -{abs(config.LIVE_DAILY_LOSS_CAP):.2f}")
    return True, ""


# --- market helpers ----------------------------------------------------------

def market_quote(ticker: str) -> dict:
    r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets/{ticker}", timeout=15)
    return r.json().get("market", {}) or {}


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --- the cycle ---------------------------------------------------------------

def run_cycle(session: dict) -> dict:
    """One pass: reconcile, then exits, then (maybe) an entry."""
    # 1. RECONCILE FIRST, ALWAYS. Never act on stale local state.
    for row in ex.load_orders():
        if row.get("status") in ("resting", "sending", "unknown") and row.get("order_id"):
            upd = ex.refresh_order_status(row["client_order_id"])
            if upd and upd["status"] != row["status"]:
                log(f"order {row['client_order_id'][:8]} {row['status']} -> {upd['status']} "
                    f"(fill={upd['fill_count']})")
    # Cancel resting ENTRY orders whose window is no longer tradeable. A GTC
    # entry that never filled must not linger into the settlement print, or into
    # a later window, where it would fill on terms the model never approved.
    for o in ex.get_resting_orders():
        row = next((r for r in ex.load_orders()
                    if r.get("order_id") == o.get("order_id")), None)
        if not row or "entry" not in row.get("note", ""):
            continue
        q = market_quote(row["ticker"])
        stale = q.get("status") != "active"
        if not stale:
            try:
                closes = datetime.fromisoformat(q["close_time"].replace("Z", "+00:00"))
                stale = (closes - datetime.now(timezone.utc)).total_seconds() / 60.0 \
                    <= config.FINAL_MINUTES_NOISY
            except Exception:
                stale = False
        if stale:
            c = ex.cancel_order(o["order_id"])
            log(f"cancelled stale resting entry {o['order_id'][:8]} on {row['ticker']} "
                f"(window no longer tradeable) ok={c['ok']}")

    rec = ex.reconcile()
    if not rec["in_sync"]:
        log(f"DRIFT: positions_not_in_log={rec['positions_not_in_local_log']} "
            f"fills_without_position={rec['local_fills_without_position']} "
            f"unresolved={rec['local_unresolved']}")

    positions = ex.position_summary()

    # 2. EXITS -- these run even when entries are halted.
    for p in positions:
        if p["contracts"] == 0:
            continue
        q = market_quote(p["ticker"])
        if q.get("status") != "active":
            continue
        contracts = p["contracts"]
        # Average cost per contract, from the EXCHANGE's numbers, not local state.
        avg_cost = abs(p["exposure"]) / abs(contracts) if contracts else 0.0
        if avg_cost <= 0:
            continue
        if contracts > 0:                       # long YES -> sell into the yes bid
            exit_px = _f(q.get("yes_bid_dollars"))
            side = "ask"
        else:                                   # short YES (long NO) -> buy back at the ask
            exit_px = 1.0 - _f(q.get("yes_ask_dollars"))
            side = "bid"
        if exit_px <= 0:
            continue
        gain = (exit_px - avg_cost) / avg_cost
        log(f"position {p['ticker']} {contracts:+.0f} @ avg {avg_cost:.4f} "
            f"exit_px={exit_px:.4f} gain={gain*100:+.1f}% "
            f"(target +{config.PAPER_TRADE_EXIT_TARGET*100:.0f}%)")
        if gain >= config.PAPER_TRADE_EXIT_TARGET:
            n = int(min(abs(contracts), config.LIVE_MAX_CONTRACTS))
            log(f"EXIT TARGET HIT -> selling {n} of {p['ticker']} at {exit_px:.4f}")
            r = ex.place_order(p["ticker"], side, n, exit_px,
                               time_in_force="immediate_or_cancel",
                               note=f"phase3 exit +{gain*100:.0f}%")
            log(f"  exit order: status={r['status']} fill={r.get('fill_count')} "
                f"err={str(r.get('error'))[:100]}")
            if r["status"] in ("filled", "partial"):
                filled = float(r.get("fill_count") or 0)
                pnl = (exit_px - avg_cost) * filled
                session["realized_pnl"] += pnl
                session["exits"] += 1
                log(f"  realized {pnl:+.4f}  session P&L {session['realized_pnl']:+.4f}")

    # 3. ENTRY -- gated by kill switch and loss cap.
    allowed, why = entries_allowed(session)
    if not allowed:
        if session.get("halted_reason") != why:
            log(f"ENTRIES HALTED: {why}")
            session["halted_reason"] = why
        return session
    session["halted_reason"] = ""

    market = get_active_market()
    if market is None or market.strike is None:
        return session
    held = {p["ticker"] for p in positions if p["contracts"] != 0}
    if market.ticker in held:
        return session
    if any(r["ticker"] == market.ticker and r["status"] in ("filled", "partial")
           for r in ex.load_orders()):
        return session                      # already traded this window

    candles = fetch_1min_candles()
    spot = fetch_spot_price(candles)
    vol = compute_signals(candles).realized_vol_pct
    d = evaluate_trade(spot.price, market, realized_vol_pct=vol)
    pyes = f" p_yes={d.p_yes*100:.1f}%" if d.p_yes is not None else ""
    log(f"{market.ticker} {d.action}/{d.reason}{pyes}")
    if d.action != "enter":
        return session

    n = min(config.LIVE_MAX_CONTRACTS, config.PAPER_TRADE_SIZE)

    # RE-READ THE BOOK IMMEDIATELY BEFORE SENDING.
    # evaluate_trade() prices off a market snapshot that is already seconds old
    # by the time we get here. Submitting an IOC limit at that stale price means
    # a book that ticked up in the interim never crosses -- observed live on
    # 2026-08-11: signals at 0.66 and 0.72 against an ask that had moved to 0.75,
    # both no_fill. Correct signals, zero fills, which in production would mean
    # the backtested returns are simply never realised.
    #
    # The fix is a fresh quote, NOT a price buffer: paying above the modelled
    # entry price would silently erode the very edge the gate just checked. So
    # re-price at the current ask and re-validate that the edge still clears; if
    # the market has moved past what the model called profitable, skip the trade
    # rather than chase it.
    q = market_quote(market.ticker)
    if d.side == "yes":
        fresh = _f(q.get("yes_ask_dollars"))
        side = "bid"
    else:
        fresh = 1.0 - _f(q.get("yes_bid_dollars"))
        side = "ask"
    if not (0.0 < fresh < 1.0):
        log(f"  no usable fresh quote for {market.ticker} ({fresh}); skipping")
        return session
    slip = fresh - d.entry_price
    fee_per = kalshi_fee(n, fresh) / n
    edge_now = (d.p_model - fresh - fee_per) if d.p_model is not None else None
    if edge_now is None or edge_now < config.PAPER_TRADE_MIN_EDGE:
        log(f"  {market.ticker} moved {slip*100:+.1f}c since evaluation "
            f"({d.entry_price:.4f} -> {fresh:.4f}); edge now "
            f"{(edge_now or 0)*100:+.1f}c < {config.PAPER_TRADE_MIN_EDGE*100:.0f}c -- SKIP, not chasing")
        return session
    px = round(fresh, 4) if d.side == "yes" else round(1.0 - fresh, 4)
    log(f"ENTRY SIGNAL {market.ticker} {d.side.upper()} edge={edge_now*100:+.1f}c "
        f"(was {d.edge*100:+.1f}c, moved {slip*100:+.1f}c) "
        f"-> {n} contracts, side={side} price={px:.4f}")
    # RESTING (good_till_canceled), not IOC.
    #
    # Five consecutive IOC entries failed to fill. The cause was NOT the order
    # construction -- a controlled test confirmed side="ask" with a YES-terms
    # price does open a NO position (0.00 -> -1.00, fill action=sell
    # outcome=no). The cause is that this book flickers between one- and
    # two-sided within seconds, so an IOC priced at the touch loses the race
    # between quote-read and arrival. A resting order does not race: it sits at
    # our price until the market comes to it.
    #
    # It is also strictly better economically. Phase 2b proved maker fills on
    # this series are charged EXACTLY $0.00 (fee_type="quadratic", not
    # "quadratic_with_maker_fees"), where a taker fill at these prices costs
    # 1-4% of stake. The tradeoff is fill uncertainty, which is handled: the
    # per-cycle refresh_order_status() picks up late fills, and stale entry
    # orders are cancelled below once their window stops being tradeable.
    r = ex.place_order(market.ticker, side, n, px,
                       time_in_force="good_till_canceled",
                       note=f"phase3 entry {d.side} edge={d.edge:.4f}")
    log(f"  entry order: status={r['status']} fill={r.get('fill_count')} "
        f"avg={r.get('avg_fill_price')} fee={r.get('fee_paid')} "
        f"err={str(r.get('error'))[:100]}")
    if r["status"] in ("filled", "partial"):
        session["entries"] += 1
        filled = float(r.get("fill_count") or 0)
        fee = _f(r.get("fee_paid")) * filled
        session["realized_pnl"] -= fee      # fees are realized immediately
    return session


def print_status():
    s = load_session()
    allowed, why = entries_allowed(s)
    print(f"env                 {config.KALSHI_ENV}")
    print(f"kill switch         {'ACTIVE -- ' + str(config.LIVE_KILL_SWITCH_FILE) if kill_switch_active() else 'clear'}")
    print(f"daily loss cap      ${config.LIVE_DAILY_LOSS_CAP:.2f}")
    print(f"session date        {s['utc_date']}")
    print(f"session realized    {s['realized_pnl']:+.4f}")
    print(f"entries / exits     {s['entries']} / {s['exits']}")
    print(f"entries allowed     {allowed}{'' if allowed else '  (' + why + ')'}")
    print(f"balance             ${kalshi_auth.get_balance().get('balance_dollars')}")
    for p in ex.position_summary():
        print(f"position            {p['ticker']} {p['contracts']:+.0f} "
              f"exposure=${p['exposure']:.4f} fees=${p['fees_paid']:.4f}")
    r = ex.reconcile()
    print(f"in_sync             {r['in_sync']}")


def main():
    if config.KALSHI_ENV != "demo":
        log(f"REFUSING TO START: KALSHI_ENV={config.KALSHI_ENV!r}, Phase 3 is demo-only.")
        return 1
    if "--status" in sys.argv:
        print_status()
        return 0

    log(f"live_trader starting -- env={config.KALSHI_ENV} "
        f"size_cap={config.LIVE_MAX_CONTRACTS} loss_cap=${config.LIVE_DAILY_LOSS_CAP} "
        f"exit_target=+{config.PAPER_TRADE_EXIT_TARGET*100:.0f}%")
    session = load_session()
    once = "--once" in sys.argv
    cycles = 0
    while True:
        try:
            session = run_cycle(session)
            save_session(session)
        except Exception:
            log("cycle error:")
            traceback.print_exc()
        cycles += 1
        if cycles % 15 == 0:
            # Heartbeat. A quiet loop and a dead loop look identical in a log
            # otherwise -- exactly the ambiguity that made the 2026-08-03 bot.py
            # incident harder to diagnose than it needed to be.
            allowed, why = entries_allowed(session)
            log(f"heartbeat -- cycle {cycles}, session P&L {session['realized_pnl']:+.4f}, "
                f"entries={session['entries']} exits={session['exits']}, "
                f"entries_allowed={allowed}{(' (' + why + ')') if not allowed else ''}")
        if once:
            return 0
        time.sleep(config.PAPER_TRADE_POLL_SECONDS)


if __name__ == "__main__":
    _setup_console_logging()
    # SINGLE INSTANCE. Two traders each honour LIVE_MAX_CONTRACTS individually
    # while together doubling account exposure, and client_order_id cannot see
    # across processes. This happened for real on 2026-08-11.
    try:
        with SingleInstance("live_trader"):
            sys.exit(main())
    except AlreadyRunning as exc:
        log(f"REFUSING TO START: {exc}")
        sys.exit(2)
