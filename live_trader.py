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
2. SIZE CAP. config.LIVE_MAX_CONTRACTS (25), enforced inside place_order().
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

=== EXITS COMPLETE; THEY DO NOT GET RECONSIDERED ===
An exit is a two-part decision -- WHETHER to get out, and then GETTING out --
and only the first is a market call. Once the +35% target is hit the ticker is
recorded in session["exiting"] and the position is worked down every cycle
until it is flat, whatever the price does next. Before this, each cycle re-ran
the whole decision, so a position that partially filled at +54% and then ticked
back to +30% was quietly abandoned half-closed. At size 4 that never showed,
because 4 contracts fit inside the touch; at size 25 against a 5-deep book it
stranded most of the position (2026-08-11: 20, 17 and 17 contracts left open).

Three things bound it, so "persistent" never means "reckless":
  EXIT_GIVE_UP_MINUTES  stop inside the last minutes -- thin book, and the
                        quote is dominated by the settlement print
  EXIT_MAX_ATTEMPTS     stop after N cycles that could not clear the position
  EXIT_COMMIT_FLOOR     stop if the gain falls below break-even; completing a
                        profit-take is the commitment, not liquidating at any
                        price. A tripped breaker is recorded, not forgotten, so
                        the next cycle cannot re-commit and start the churn again.

Exits price off the ORDERBOOK (book_touch), never the market quote, whose bid
lagged the real book by up to 48 cents in measurement -- enough to price a sell
at a bid nobody was showing and get a clean zero fill.

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
    cap = config.effective_loss_cap()
    if session["realized_pnl"] <= -cap:
        return False, (f"daily loss cap hit: realized {session['realized_pnl']:+.2f} "
                       f"<= -{cap:.2f}")
    return True, ""


# --- market helpers ----------------------------------------------------------

def market_quote(ticker: str, retries: int = 3) -> dict:
    """Public market read, retried on transport failure. An unhandled
    ConnectionError here used to abort the entire cycle -- including any exit
    in progress -- see kalshi_auth._request for the same fix."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{config.KALSHI_TRADE_BASE}/markets/{ticker}", timeout=15)
            return r.json().get("market", {}) or {}
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            if attempt == retries - 1:
                log(f"  market_quote({ticker}) failed after {retries} tries: {exc}")
                return {}
            time.sleep(0.4 * (2 ** attempt))
    return {}


def book_touch(ticker: str, retries: int = 3) -> dict:
    """The REAL top of book: {yes_bid, yes_ask, yes_bid_size, yes_ask_size}.

    /markets/{ticker} carries yes_bid_dollars and yes_ask_dollars, and those
    fields LAG. Measured on the demo exchange 2026-08-11, sampling both feeds
    every 5s:

        t=0s   market 0.58/0.64    book 0.58/0.64    agree
        t=10s  market 0.58/0.64    book 0.10/0.67    market bid 48c too high
        t=15s  market 0.58/0.64    book 0.10/0.67    still stale
        t=20s  market 0.10/0.68    book 0.10/0.67    caught up

    A 48-cent stale bid breaks an exit two separate ways. It priced sell orders
    at a bid nobody was showing, so the IOC took nothing and reported a clean
    zero fill -- which is exactly the "+49% -> filled 0/17" line in the incident.
    And it inflated the gain calculation, so the bot believed it was up +49% on a
    position the market had already repriced.

    The orderbook is the book you actually trade against, so exits read it.
    Returns {} on failure; the caller falls back to the (lagging) market quote,
    since a stale price beats no exit attempt at all.
    """
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{config.KALSHI_TRADE_BASE}/markets/{ticker}/orderbook", timeout=15)
            ob = (r.json().get("orderbook_fp") or {})
            # Kalshi quotes both sides as BIDS: yes_dollars are bids to buy YES,
            # no_dollars are bids to buy NO. The yes ask is therefore 1 - best no
            # bid. Levels come sorted worst-to-best, so the touch is the last one.
            yes = ob.get("yes_dollars") or []
            no = ob.get("no_dollars") or []
            out = {}
            if yes:
                out["yes_bid"] = float(yes[-1][0])
                out["yes_bid_size"] = float(yes[-1][1])
            if no:
                out["yes_ask"] = 1.0 - float(no[-1][0])
                out["yes_ask_size"] = float(no[-1][1])
            return out
        except (requests.ConnectionError, requests.Timeout, ValueError,
                IndexError, TypeError) as exc:
            if attempt == retries - 1:
                log(f"  book_touch({ticker}) failed after {retries} tries: {exc}")
                return {}
            time.sleep(0.4 * (2 ** attempt))
    return {}


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

    # SESSION P&L FROM THE EXCHANGE, not from local arithmetic.
    #
    # Local accumulation was wrong once already (claimed +$0.76 on a trade the
    # exchange booked at -$0.32). But reading /portfolio/positions is ALSO wrong:
    # once a market SETTLES the position disappears from that endpoint entirely
    # -- all/settled/unsettled every return nothing -- so the day's losses reset
    # to zero on every settlement and a daily loss cap could never accumulate or
    # fire. Verified: it logged "corrected ... to exchange-booked +0.0000" for a
    # day that had really lost money.
    #
    # Settled P&L lives in /portfolio/settlements instead:
    #     revenue = (winning side count) * $1.00
    #     pnl     = revenue - (yes_total_cost + no_total_cost) - fee_cost
    # That formula reproduces the exchange's own -0.3200 / -0.3893 exactly.
    # Open positions still contribute their realized_pnl from /positions.
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        realised = 0.0
        settle = kalshi_auth._request("/portfolio/settlements", params={"limit": 200})
        for s in settle.get("settlements", []) or []:
            if not str(s.get("settled_time", "")).startswith(today):
                continue
            yc, nc = _f(s.get("yes_count_fp")), _f(s.get("no_count_fp"))
            cost = _f(s.get("yes_total_cost_dollars")) + _f(s.get("no_total_cost_dollars"))
            revenue = (yc if s.get("market_result") == "yes" else nc) * 1.0
            realised += revenue - cost - _f(s.get("fee_cost"))
        for m in (kalshi_auth._request("/portfolio/positions",
                  params={"settlement_status": "unsettled"}).get("market_positions") or []):
            realised += _f(m.get("realized_pnl_dollars")) - _f(m.get("fees_paid_dollars"))
        if abs(realised - session["realized_pnl"]) > 0.005:
            log(f"session P&L {session['realized_pnl']:+.4f} -> exchange-booked {realised:+.4f}")
        session["realized_pnl"] = realised
    except Exception as exc:
        log(f"could not read exchange P&L ({exc}); loss cap running on the last "
            f"known value {session['realized_pnl']:+.4f} this cycle")

    rec = ex.reconcile()
    if not rec["in_sync"]:
        log(f"DRIFT: positions_not_in_log={rec['positions_not_in_local_log']} "
            f"fills_without_position={rec['local_fills_without_position']} "
            f"unresolved={rec['local_unresolved']}")

    positions = ex.position_summary()

    # 2. EXITS -- run even when entries are halted (kill switch / loss cap).
    #
    # PERSISTENT EXIT COMPLETION. The original logic re-decided from scratch each
    # cycle: if gain was >= target it fired one IOC, and if the price ticked back
    # below target next cycle it simply stopped. That was invisible while orders
    # fit inside book depth (size 4), and broke badly at size 25 against a demo
    # book ~3-5 deep. Observed 2026-08-11:
    #     +54.3% -> ordered 25, filled  5   (20 left open)
    #     +44.4% -> ordered 20, filled  3   (17 left open)
    #     -22.2% -> no exit attempted at all
    #     +49.2% -> ordered 17, filled  0   (17 left, rode to settlement)
    # A decision to take profit was abandoned mid-execution because the market
    # moved -- the one moment it must NOT be abandoned.
    #
    # Now: once a position starts exiting it is recorded in session["exiting"]
    # and worked down EVERY cycle until flat, regardless of current gain.
    exiting = session.setdefault("exiting", {})

    for p in positions:
        tkr = p["ticker"]
        if p["contracts"] == 0:
            if tkr in exiting:
                log(f"EXIT COMPLETE {tkr} -- flat after "
                    f"{exiting[tkr].get('attempts', 0)} attempt(s)")
                exiting.pop(tkr, None)
                session["exits"] = session.get("exits", 0) + 1
            continue

        q = market_quote(tkr)
        if not q:
            # Empty means the READ failed, not that the market closed. Dropping a
            # live exit commitment because the network blinked is precisely the
            # abandonment this fix exists to prevent, so hold the state and try
            # again next cycle.
            log(f"  {tkr}: quote unavailable this cycle; exit state held")
            continue
        if q.get("status") != "active":
            if tkr in exiting:
                log(f"EXIT ABANDONED {tkr} -- market no longer active; "
                    f"{abs(p['contracts']):.0f} contract(s) ride to settlement")
                exiting.pop(tkr, None)
            continue

        contracts = p["contracts"]
        avg_cost = abs(p["exposure"]) / abs(contracts) if contracts else 0.0
        if avg_cost <= 0:
            continue

        # Price off the ORDERBOOK, not the market quote -- see book_touch().
        # Two units lessons live in these four lines, both paid for on
        # 2026-08-11: gain is measured in the units of the position held, while
        # the ORDER PRICE is always in YES terms.
        book = book_touch(tkr)
        if contracts > 0:                       # long YES: sell into the yes bid
            px = book.get("yes_bid", _f(q.get("yes_bid_dollars")))
            value_px, order_px, side = px, px, "ask"
        else:                                   # short YES (long NO): buy YES back
            px = book.get("yes_ask", _f(q.get("yes_ask_dollars")))
            value_px, order_px, side = 1.0 - px, px, "bid"
        if value_px <= 0 or not (0.0 < order_px < 1.0):
            continue
        gain = (value_px - avg_cost) / avg_cost

        mins_left = 99.0
        try:
            closes = datetime.fromisoformat(q["close_time"].replace("Z", "+00:00"))
            mins_left = (closes - datetime.now(timezone.utc)).total_seconds() / 60.0
        except Exception:
            pass

        state = exiting.get(tkr)

        # CIRCUIT BREAKERS.
        # A tripped breaker is RECORDED, not deleted. Deleting it would clear the
        # commitment, and the next cycle would see gain >= target with no state
        # and commit all over again -- reintroducing the abandon/re-decide churn
        # this whole fix exists to remove. The record dies with the position.
        def _give_up(why):
            log(f"EXIT GIVE-UP {tkr} ({why}): {abs(contracts):.0f} contract(s) "
                f"left, {mins_left:.1f}min to close -- letting it ride to settlement")
            exiting.setdefault(tkr, {"started": _stamp(), "attempts": 0})["gave_up"] = why

        if state and state.get("gave_up"):
            continue

        # (a) Too close to settlement. Applies to a fresh target hit as well as a
        # remainder in progress: inside the last minutes the book thins and the
        # print dominates the quote, which is the same reason entries stop at
        # FINAL_MINUTES_NOISY. Not exiting costs variance, not expected value --
        # at price p a contract is worth ~p either way, so paying a taker fee and
        # a widened spread to get out of a position that settles in 90 seconds
        # buys nothing.
        if mins_left <= config.EXIT_GIVE_UP_MINUTES:
            if state:
                _give_up("window closing")
            continue

        if state:
            if state.get("attempts", 0) >= config.EXIT_MAX_ATTEMPTS:
                _give_up("max attempts")
                continue
            # (b) The commitment is to COMPLETE a profit-take, not to liquidate at
            # any price. Staying committed through a dip from +54% to +12% is the
            # entire point of this fix. Selling BELOW cost is a different act: a
            # stop-loss, which this strategy never validated and which expected
            # value does not favour -- the quote is roughly the settlement
            # probability, so holding pays about the same as selling, minus the
            # fee and spread that selling actually costs.
            if gain < config.EXIT_COMMIT_FLOOR:
                _give_up(f"gain {gain*100:+.0f}% below the "
                         f"{config.EXIT_COMMIT_FLOOR*100:+.0f}% floor")
                continue

        if state is None:
            if gain < config.PAPER_TRADE_EXIT_TARGET:
                log(f"position {tkr} {contracts:+.0f} @ avg {avg_cost:.4f} "
                    f"value_px={value_px:.4f} gain={gain*100:+.1f}% "
                    f"(target +{config.PAPER_TRADE_EXIT_TARGET*100:.0f}%)")
                continue
            state = exiting[tkr] = {"started": _stamp(), "attempts": 0,
                                    "gain_at_decision": gain,
                                    "contracts_at_decision": abs(contracts),
                                    # the position's value AT the moment we
                                    # decided -- the reference every later sweep
                                    # measures adverse movement against
                                    "trigger_value": value_px}
            log(f"EXIT TARGET HIT {tkr}: gain {gain*100:+.1f}% on "
                f"{abs(contracts):.0f} contract(s) -- COMMITTING to exit, will work "
                f"the remainder down until flat")

        # Committed. Sweep within the cycle: this book replenishes in seconds, so
        # several small IOCs beat one large one that takes only what is at the
        # touch right now and cancels the rest.
        state["attempts"] = state.get("attempts", 0) + 1
        remaining = abs(contracts)
        got_total = 0.0
        # Reference for the within-cycle adverse-move guard below: what the
        # position was worth when this cycle's sweeps began.
        cycle_ref_value = value_px
        for sweep in range(config.EXIT_SWEEPS_PER_CYCLE):
            if remaining <= 0:
                break
            if sweep:                       # re-read the touch between sweeps
                b2 = book_touch(tkr)
                q2 = market_quote(tkr)
                px = (b2.get("yes_bid", _f(q2.get("yes_bid_dollars"))) if contracts > 0
                      else b2.get("yes_ask", _f(q2.get("yes_ask_dollars"))))
            else:
                px = order_px
            if not (0.0 < px < 1.0):
                break

            # RE-CHECK THE GUARDS ON EVERY SWEEP, AGAINST THE FRESH PRICE.
            #
            # Checking them once per cycle was a real, costly hole. On
            # 2026-08-11 a committed exit swept three times in six seconds; the
            # ask went 0.47 -> 0.90 between sweep 1 and sweep 2, and the bot
            # bought back at 0.90 a position that cost 0.44 -- realising -$1.70
            # on 5 contracts. The floor was supposed to prevent exactly that and
            # never got consulted, because it had been evaluated once at the top
            # of the cycle against a price that no longer existed.
            sweep_value = px if contracts > 0 else 1.0 - px
            sweep_gain = (sweep_value - avg_cost) / avg_cost

            # (i) the floor, now enforced per order rather than per cycle
            if sweep_gain < config.EXIT_COMMIT_FLOOR:
                _give_up(f"sweep {sweep+1}: gain {sweep_gain*100:+.0f}% fell below "
                         f"the {config.EXIT_COMMIT_FLOOR*100:+.0f}% floor mid-exit")
                break

            # (ii) a bound on how far price may run against us WITHIN one cycle.
            #
            # The within-cycle/across-cycle distinction is the whole point, and
            # getting it wrong breaks one fix or the other:
            #
            #   ACROSS cycles, a dip is normal and must be ridden. 0.62 -> 0.45
            #   over several cycles is a 17c move and is exactly the case the
            #   persistence fix exists to survive -- the floor governs there,
            #   and nothing else should.
            #
            #   WITHIN a cycle, sweeps are seconds apart. A large move over
            #   seconds is not a dip, it is the book repricing faster than we can
            #   react -- 0.47 -> 0.90 in six seconds on 2026-08-11. Firing the
            #   next order into that is chasing, not executing.
            #
            # So this measures against the value at the START OF THIS CYCLE, not
            # against the original commitment, and it only stops THIS cycle's
            # sweeps. It is not a give-up: next cycle re-reads the book, and if
            # the market has settled at a level that still clears the floor, the
            # exit carries on there. Only the floor ends a commitment for good.
            drift = cycle_ref_value - sweep_value
            if drift > config.EXIT_MAX_ADVERSE_MOVE:
                log(f"  exit sweep {sweep+1} {tkr}: price moved {drift*100:.0f}c "
                    f"within this cycle ({cycle_ref_value:.2f} -> {sweep_value:.2f}, "
                    f"limit {config.EXIT_MAX_ADVERSE_MOVE*100:.0f}c) -- pausing "
                    f"sweeps, will re-assess next cycle")
                break
            n = int(min(remaining, config.LIVE_MAX_CONTRACTS))
            r = ex.place_order(tkr, side, n, px, time_in_force="immediate_or_cancel",
                               note=f"phase3 exit sweep{sweep+1} att{state['attempts']}")
            got = float(r.get("fill_count") or 0)
            got_total += got
            remaining -= got
            log(f"  exit sweep {sweep+1}/{config.EXIT_SWEEPS_PER_CYCLE} {tkr}: "
                f"asked {n} @ {px:.4f} -> {r['status']} filled {got:.0f}, "
                f"{remaining:.0f} left")
            if got > 0:
                # P&L off the ACTUAL fill price, never the intended one, and in
                # the units of the position held -- both lessons paid for on
                # 2026-08-11, when a -$0.39 loss was reported as +$0.76.
                avg_fill = _f(r.get("avg_fill_price"), 0.0)
                if avg_fill > 0:
                    realised_value = avg_fill if contracts > 0 else 1.0 - avg_fill
                else:
                    realised_value = value_px
                fee = _f(r.get("fee_paid")) * got
                session["exit_fills"] = session.get("exit_fills", 0) + 1
                log(f"    realised {(realised_value - avg_cost) * got:+.4f} "
                    f"minus fee {fee:.4f}")
            if got == 0:
                break                      # no liquidity right now; wait a cycle
            time.sleep(config.EXIT_SWEEP_PAUSE_SECONDS)

        log(f"  exit attempt {state['attempts']} on {tkr}: filled {got_total:.0f} "
            f"of {abs(contracts):.0f}, {remaining:.0f} still open "
            f"({mins_left:.1f}min to close)")

    live_tickers = {p["ticker"] for p in positions if p["contracts"] != 0}
    for gone in [k for k in exiting if k not in live_tickers]:
        log(f"EXIT STATE CLEARED {gone} -- position no longer open on the exchange")
        exiting.pop(gone, None)
        session["exits"] = session.get("exits", 0) + 1

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
    # ONE ENTRY ATTEMPT PER WINDOW. Any prior entry order on this ticker, in ANY
    # terminal or in-flight state, closes the window for good.
    #
    # This deliberately blocks retries. The narrower set (filled/partial/resting/
    # sending/unknown) let a no_fill, rejected or cancelled attempt be retried on
    # the next cycle, and that really happened: KXBTC15M-26AUG110315-15 sent 3
    # entry orders and ...110330-30 sent 2. None of those produced a duplicate
    # position, because a no-filled IOC leaves nothing behind -- but "rejected"
    # is the dangerous one. An HTTP 503 that the exchange accepted before failing
    # to respond is recorded as rejected here while being live there; retrying
    # then mints a SECOND client_order_id and a second real position, and the
    # per-order size cap cannot see it. One 503 has already been observed
    # (2026-08-11, mid-exit), so this is a demonstrated failure mode, not a
    # hypothetical.
    #
    # The cost is real and accepted: an unlucky no-fill now forfeits the window
    # instead of retrying. Entries rest as GTC rather than firing IOC, so they
    # sit in the book waiting to be filled rather than failing outright, which is
    # what makes that cost affordable.
    #
    # Note this checks ENTRY orders only. Exit orders on the same ticker are
    # expected and must never look like "already entered".
    if any(r["ticker"] == market.ticker and "entry" in r.get("note", "")
           for r in ex.load_orders()):
        return session                      # this window has had its one attempt

    candles = fetch_1min_candles()
    spot = fetch_spot_price(candles)
    vol = compute_signals(candles).realized_vol_pct
    d = evaluate_trade(spot.price, market, realized_vol_pct=vol)
    pyes = f" p_yes={d.p_yes*100:.1f}%" if d.p_yes is not None else ""
    log(f"{market.ticker} {d.action}/{d.reason}{pyes}")
    if d.action != "enter":
        return session

    # Intended size, independently capped by place_order(). Deliberately NOT
    # PAPER_TRADE_SIZE -- that is the simulator's size and coupling live
    # sizing to it hid the fact that raising the cap changed nothing.
    n = min(config.LIVE_TRADE_SIZE, config.LIVE_MAX_CONTRACTS)

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
    # Re-read from THE SAME SOURCE evaluate_trade() priced off (get_active_market,
    # the /markets list endpoint). Using /markets/{ticker} here instead compared
    # two feeds that disagree materially and persistently -- measured at the same
    # instant: list bid/ask 0.22/0.23 (1c spread) vs single 0.11/0.37 (26c). That
    # made every evaluation look like the market had "moved" 7-15c, and this
    # guard then declined 5 of 7 valid signals for a phantom reason, so the bot
    # placed ZERO orders for over an hour. Consistency matters more here than
    # which feed is nominally fresher: the guard exists to catch movement between
    # evaluation and send, and it can only do that if both reads are comparable.
    fresh_market = get_active_market()
    if fresh_market is None or fresh_market.ticker != market.ticker:
        log(f"  {market.ticker} no longer the active market; skipping")
        return session
    if d.side == "yes":
        fresh = _f(fresh_market.yes_ask)
        side = "bid"
    else:
        fresh = 1.0 - _f(fresh_market.yes_bid)
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
    _cap = config.effective_loss_cap()
    print(f"daily loss cap      ${_cap:.2f}"
          + (f"  (standing ${abs(config.LIVE_DAILY_LOSS_CAP):.2f} + "
             f"${abs(config.LIVE_LOSS_CAP_OFFSET):.2f} offset expiring after "
             f"{config.LIVE_LOSS_CAP_OFFSET_DATE})"
             if _cap != abs(config.LIVE_DAILY_LOSS_CAP) else ""))
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
        f"size_cap={config.LIVE_MAX_CONTRACTS} "
        f"loss_cap=${config.effective_loss_cap():.2f} "
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
    # --status is READ-ONLY and must work while the trader is running -- that is
    # exactly when you want to inspect it. Only the trading loop takes the lock.
    if "--status" in sys.argv:
        sys.exit(main())
    try:
        with SingleInstance("live_trader"):
            sys.exit(main())
    except AlreadyRunning as exc:
        log(f"REFUSING TO START: {exc}")
        sys.exit(2)
