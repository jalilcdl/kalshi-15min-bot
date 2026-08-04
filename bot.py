"""
Kalshi 15-min BTC Intel Bot -- Telegram "game plan" alerts.

=== INCIDENT (2026-08-03) -- read this before touching the alert logic ===
While debugging a missing-entry-alert report, I (the agent) ran manual test
scripts that wrote directly to data/trade_log.csv -- the SAME file this
already-running bot.py process polls every ~60s -- without ever pausing it
first. That caused two real, separate failures on the user's actual phone:
  1. A synthetic test row ("TEST-DOUBLE-RESOLVE") got picked up by the live
     process's check_target_hits() and sent as a real alert.
  2. A subsequent trade_log.csv reconstruction (fixing an earlier, unrelated
     mistake) suddenly introduced ~100 rows already marked exit_reason=
     "target_hit" that predated this alert code entirely (real historical
     paper trades from Aug 1-3, made before the alert feature existed). The
     already-running process's in-memory "already notified" set had no idea
     about them and fired ALL of them within about 18 seconds -- a
     spam/flood of ~70 messages, several showing absurd gains (+775%, etc.)
     because they were old, already-decided outcomes being reported as if
     they'd just happened.
Root cause: mutating shared state a live consumer polls, without pausing
the consumer -- an operational mistake, not a design flaw in isolation. But
it exposed three real design gaps, fixed below, that make a recurrence (from
ANY cause -- a crash, a long sleep/offline period, anything) structurally
harmless instead of another flood:
  - Alert dedup state was purely in-memory and re-primed from trade_log.csv's
    CURRENT state on every restart -- now persisted to data/bot_alert_state.json,
    so a restart never re-evaluates history that's already been handled.
  - Nothing checked whether an event was still ACTIONABLE (its window
    already closed) before alerting -- now every alert path checks staleness
    (ALERT_STALENESS_SECONDS) and silently marks old events handled instead
    of sending them. This is what would have prevented the flood even if the
    priming/persistence gap had reappeared some other way.
  - telegram_client.send()'s return value was never checked -- a transient
    network failure (confirmed happening periodically in this exact log,
    RemoteDisconnected errors) silently dropped an alert forever, since the
    ticker got marked "handled" regardless of whether delivery succeeded.
    Two real, otherwise-unexplained missing entry alerts during this incident
    match this exact mechanism. Now only marked handled on confirmed send.
Also added: a heartbeat log line every ~15 cycles, because this file
previously printed NOTHING during quiet periods -- impossible to tell "the
loop is alive and has nothing to report" from "the loop is dead" after the
fact, which is exactly the ambiguity that made this incident harder to
diagnose than it needed to be.
=== end incident notes ===

=== SECOND INCIDENT (2026-08-03, same day) -- read this too ===
A user screenshot flagged the check_new_entries() follow-up message as
"Trade confirmed" -- wording that reads like a real-order confirmation to
someone trading real cash off these alerts. It was always describing a
paper_trader.py SIMULATED fill, never a real one; the wording was just wrong.
Renamed build_trade_confirmed_alert() -> build_paper_entry_followup_alert()
in alert_message.py and rewrote the copy to say "PAPER trade (simulated, NOT
a real order)" explicitly -- never use "confirmed"/"executed" language for
anything that isn't a real, user-placed order.
Second question raised: this follow-up, by construction, only fires for
entries the ~60-120s check missed and paper_trader.py's continuous re-poll
caught later in the window -- i.e. always the late kind. research/exit_timing/
README.md §5c (same-day validated finding) shows target-hit rate declines
monotonically and significantly with later entry (78.6% at 1min down to
49.8% at 10min). That finding was scoped to "should the ALERT timing move
later" (answer: no, keep ~60-120s) -- it never examined whether
paper_trader.py's own continuous re-evaluation loop should stop opportunistically
entering deep into a window once the early check has already said no. No
code changes to that entry logic yet -- see the investigation reported back
to the user for the recommendation. The follow-up alert now at least surfaces
how late the entry landed so this is visible instead of hidden.
=== end second incident notes ===

Three alert types, all built from the validated distance+time+volatility
settlement model and the same fee/edge logic paper_trader.py actually trades
on (NOT the 7-indicator confluence engine or the old strike_distance.py
heuristic -- this project has since shown neither has validated edge; see
research/kalshi-btc-validation/ and research/strike_probability/README.md).

  1. Entry alert -- once per 15-min window, fired ~60-240s after it opens.
  2. Paper-entry follow-up (safety net) -- fires if paper_trader.py logs a
     SIMULATED trade later in the window that the early check didn't already
     flag. Never describes a real order.
  3. Target-hit nudge -- fired when a live paper position crosses the
     favorable-move exit target, sourced from trade_log.csv.

This is an information/alert layer only. It places no orders and executes
no trades.

Usage:
  python bot.py           # run the loop
  python bot.py --once    # single evaluation + alert pass, then exit (smoke test)
"""

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Under pythonw.exe there is no console (stdout/stderr are None) -- log to file.
# In a regular console, force UTF-8 so the alert emoji can print (cp1252 can't).
if sys.stdout is None or sys.stderr is None:
    _log = open(Path(__file__).parent / "bot.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _log
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import telegram_client
import trade_log
from alert_message import build_entry_alert, build_target_hit_alert, build_paper_entry_followup_alert
from coinbase_feed import fetch_1min_candles, latest_price
from kalshi_feed import get_active_market, get_market_by_ticker
from paper_trader import evaluate_trade

ENTRY_ALERT_MIN_AGE_SECONDS = 45   # don't fire before ~this point in the window
ENTRY_ALERT_MAX_AGE_SECONDS = 240  # if we first see a window this late (e.g. bot
                                   # just restarted), skip it silently rather than
                                   # send a jarringly-late "game plan"
ALERT_STALENESS_SECONDS = 16 * 60  # ~one window + buffer. An entry or target-hit
                                   # event older than this is no longer actionable
                                   # (the window's over or about to be) -- mark it
                                   # handled without sending, whatever caused the
                                   # delay. This is the guard that would have
                                   # prevented the 2026-08-03 flood on its own.
STATE_FILE = Path(__file__).parent / "data" / "bot_alert_state.json"
HEARTBEAT_EVERY_N_CYCLES = 15       # ~15 min at the normal 60s poll


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_ts(s):
    if not s or (isinstance(s, float) and str(s) == "nan"):
        return None
    try:
        dt = datetime.fromisoformat(str(s))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def prime_notified_target_hits() -> set[str]:
    """Mark every EXISTING target-hit trade as already notified. Only used
    on a genuine first run (no persisted state file yet)."""
    df = trade_log.load_log()
    if df.empty:
        return set()
    hit = df["exit_reason"] == "target_hit"
    return set(df.loc[hit.fillna(False), "market_ticker"])


def prime_notified_entries() -> set[str]:
    """Mark every EXISTING paper entry as already notified. Only used on a
    genuine first run (no persisted state file yet)."""
    df = trade_log.load_log()
    if df.empty:
        return set()
    return set(df.loc[df["mode"] == "paper", "market_ticker"])


def load_state():
    """Persisted across restarts on purpose -- see the incident notes above.
    Falls back to priming fresh from trade_log.csv only on a true first run."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return (data.get("alerted_entries", {}),
                    set(data.get("notified_new_entries", [])),
                    set(data.get("notified_target_hits", [])))
        except Exception as exc:
            print(f"[{_stamp()}] couldn't read {STATE_FILE} ({exc}) -- "
                  f"re-priming fresh from trade_log.csv", flush=True)
    return {}, prime_notified_entries(), prime_notified_target_hits()


def save_state(alerted_entries, notified_new_entries, notified_target_hits):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "alerted_entries": alerted_entries,
        "notified_new_entries": sorted(notified_new_entries),
        "notified_target_hits": sorted(notified_target_hits),
    }))


def check_target_hits(notified: set[str]) -> bool:
    """Returns True if `notified` changed (caller persists state when so)."""
    df = trade_log.load_log()
    if df.empty:
        return False
    hit_mask = (df["exit_reason"] == "target_hit").fillna(False)
    new_hits = df[hit_mask & ~df["market_ticker"].isin(notified)]
    now = datetime.now(timezone.utc)
    changed = False
    for row in new_hits.itertuples():
        exit_dt = _parse_ts(row.exit_time)
        if exit_dt and (now - exit_dt).total_seconds() > ALERT_STALENESS_SECONDS:
            age_min = (now - exit_dt).total_seconds() / 60.0
            print(f"[{_stamp()}] {row.market_ticker} target-hit is {age_min:.0f}min stale "
                  f"-- marking handled without alerting", flush=True)
            notified.add(row.market_ticker)
            changed = True
            continue

        entry_price = row.entry_price_cents / 100.0
        gain = (row.exit_price - entry_price) / entry_price
        msg = build_target_hit_alert(row.market_ticker, row.side, row.entry_price_cents,
                                     row.exit_price, gain)
        if telegram_client.send(msg):
            notified.add(row.market_ticker)
            changed = True
            print(f"[{_stamp()}] TARGET-HIT ALERT sent for {row.market_ticker} (+{gain*100:.0f}%)", flush=True)
        else:
            print(f"[{_stamp()}] TARGET-HIT ALERT send FAILED for {row.market_ticker} "
                  f"-- will retry next cycle", flush=True)
    return changed


def check_new_entries(alerted: dict[str, str], notified_entries: set[str]) -> bool:
    """Safety net: fires for any real paper entry (from trade_log.csv, the
    ground truth) that the early game-plan check didn't already announce as
    "enter". Skips tickers the early check already said "enter" for."""
    df = trade_log.load_log()
    if df.empty:
        return False
    paper = df[df["mode"] == "paper"]
    new_entries = paper[~paper["market_ticker"].isin(notified_entries)]
    now = datetime.now(timezone.utc)
    changed = False
    for row in new_entries.itertuples():
        if alerted.get(row.market_ticker) == "enter":
            notified_entries.add(row.market_ticker)
            changed = True
            continue  # already told them via the early game-plan alert

        entry_dt = _parse_ts(row.entry_time)
        if entry_dt and (now - entry_dt).total_seconds() > ALERT_STALENESS_SECONDS:
            age_min = (now - entry_dt).total_seconds() / 60.0
            print(f"[{_stamp()}] {row.market_ticker} entry is {age_min:.0f}min stale "
                  f"-- marking handled without alerting", flush=True)
            notified_entries.add(row.market_ticker)
            changed = True
            continue

        mins_into_window = None
        if entry_dt:
            try:
                market = get_market_by_ticker(row.market_ticker)
                if market is not None:
                    mins_into_window = (entry_dt - market.open_time).total_seconds() / 60.0
            except Exception:
                pass  # cosmetic only -- don't let a lookup failure block/retry a real alert

        msg = build_paper_entry_followup_alert(row.market_ticker, row.side, row.entry_price_cents,
                                               row.p_model, mins_into_window)
        if telegram_client.send(msg):
            notified_entries.add(row.market_ticker)
            changed = True
            print(f"[{_stamp()}] PAPER-ENTRY FOLLOW-UP sent for {row.market_ticker} "
                  f"(missed by the early check, ~{mins_into_window:.0f}m in)"
                  if mins_into_window is not None else
                  f"[{_stamp()}] PAPER-ENTRY FOLLOW-UP sent for {row.market_ticker} (missed by the early check)",
                  flush=True)
        else:
            print(f"[{_stamp()}] PAPER-ENTRY FOLLOW-UP send FAILED for {row.market_ticker} "
                  f"-- will retry next cycle", flush=True)
    return changed


def check_entry_alert(alerted: dict[str, str]) -> bool:
    """Returns True if `alerted` changed (caller persists state when so)."""
    market = get_active_market()
    if market is None or market.strike is None or market.ticker in alerted:
        return False
    elapsed = (datetime.now(timezone.utc) - market.open_time).total_seconds()
    if elapsed < ENTRY_ALERT_MIN_AGE_SECONDS:
        return False  # too early -- wait for a later cycle so we still catch it in range

    if elapsed > ENTRY_ALERT_MAX_AGE_SECONDS:
        alerted[market.ticker] = "late"
        print(f"[{_stamp()}] {market.ticker} already {elapsed:.0f}s old when first seen "
              f"-- skipping late entry alert", flush=True)
        return True

    candles = fetch_1min_candles()
    price = latest_price(candles)
    decision = evaluate_trade(price, market)
    msg = build_entry_alert(price, market, decision)
    if telegram_client.send(msg):
        alerted[market.ticker] = decision.action  # "enter" or "skip" -- read by check_new_entries()
        print(f"[{_stamp()}] ENTRY ALERT sent for {market.ticker} "
              f"({decision.action}/{decision.reason})", flush=True)
        return True
    print(f"[{_stamp()}] ENTRY ALERT send FAILED for {market.ticker} -- will retry next cycle", flush=True)
    return False


def run_loop():
    print(f"[bot] starting -- game-plan alerts, "
          f"telegram={'configured' if telegram_client.configured() else 'DRY RUN'}", flush=True)
    alerted_entries, notified_new_entries, notified_target_hits = load_state()
    print(f"[bot] loaded state: {len(alerted_entries)} alerted windows, "
          f"{len(notified_new_entries)} notified entries, "
          f"{len(notified_target_hits)} notified target-hits", flush=True)

    cycle = 0
    while True:
        cycle += 1
        changed = False
        try:
            changed |= check_entry_alert(alerted_entries)
            changed |= check_new_entries(alerted_entries, notified_new_entries)
            changed |= check_target_hits(notified_target_hits)
        except Exception:
            print(f"[{_stamp()}] evaluation error:", flush=True)
            traceback.print_exc()
        if changed:
            save_state(alerted_entries, notified_new_entries, notified_target_hits)
        if cycle % HEARTBEAT_EVERY_N_CYCLES == 0:
            print(f"[{_stamp()}] heartbeat -- cycle {cycle}, "
                  f"{len(notified_target_hits)} target-hits notified to date", flush=True)
        time.sleep(max(5.0, config.PAPER_TRADE_POLL_SECONDS - (time.time() % config.PAPER_TRADE_POLL_SECONDS) + 2.0))


if __name__ == "__main__":
    if "--once" in sys.argv:
        alerted, notified_entries, notified_hits = load_state()
        c1 = check_entry_alert(alerted)
        c2 = check_new_entries(alerted, notified_entries)
        c3 = check_target_hits(notified_hits)
        if c1 or c2 or c3:
            save_state(alerted, notified_entries, notified_hits)
    else:
        run_loop()
