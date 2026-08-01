"""Kalshi 15-min BTC Intel Bot — main loop.

Evaluates the 7-indicator confluence every 1-min bar and pushes a Telegram
alert on:
  - market state change (e.g. RANGING -> UPTREND)
  - confidence band change (e.g. moderate -> high)
  - a new 15-min Kalshi window opening
Otherwise alerts are throttled to at most one per MIN_ALERT_GAP_SECONDS.

Usage:
  python bot.py           # run the loop
  python bot.py --once    # single evaluation + alert, then exit (smoke test)

This is an information/alert layer only. It places no orders and executes
no trades.
"""

import sys
import time
import traceback
from datetime import datetime, timezone

# Under pythonw.exe there is no console (stdout/stderr are None) — log to file.
# In a regular console, force UTF-8 so the alert emoji can print (cp1252 can't).
if sys.stdout is None or sys.stderr is None:
    from pathlib import Path
    _log = open(Path(__file__).parent / "bot.log", "a",
                encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _log
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import telegram_client
from alert_message import build_message
from classifier import classify
from coinbase_feed import fetch_1min_candles, latest_price
from indicators import compute_signals
from kalshi_feed import get_active_market
from strike_distance import evaluate as evaluate_strike


def evaluate_once():
    """One full evaluation pass; returns (message, state, market)."""
    candles = fetch_1min_candles()
    price = latest_price(candles)
    try:
        market = get_active_market()
    except Exception as e:  # Kalshi context is optional; degrade gracefully
        print(f"[kalshi] fetch failed, continuing without strike data: {e}",
              flush=True)
        market = None
    sig = compute_signals(candles)
    state = classify(sig)
    strike = evaluate_strike(price, market, sig, state)
    return build_message(price, market, sig, state, strike), state, market


def run_loop():
    print(f"[bot] starting — series={config.KALSHI_SERIES_TICKER}, "
          f"telegram={'configured' if telegram_client.configured() else 'DRY RUN'}",
          flush=True)
    last_state: str | None = None
    last_band: str | None = None
    last_event: str | None = None
    last_alert_ts = 0.0

    while True:
        try:
            message, state, market = evaluate_once()
            now = time.time()
            event = market.event_ticker if market else None

            state_changed = last_state is not None and state.state != last_state
            band_changed = last_band is not None and state.confidence_band != last_band
            new_window = last_event is not None and event is not None and event != last_event
            first_run = last_state is None
            throttled_ok = now - last_alert_ts >= config.MIN_ALERT_GAP_SECONDS

            should_alert = (
                first_run
                or state_changed
                or new_window
                or (band_changed and throttled_ok)
            )

            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            if should_alert:
                reason = ("startup" if first_run else
                          "state change" if state_changed else
                          "new window" if new_window else "confidence change")
                print(f"[{stamp}] ALERT ({reason}): {state.state} "
                      f"{state.confidence}/7", flush=True)
                if telegram_client.send(message):
                    last_alert_ts = now
            else:
                print(f"[{stamp}] {state.state} {state.confidence}/7 "
                      f"({state.confidence_band}) — no alert", flush=True)

            last_state, last_band, last_event = state.state, state.confidence_band, event
        except Exception:
            print("[bot] evaluation error:", flush=True)
            traceback.print_exc()

        # Sleep to just past the top of the next minute so the newest bar exists
        time.sleep(max(5.0, 60.0 - (time.time() % 60.0) + 2.0))


if __name__ == "__main__":
    if "--once" in sys.argv:
        message, state, market = evaluate_once()
        telegram_client.send(message)
    else:
        run_loop()
