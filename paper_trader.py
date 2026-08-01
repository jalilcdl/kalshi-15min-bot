"""
Paper-trading loop: simulates fills against REAL live Kalshi quotes (not the
model's own theoretical fair price) using the validated distance+time+
volatility strike-probability model, and logs every simulated trade to
data/trade_log.csv (mode="paper") so it accumulates and is viewable from the
dashboard's Trade log page, including on a phone.

This places NO real orders and touches no money -- it only decides "would I
have bought here" against the real quoted ask, and later checks the real
settlement, so the resulting P&L is a fee- and spread-aware forecast of what
a live version of this strategy would have earned, not a backtest against
theoretical prices.

Entry rule (see config.py for the tunable constants):
  - Compute p_yes from the fitted model (model/strike_probability.py).
  - Compare p_yes to the real YES ask, and (1-p_yes) to the real NO ask
    (derived from the YES book: no_ask = 1 - yes_bid).
  - Subtract the real Kalshi taker fee (fees.py) from each side's edge.
  - Only paper-enter if the best side's net edge clears PAPER_TRADE_MIN_EDGE.
  - At most one paper entry per market (checked against the existing log).
  - Skip the final FINAL_MINUTES_NOISY minutes of a window (settlement-print
    noise dominates there -- same guardrail strike_distance.py already uses).

Exit rule: this simulates the ACTUAL intended strategy -- buy early, sell
once the position has moved config.PAPER_TRADE_EXIT_TARGET in your favor,
rather than holding every position to settlement. Every cycle, each pending
row's real current exit value (the bid you could actually sell into, not
the ask) is checked against that target; if reached, the trade closes right
there (exit_reason="target_hit") at the real quoted price, fee included on
BOTH legs. If the target is never reached, the position rides to settlement
as before (exit_reason="settlement"). See research/exit_timing/README.md for
the walk-forward backtest this threshold and the decision to exit-early at
all are based on -- read it before assuming "sell early" is obviously better
than holding; the honest answer there is more nuanced than that.

Usage:
  python paper_trader.py --once     # one evaluation + one resolution pass, then exit
  python paper_trader.py            # runs forever, evaluating every POLL_SECONDS
"""
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Under pythonw.exe there is no console -- log to file. In a real console,
# force UTF-8 so nothing chokes on non-ASCII. Same pattern as bot.py.
if sys.stdout is None or sys.stderr is None:
    _log = open(Path(__file__).parent / "paper_trader.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _log
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import config
import trade_log
from coinbase_feed import fetch_1min_candles, latest_price
from fees import kalshi_fee
from indicators import compute_signals
from kalshi_feed import get_active_market, get_market_by_ticker
from model.strike_probability import compute_features, predict_p_yes


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def decide_trade(price: float, market) -> dict | None:
    """Returns a candidate trade dict, or None if no side clears the edge threshold."""
    mins_remaining = market.minutes_remaining()
    if mins_remaining <= config.FINAL_MINUTES_NOISY:
        return None
    if market.yes_bid is None or market.yes_ask is None:
        return None

    candles = fetch_1min_candles()
    sig = compute_signals(candles)

    # Guardrail: don't trust the model within noise-level distance of the strike --
    # see config.PAPER_TRADE_MIN_DIST_OVER_REACHABLE for why (caught by the smoke test).
    feats = compute_features(price, market.strike, mins_remaining, sig.realized_vol_pct)
    if feats["dist_over_reachable"] < config.PAPER_TRADE_MIN_DIST_OVER_REACHABLE:
        return None

    p_yes = predict_p_yes(price, market.strike, mins_remaining, sig.realized_vol_pct)

    no_ask = 1.0 - market.yes_bid
    size = config.PAPER_TRADE_SIZE
    fee_yes = kalshi_fee(size, market.yes_ask)
    fee_no = kalshi_fee(size, no_ask)
    edge_yes = p_yes - market.yes_ask - fee_yes / size
    edge_no = (1.0 - p_yes) - no_ask - fee_no / size

    if edge_yes >= config.PAPER_TRADE_MIN_EDGE and edge_yes >= edge_no:
        return dict(side="yes", entry_price=market.yes_ask, fee=fee_yes,
                    p_model=p_yes, edge=edge_yes, realized_vol=sig.realized_vol_pct)
    if edge_no >= config.PAPER_TRADE_MIN_EDGE:
        return dict(side="no", entry_price=no_ask, fee=fee_no,
                    p_model=1.0 - p_yes, edge=edge_no, realized_vol=sig.realized_vol_pct)
    return None


def already_entered(df: pd.DataFrame, ticker: str) -> bool:
    if df.empty:
        return False
    return bool(((df["market_ticker"] == ticker) & (df["mode"] == "paper")).any())


def log_paper_trade(price: float, market, trade: dict) -> None:
    df = trade_log.load_log()
    if already_entered(df, market.ticker):
        return
    size = config.PAPER_TRADE_SIZE
    cost = size * trade["entry_price"] + trade["fee"]
    now = datetime.now(timezone.utc)
    row = {
        "date": now.date().isoformat(),
        "entry_time": now.isoformat(),
        "market_ticker": market.ticker,
        "side": trade["side"],
        "entry_price_cents": round(trade["entry_price"] * 100, 2),
        "size": size,
        "model_read": (f"paper: p={trade['p_model']:.3f} edge={trade['edge']:+.3f} "
                        f"vol={trade['realized_vol']:.3f}%"),
        "btc_price_at_entry": price,
        "strike": market.strike,
        "result": "",
        "cost": round(cost, 4),
        "payout": "",
        "notes": "simulated fill vs real live quote -- see paper_trader.py",
        "mode": "paper",
        "p_model": round(trade["p_model"], 4),
        "fee": round(trade["fee"], 4),
    }
    raw = pd.read_csv(trade_log.TRADE_LOG_FILE) if trade_log.TRADE_LOG_FILE.exists() else pd.DataFrame(columns=trade_log.COLUMNS)
    raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
    raw.to_csv(trade_log.TRADE_LOG_FILE, index=False)
    print(f"[{_stamp()}] PAPER ENTRY {market.ticker} side={trade['side']} "
          f"price={trade['entry_price']:.2f} p_model={trade['p_model']:.3f} "
          f"edge={trade['edge']:+.3f} fee={trade['fee']:.3f}", flush=True)


def _exit_value(side: str, market) -> float | None:
    """What you could sell your position for RIGHT NOW -- the bid, not the ask,
    on the side you hold. For NO, that's the mirror of the YES book: no_bid = 1 - yes_ask."""
    if market.yes_bid is None or market.yes_ask is None:
        return None
    return market.yes_bid if side == "yes" else 1.0 - market.yes_ask


def resolve_pending() -> int:
    """Checks every pending paper trade against Kalshi: closes it early if the
    position has hit the favorable-move exit target (real quoted price, real
    fee on the exit too), else settles it if the window has closed, else
    leaves it pending. Returns the number resolved (either way) this pass."""
    if not trade_log.TRADE_LOG_FILE.exists():
        return 0
    df = pd.read_csv(trade_log.TRADE_LOG_FILE)
    if df.empty:
        return 0
    # An all-blank "result"/"payout"/etc. column round-trips through CSV as float64
    # (nothing but NaN to infer a dtype from), and pandas now refuses to silently
    # widen a float64 column to hold a string like "yes" via .loc assignment.
    # Force these to object dtype up front so resolving a trade doesn't crash.
    for c in ("result", "payout", "exit_time", "exit_price", "exit_fee", "exit_reason"):
        df[c] = df[c].astype("object")
    pending_mask = (df["mode"] == "paper") & (df["result"].isna() | (df["result"] == ""))
    pending = df[pending_mask]
    if pending.empty:
        return 0

    resolved = 0
    size = config.PAPER_TRADE_SIZE
    for idx, row in pending.iterrows():
        try:
            market = get_market_by_ticker(row["market_ticker"])
        except Exception as exc:
            print(f"[{_stamp()}] resolve lookup failed for {row['market_ticker']}: {exc}", flush=True)
            continue
        if market is None:
            continue

        # 1) Already settled -- hold-to-settlement path (target was never hit in time).
        if market.result in ("yes", "no"):
            win = (row["side"] == market.result)
            payout = size * 1.0 if win else 0.0
            df.loc[idx, "result"] = market.result
            df.loc[idx, "payout"] = payout
            df.loc[idx, "exit_reason"] = "settlement"
            resolved += 1
            outcome = "WIN" if win else "LOSS"
            print(f"[{_stamp()}] PAPER RESOLVED (settlement) {row['market_ticker']} -> "
                  f"{market.result.upper()} ({outcome}, payout=${payout:.2f})", flush=True)
            continue

        # 2) Still open -- check whether the favorable-move exit target has been hit.
        exit_val = _exit_value(row["side"], market)
        if exit_val is None:
            continue
        entry_price = row["entry_price_cents"] / 100.0
        gain = (exit_val - entry_price) / entry_price
        if gain >= config.PAPER_TRADE_EXIT_TARGET:
            exit_fee = kalshi_fee(size, exit_val)
            payout = size * exit_val - exit_fee
            now = datetime.now(timezone.utc)
            df.loc[idx, "exit_time"] = now.isoformat()
            df.loc[idx, "exit_price"] = round(exit_val, 4)
            df.loc[idx, "exit_fee"] = round(exit_fee, 4)
            df.loc[idx, "exit_reason"] = "target_hit"
            df.loc[idx, "payout"] = round(payout, 4)
            resolved += 1
            print(f"[{_stamp()}] PAPER SOLD EARLY {row['market_ticker']} side={row['side']} "
                  f"gain={gain:+.1%} exit_price={exit_val:.2f} payout=${payout:.2f}", flush=True)

    if resolved:
        df.to_csv(trade_log.TRADE_LOG_FILE, index=False)
    return resolved


def evaluate_once():
    market = get_active_market()
    if market is None or market.strike is None:
        print(f"[{_stamp()}] no active market with a published strike yet", flush=True)
        return
    candles = fetch_1min_candles()
    price = latest_price(candles)

    df = trade_log.load_log()
    if already_entered(df, market.ticker):
        print(f"[{_stamp()}] {market.ticker} already has a paper entry, skipping", flush=True)
        return

    trade = decide_trade(price, market)
    if trade is None:
        print(f"[{_stamp()}] {market.ticker} price=${price:,.2f} strike=${market.strike:,.2f} "
              f"-- no edge clears threshold, no trade", flush=True)
        return
    log_paper_trade(price, market, trade)


def run_loop():
    print(f"[paper_trader] starting -- size={config.PAPER_TRADE_SIZE} contracts, "
          f"min_edge={config.PAPER_TRADE_MIN_EDGE}, poll={config.PAPER_TRADE_POLL_SECONDS}s", flush=True)
    while True:
        try:
            resolve_pending()
            evaluate_once()
        except Exception:
            print(f"[{_stamp()}] evaluation error:", flush=True)
            traceback.print_exc()
        time.sleep(max(5.0, config.PAPER_TRADE_POLL_SECONDS - (time.time() % config.PAPER_TRADE_POLL_SECONDS) + 2.0))


if __name__ == "__main__":
    if "--once" in sys.argv:
        resolve_pending()
        evaluate_once()
    else:
        run_loop()
